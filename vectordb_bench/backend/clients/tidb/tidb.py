import concurrent.futures
import io
import json
import logging
import time
from contextlib import contextmanager
from typing import Any
from urllib.parse import urlsplit, urlunsplit
from urllib.request import urlopen

import pymysql

from ..api import VectorDB
from .config import TiDBIndexConfig

log = logging.getLogger(__name__)

SPFRESH_INCREMENTAL_LIFECYCLE_PATH = "/debug/spfresh/incremental_lifecycle"
SPFRESH_OPTIMIZE_TIMEOUT_SECONDS = 600.0
SPFRESH_REGISTRATION_TIMEOUT_SECONDS = 30.0
SPFRESH_POLL_INTERVAL_SECONDS = 1.0
SPFRESH_REBUILD_STATE_KEYWORDS = ("rebuild", "backfill", "snapshot", "bootstrap", "initial")
SPFRESH_FAILURE_STATE_KEYWORDS = ("fail", "error", "panic", "stopped")
MAX_ALLOWED_PACKET_BYTES = 64 * 1024 * 1024


class TiDB(VectorDB):
    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: TiDBIndexConfig,
        collection_name: str = "vector_bench_test",
        drop_old: bool = False,
        **kwargs,
    ):
        self.name = "TiDB"
        self.db_config = dict(db_config)
        self.case_config = db_case_config
        self.table_name = collection_name
        self.dim = dim
        self.conn = None  # To be inited by init()
        self.cursor = None  # To be inited by init()
        self.worker_debug_url = self.db_config.pop("worker_debug_url", None)
        self._cached_table_id: int | None = None
        self._cached_index_id: int | None = None
        self._max_insert_commit_ts: int | None = None

        self.search_fn = db_case_config.search_param()["metric_fn"]

        if drop_old:
            self._drop_table()
            self._create_table()

    @contextmanager
    def init(self):
        with self._get_connection() as (conn, cursor):
            self.conn = conn
            self.cursor = cursor
            try:
                yield
            finally:
                self.conn = None
                self.cursor = None

    @contextmanager
    def _get_connection(self):
        with pymysql.connect(**self.db_config) as conn:
            conn.autocommit = False
            with conn.cursor() as cursor:
                yield conn, cursor

    def _drop_table(self):
        self._cached_table_id = None
        self._cached_index_id = None
        try:
            with self._get_connection() as (conn, cursor):
                cursor.execute(f"DROP TABLE IF EXISTS {self.table_name}")
                conn.commit()
        except Exception as e:
            log.warning("Failed to drop table: %s error: %s", self.table_name, e)
            raise

    def _create_table(self):
        self._cached_table_id = None
        self._cached_index_id = None
        index_name = self._spfresh_index_name()
        metric_fn = self.case_config.index_param()["metric_fn"]
        try:
            with self._get_connection() as (conn, cursor):
                cursor.execute(f"""
                    CREATE TABLE {self.table_name} (
                        id BIGINT PRIMARY KEY,
                        embedding VECTOR({self.dim}) NOT NULL,
                        VECTOR INDEX {index_name} (({metric_fn}(embedding))) USING SPFRESH
                    );
                    """)
                conn.commit()
        except Exception as e:
            log.warning("Failed to create table: %s error: %s", self.table_name, e)
            raise

    def ready_to_load(self) -> bool:
        pass

    def optimize(self, data_size: int | None = None) -> None:
        if self.cursor is None or self.conn is None:
            raise RuntimeError("TiDB.optimize() must be called within `with self.init():`")
        if not self.worker_debug_url:
            raise RuntimeError(
                "TiDB/SPFresh optimize requires --worker-debug-url because optimize_duration now waits "
                "for async incremental catch-up instead of running ALTER TABLE ADD VECTOR INDEX.",
            )
        if self._max_insert_commit_ts is None:
            log.info("Skipping TiDB/SPFresh optimize wait because no insert commit_ts was recorded during load")
            return

        barrier_ts = self._max_insert_commit_ts
        table_id = self._resolve_table_id()
        index_id = self._resolve_index_id()

        # For TiDB/SPFresh, optimize_duration is now the wait for async incremental catch-up, not DDL build time.
        log.info(
            "Waiting for TiDB/SPFresh async incremental catch-up: table=%s index=%s barrier_ts=%s",
            table_id,
            index_id,
            barrier_ts,
        )
        self._wait_for_incremental_catch_up(
            barrier_ts=barrier_ts,
            table_id=table_id,
            index_id=index_id,
        )
        self._max_insert_commit_ts = None

    def export_load_state(self) -> dict[str, int] | None:
        if self._max_insert_commit_ts is None:
            return None
        return {"max_insert_commit_ts": self._max_insert_commit_ts}

    def import_load_state(self, state: dict[str, int] | None) -> None:
        if not state:
            self._max_insert_commit_ts = None
            return
        self._max_insert_commit_ts = int(state["max_insert_commit_ts"])

    def _spfresh_index_name(self) -> str:
        metric_suffix = (
            getattr(self.case_config.metric_type, "value", str(self.case_config.metric_type)).lower()
            if getattr(self.case_config, "metric_type", None)
            else "unknown"
        )
        return f"idx_embedding_spfresh_{metric_suffix}"

    def _last_commit_ts(self, cursor: Any) -> int:
        cursor.execute("SELECT json_extract(@@tidb_last_txn_info, '$.commit_ts')")
        row = cursor.fetchone()
        if row is None or row[0] is None:
            msg = "Failed to read TiDB commit_ts from @@tidb_last_txn_info"
            raise RuntimeError(msg)
        return int(row[0])

    def _resolve_table_id(self) -> int:
        if self._cached_table_id is not None:
            return self._cached_table_id
        self.cursor.execute(
            """
            SELECT tidb_table_id
            FROM information_schema.tables
            WHERE table_schema = DATABASE() AND table_name = %s
            """,
            (self.table_name,),
        )
        row = self.cursor.fetchone()
        if row is None:
            msg = f"Failed to resolve table_id for TiDB table {self.table_name}"
            raise RuntimeError(msg)
        self._cached_table_id = int(row[0])
        return self._cached_table_id

    def _resolve_index_id(self) -> int:
        if self._cached_index_id is not None:
            return self._cached_index_id
        self.cursor.execute(
            """
            SELECT index_id
            FROM information_schema.tidb_indexes
            WHERE table_schema = DATABASE() AND table_name = %s AND key_name = %s
            """,
            (self.table_name, self._spfresh_index_name()),
        )
        row = self.cursor.fetchone()
        if row is None:
            msg = (
                f"Failed to resolve index_id for TiDB/SPFresh index {self._spfresh_index_name()} "
                f"on {self.table_name}"
            )
            raise RuntimeError(msg)
        self._cached_index_id = int(row[0])
        return self._cached_index_id

    def _worker_lifecycle_url(self) -> str:
        parsed = urlsplit(self.worker_debug_url)
        if parsed.scheme not in {"http", "https"}:
            msg = f"Invalid TiDB worker debug URL: {self.worker_debug_url}"
            raise RuntimeError(msg)
        if parsed.path.endswith(SPFRESH_INCREMENTAL_LIFECYCLE_PATH):
            return self.worker_debug_url
        path = parsed.path.rstrip("/") + SPFRESH_INCREMENTAL_LIFECYCLE_PATH
        return urlunsplit((parsed.scheme, parsed.netloc, path, parsed.query, parsed.fragment))

    def _fetch_incremental_lifecycle(self) -> list[dict[str, Any]]:
        with urlopen(self._worker_lifecycle_url(), timeout=5) as response:  # noqa: S310
            payload = json.loads(response.read().decode("utf-8"))

        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            if all(key in payload for key in ("table_id", "index_id", "applied_base_ts")):
                return [payload]
            for value in payload.values():
                if isinstance(value, list):
                    return value
        raise RuntimeError("Unexpected TiDB/SPFresh incremental lifecycle payload shape")

    def _matching_incremental_lifecycle(
        self,
        rows: list[dict[str, Any]],
        table_id: int,
        index_id: int,
    ) -> dict[str, Any] | None:
        return next(
            (
                row
                for row in rows
                if int(row.get("table_id", -1)) == table_id and int(row.get("index_id", -1)) == index_id
            ),
            None,
        )

    def _wait_for_incremental_catch_up(
        self,
        barrier_ts: int,
        table_id: int,
        index_id: int,
        timeout_seconds: float = SPFRESH_OPTIMIZE_TIMEOUT_SECONDS,
        registration_timeout_seconds: float = SPFRESH_REGISTRATION_TIMEOUT_SECONDS,
        poll_interval_seconds: float = SPFRESH_POLL_INTERVAL_SECONDS,
    ) -> None:
        deadline = time.monotonic() + timeout_seconds
        registration_deadline = time.monotonic() + min(timeout_seconds, registration_timeout_seconds)
        last_observed = None

        while True:
            now = time.monotonic()
            if now > deadline:
                msg = (
                    f"Timed out waiting for TiDB/SPFresh incremental catch-up to barrier_ts={barrier_ts}; "
                    f"last_observed={last_observed}"
                )
                raise RuntimeError(msg)

            rows = self._fetch_incremental_lifecycle()
            lifecycle = self._matching_incremental_lifecycle(rows, table_id, index_id)
            if lifecycle is None:
                if now > registration_deadline:
                    msg = (
                        f"TiDB/SPFresh worker never registered table_id={table_id} index_id={index_id} "
                        f"before barrier_ts={barrier_ts}"
                    )
                    raise RuntimeError(msg)
                time.sleep(poll_interval_seconds)
                continue

            last_observed = {
                "applied_base_ts": lifecycle.get("applied_base_ts"),
                "global_resolved_ts": lifecycle.get("global_resolved_ts"),
                "buffered_events": lifecycle.get("buffered_events"),
                "lag_seconds": lifecycle.get("lag_seconds"),
                "runtime_state": lifecycle.get("runtime_state"),
                "last_error": lifecycle.get("last_error"),
            }
            runtime_state = str(lifecycle.get("runtime_state", "")).lower()
            last_error = lifecycle.get("last_error")

            if last_error:
                msg = (
                    f"TiDB/SPFresh worker reported last_error while waiting for barrier_ts={barrier_ts}: "
                    f"{last_error}"
                )
                raise RuntimeError(msg)
            if any(keyword in runtime_state for keyword in SPFRESH_FAILURE_STATE_KEYWORDS):
                msg = (
                    f"TiDB/SPFresh worker entered failure runtime_state={lifecycle.get('runtime_state')} "
                    f"while waiting for barrier_ts={barrier_ts}"
                )
                raise RuntimeError(msg)
            if any(keyword in runtime_state for keyword in SPFRESH_REBUILD_STATE_KEYWORDS):
                msg = (
                    f"TiDB/SPFresh worker entered rebuild-like runtime_state={lifecycle.get('runtime_state')} "
                    f"while waiting for barrier_ts={barrier_ts}"
                )
                raise RuntimeError(msg)

            applied_base_ts = int(lifecycle.get("applied_base_ts", 0))
            global_resolved_ts_value = lifecycle.get("global_resolved_ts")
            buffered_events_value = lifecycle.get("buffered_events")
            global_resolved_ts = int(global_resolved_ts_value or 0)
            buffered_events = int(buffered_events_value or 0)
            # AppliedBaseTs is the authoritative read visibility watermark. If it has crossed
            # the barrier, every earlier write is already visible through SPFresh search.
            if applied_base_ts >= barrier_ts:
                log.info(
                    "TiDB/SPFresh async incremental catch-up reached barrier_ts=%s via applied_base_ts=%s "
                    "global_resolved_ts=%s buffered_events=%s lag_seconds=%s runtime_state=%s",
                    barrier_ts,
                    applied_base_ts,
                    global_resolved_ts_value,
                    buffered_events_value,
                    lifecycle.get("lag_seconds"),
                    lifecycle.get("runtime_state"),
                )
                return
            # Newer worker debug endpoints expose per-table resolved_ts and buffered event counts.
            # When available, they can prove visibility slightly earlier than AppliedBaseTs alone.
            if global_resolved_ts_value is not None and buffered_events_value is not None and global_resolved_ts >= barrier_ts and buffered_events == 0:
                log.info(
                    "TiDB/SPFresh async incremental catch-up reached barrier_ts=%s applied_base_ts=%s "
                    "global_resolved_ts=%s buffered_events=%s lag_seconds=%s runtime_state=%s",
                    barrier_ts,
                    applied_base_ts,
                    global_resolved_ts_value,
                    buffered_events_value,
                    lifecycle.get("lag_seconds"),
                    lifecycle.get("runtime_state"),
                )
                return

            log.info(
                "TiDB/SPFresh async incremental catch-up pending: barrier_ts=%s applied_base_ts=%s "
                "global_resolved_ts=%s buffered_events=%s lag_seconds=%s runtime_state=%s",
                barrier_ts,
                applied_base_ts,
                global_resolved_ts,
                buffered_events,
                lifecycle.get("lag_seconds"),
                lifecycle.get("runtime_state"),
            )
            time.sleep(poll_interval_seconds)

    def _insert_embeddings_serial(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        offset: int,
        size: int,
    ) -> int:
        try:
            with self._get_connection() as (conn, cursor):
                buf = io.StringIO()
                buf.write(f"INSERT INTO {self.table_name} (id, embedding) VALUES ")  # noqa: S608
                for i in range(offset, offset + size):
                    if i > offset:
                        buf.write(",")
                    buf.write(f'({metadata[i]}, "{embeddings[i]!s}")')
                cursor.execute(buf.getvalue())
                conn.commit()
                return self._last_commit_ts(cursor)
        except Exception as e:
            log.warning("Failed to insert data into table: %s", e)
            raise

    def _max_insert_rows_per_transaction(self) -> int:
        return max(1, MAX_ALLOWED_PACKET_BYTES // 24 // self.dim)

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        **kwargs: Any,
    ) -> tuple[int, Exception]:
        workers = 10
        batch_size = max(1, len(embeddings) // workers)
        batch_size = min(batch_size, self._max_insert_rows_per_transaction())
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = []
            for i in range(0, len(embeddings), batch_size):
                offset = i
                size = min(batch_size, len(embeddings) - i)
                future = executor.submit(self._insert_embeddings_serial, embeddings, metadata, offset, size)
                futures.append(future)
            done, pending = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_EXCEPTION)
            executor.shutdown(wait=False)
            max_commit_ts = self._max_insert_commit_ts
            for future in done:
                commit_ts = future.result()
                if max_commit_ts is None or commit_ts > max_commit_ts:
                    max_commit_ts = commit_ts
            for future in pending:
                future.cancel()
        self._max_insert_commit_ts = max_commit_ts
        return len(metadata), None

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,
        timeout: int | None = None,
        **kwargs: Any,
    ) -> list[int]:
        sql = f"""
            SELECT id FROM {self.table_name}
            ORDER BY {self.search_fn}(embedding, "{query!s}") LIMIT {k};
            """  # noqa: S608
        self.cursor.execute(sql)
        result = self.cursor.fetchall()
        return [int(i[0]) for i in result]
