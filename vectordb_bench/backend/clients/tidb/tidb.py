import concurrent.futures
from dataclasses import dataclass
import io
import logging
import time
from contextlib import contextmanager
from typing import Any

import pymysql

from ..api import VectorDB
from .config import TiDBIndexConfig

log = logging.getLogger(__name__)

SPFRESH_OPTIMIZE_TIMEOUT_SECONDS = 600.0
SPFRESH_REGISTRATION_TIMEOUT_SECONDS = 30.0
SPFRESH_POLL_INTERVAL_SECONDS = 1.0
SPFRESH_INDEX_STATE_READY = "READY"
SPFRESH_INDEX_STATE_BUILDING = "BUILDING"
SPFRESH_INDEX_STATE_NEEDS_REBUILD = "NEEDS_REBUILD"
SPFRESH_INDEX_STATE_UNKNOWN = "UNKNOWN"
MAX_ALLOWED_PACKET_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class SPFreshIndexStatus:
    applied_base_ts: int | None
    index_state: str | None
    is_ready: bool
    lag_seconds: int | None
    owner_lease_expire_ts: int | None
    observed_at: Any


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
        try:
            with self._get_connection() as (conn, cursor):
                cursor.execute(f"DROP TABLE IF EXISTS {self.table_name}")
                conn.commit()
        except Exception as e:
            log.warning("Failed to drop table: %s error: %s", self.table_name, e)
            raise

    def _create_table(self):
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
        if self._max_insert_commit_ts is None:
            log.info("Skipping TiDB/SPFresh optimize wait because no insert commit_ts was recorded during load")
            return

        barrier_ts = self._max_insert_commit_ts
        index_name = self._spfresh_index_name()

        # For TiDB/SPFresh, optimize_duration is now the wait for async incremental catch-up, not DDL build time.
        log.info(
            "Waiting for TiDB/SPFresh async incremental catch-up: table=%s index=%s barrier_ts=%s",
            self.table_name,
            index_name,
            barrier_ts,
        )
        self._wait_for_spfresh_ready(barrier_ts=barrier_ts)
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

    def _fetch_spfresh_index_status(self) -> SPFreshIndexStatus | None:
        self.cursor.execute(
            """
            SELECT applied_base_ts, index_state, is_ready, lag_seconds, owner_lease_expire_ts, observed_at
            FROM information_schema.TIDB_SPFRESH_INDEX_STATUS
            WHERE table_schema = DATABASE() AND table_name = %s AND index_name = %s
            """,
            (self.table_name, self._spfresh_index_name()),
        )
        row = self.cursor.fetchone()
        if row is None:
            return None
        return SPFreshIndexStatus(
            applied_base_ts=None if row[0] is None else int(row[0]),
            index_state=None if row[1] is None else str(row[1]).upper(),
            is_ready=bool(row[2]),
            lag_seconds=None if row[3] is None else int(row[3]),
            owner_lease_expire_ts=None if row[4] is None else int(row[4]),
            observed_at=row[5],
        )

    def _wait_for_spfresh_ready(
        self,
        barrier_ts: int,
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
                    f"Timed out waiting for TiDB/SPFresh status catch-up to barrier_ts={barrier_ts}; "
                    f"last_observed={last_observed}"
                )
                raise RuntimeError(msg)

            status = self._fetch_spfresh_index_status()
            if status is None:
                if now > registration_deadline:
                    msg = (
                        f"TiDB/SPFresh status row never appeared for table={self.table_name} "
                        f"index={self._spfresh_index_name()} before barrier_ts={barrier_ts}"
                    )
                    raise RuntimeError(msg)
                time.sleep(poll_interval_seconds)
                continue

            last_observed = {
                "applied_base_ts": status.applied_base_ts,
                "index_state": status.index_state,
                "is_ready": int(status.is_ready),
                "lag_seconds": status.lag_seconds,
                "owner_lease_expire_ts": status.owner_lease_expire_ts,
                "observed_at": status.observed_at,
            }
            if status.index_state == SPFRESH_INDEX_STATE_UNKNOWN:
                msg = f"TiDB/SPFresh status became UNKNOWN while waiting for barrier_ts={barrier_ts}: {last_observed}"
                raise RuntimeError(msg)
            if status.index_state == SPFRESH_INDEX_STATE_NEEDS_REBUILD:
                msg = (
                    f"TiDB/SPFresh index entered NEEDS_REBUILD while waiting for barrier_ts={barrier_ts}: "
                    f"{last_observed}"
                )
                raise RuntimeError(msg)

            if status.applied_base_ts is not None and status.applied_base_ts >= barrier_ts and status.is_ready:
                log.info(
                    "TiDB/SPFresh status reached barrier_ts=%s applied_base_ts=%s is_ready=%s "
                    "lag_seconds=%s owner_lease_expire_ts=%s observed_at=%s index_state=%s",
                    barrier_ts,
                    status.applied_base_ts,
                    int(status.is_ready),
                    status.lag_seconds,
                    status.owner_lease_expire_ts,
                    status.observed_at,
                    status.index_state,
                )
                return

            log.info(
                "TiDB/SPFresh status catch-up pending: barrier_ts=%s applied_base_ts=%s "
                "is_ready=%s lag_seconds=%s owner_lease_expire_ts=%s observed_at=%s index_state=%s",
                barrier_ts,
                status.applied_base_ts,
                int(status.is_ready),
                status.lag_seconds,
                status.owner_lease_expire_ts,
                status.observed_at,
                status.index_state,
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
