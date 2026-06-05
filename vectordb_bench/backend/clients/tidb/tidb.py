import concurrent.futures
import io
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import pymysql

from ....metric import OptimizeResult
from ..api import VectorDB
from .config import SPFreshBuildMode, TiDBIndexConfig

log = logging.getLogger(__name__)

SPFRESH_OPTIMIZE_TIMEOUT_SECONDS = 600.0
SPFRESH_REGISTRATION_TIMEOUT_SECONDS = 30.0
SPFRESH_POLL_INTERVAL_SECONDS = 1.0
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
        self._max_delete_commit_ts: int | None = None

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
        index_sql = ""
        if self._should_inline_spfresh_index():
            index_sql = f",\n                        {self._spfresh_index_definition()}"

        try:
            with self._get_connection() as (conn, cursor):
                cursor.execute(
                    f"""
                    CREATE TABLE {self.table_name} (
                        id BIGINT PRIMARY KEY,
                        embedding VECTOR({self.dim}) NOT NULL
                        {index_sql}
                    );
                    """
                )
                conn.commit()
        except Exception as e:
            log.warning("Failed to create table: %s error: %s", self.table_name, e)
            raise

    def ready_to_load(self) -> bool:
        return True

    def optimize(self, data_size: int | None = None) -> OptimizeResult:
        if self.cursor is None or self.conn is None:
            raise RuntimeError("TiDB.optimize() must be called within `with self.init():`")
        if not self.case_config.build_spfresh_index:
            log.info("Skipping TiDB/SPFresh optimize wait because SPFRESH index creation is disabled")
            return OptimizeResult()

        if self.case_config.spfresh_build_mode in (SPFreshBuildMode.NON_INLINE, SPFreshBuildMode.SPLIT):
            return self.build_spfresh_index()

        return self.wait_spfresh_incremental_catchup()

    def build_spfresh_index(self) -> OptimizeResult:
        if self.cursor is None or self.conn is None:
            raise RuntimeError("TiDB.build_spfresh_index() must be called within `with self.init():`")
        if not self.case_config.build_spfresh_index:
            log.info("Skipping TiDB/SPFresh index build because SPFRESH index creation is disabled")
            return OptimizeResult()

        log.info(
            "Building TiDB/SPFresh index: table=%s index=%s",
            self.table_name,
            self._spfresh_index_name(),
        )
        start = time.perf_counter()
        self._create_spfresh_index()
        build_duration = time.perf_counter() - start
        return OptimizeResult(
            optimize_duration=build_duration,
            spfresh_build_duration=build_duration,
        )

    def wait_spfresh_incremental_catchup(self) -> OptimizeResult:
        if self.cursor is None or self.conn is None:
            raise RuntimeError("TiDB.wait_spfresh_incremental_catchup() must be called within `with self.init():`")
        if not self.case_config.build_spfresh_index:
            log.info("Skipping TiDB/SPFresh catch-up wait because SPFRESH index creation is disabled")
            return OptimizeResult()

        barrier_ts = self._max_insert_commit_ts
        if barrier_ts is None:
            barrier_ts = self._current_tso_marker()
        index_name = self._spfresh_index_name()

        log.info(
            "Waiting for TiDB/SPFresh async incremental catch-up: table=%s index=%s barrier_ts=%s",
            self.table_name,
            index_name,
            barrier_ts,
        )
        start = time.perf_counter()
        self._wait_for_spfresh_ready(barrier_ts=barrier_ts)
        catchup_duration = time.perf_counter() - start
        self._max_insert_commit_ts = None
        return OptimizeResult(
            optimize_duration=catchup_duration,
            spfresh_incremental_catchup_duration=catchup_duration,
        )

    def export_load_state(self) -> dict[str, int] | None:
        if self._max_insert_commit_ts is None:
            return None
        return {"max_insert_commit_ts": self._max_insert_commit_ts}

    def import_load_state(self, state: dict[str, int] | None) -> None:
        if not state:
            self._max_insert_commit_ts = None
            return
        self._max_insert_commit_ts = int(state["max_insert_commit_ts"])

    def export_delete_state(self) -> dict[str, int] | None:
        if self._max_delete_commit_ts is None:
            return None
        return {"max_delete_commit_ts": self._max_delete_commit_ts}

    def import_delete_state(self, state: dict[str, int] | None) -> None:
        if not state:
            self._max_delete_commit_ts = None
            return
        self._max_delete_commit_ts = int(state["max_delete_commit_ts"])

    def _spfresh_index_name(self) -> str:
        metric_suffix = (
            getattr(self.case_config.metric_type, "value", str(self.case_config.metric_type)).lower()
            if getattr(self.case_config, "metric_type", None)
            else "unknown"
        )
        return f"idx_embedding_spfresh_{metric_suffix}"

    def _should_inline_spfresh_index(self) -> bool:
        return self.case_config.build_spfresh_index and self.case_config.spfresh_build_mode == SPFreshBuildMode.INLINE

    def _spfresh_index_definition(self) -> str:
        metric_fn = self.case_config.index_param()["metric_fn"]
        return f"VECTOR INDEX {self._spfresh_index_name()} (({metric_fn}(embedding))) USING SPFRESH"

    def _create_spfresh_index(self) -> None:
        self.cursor.execute(f"ALTER TABLE {self.table_name} ADD {self._spfresh_index_definition()}")
        self.conn.commit()

    def _last_commit_ts(self, cursor: Any) -> int:
        cursor.execute("SELECT json_extract(@@tidb_last_txn_info, '$.commit_ts')")
        row = cursor.fetchone()
        if row is None or row[0] is None:
            msg = "Failed to read TiDB commit_ts from @@tidb_last_txn_info"
            raise RuntimeError(msg)
        return int(row[0])

    def _current_tso_marker(self) -> int:
        self.cursor.execute("BEGIN")
        self.cursor.execute("SELECT @@tidb_current_ts")
        row = self.cursor.fetchone()
        self.conn.commit()
        if row is None or row[0] is None:
            msg = "Failed to read TiDB current TSO marker from @@tidb_current_ts"
            raise RuntimeError(msg)
        return int(row[0])

    @staticmethod
    def _max_commit_ts(current: int | None, candidate: int | None) -> int | None:
        if candidate is None:
            return current
        if current is None or candidate > current:
            return candidate
        return current

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

    def wait_spfresh_post_delete_catchup(
        self,
        timeout: float,
        poll_interval: float,
    ) -> int | None:
        if not self.case_config.build_spfresh_index:
            log.info("Skipping post-delete SPFRESH catch-up because SPFRESH index creation is disabled")
            return None
        if self._max_delete_commit_ts is None:
            log.info("Skipping post-delete SPFRESH catch-up because no delete commit_ts was recorded")
            return None

        barrier_ts = self._max_delete_commit_ts
        log.info(
            "Waiting for post-delete TiDB/SPFresh catch-up: table=%s index=%s barrier_ts=%s",
            self.table_name,
            self._spfresh_index_name(),
            barrier_ts,
        )

        if self.cursor is not None:
            self._wait_for_spfresh_ready(
                barrier_ts=barrier_ts,
                timeout_seconds=timeout,
                poll_interval_seconds=poll_interval,
            )
        else:
            with self.init():
                self._wait_for_spfresh_ready(
                    barrier_ts=barrier_ts,
                    timeout_seconds=timeout,
                    poll_interval_seconds=poll_interval,
                )

        self._max_delete_commit_ts = None
        return barrier_ts

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
                max_commit_ts = self._max_commit_ts(max_commit_ts, commit_ts)
            for future in pending:
                future.cancel()
        self._max_insert_commit_ts = max_commit_ts
        return len(metadata), None

    def delete_embeddings(
        self,
        ids: list[int],
        **kwargs: Any,
    ) -> tuple[int, Exception | None]:
        if self.cursor is None or self.conn is None:
            raise RuntimeError("TiDB.delete_embeddings() must be called within `with self.init():`")

        deleted = 0
        delete_sql = f"DELETE FROM {self.table_name} WHERE id = %s"  # noqa: S608
        commit_interval = self.case_config.delete_commit_interval
        max_commit_ts = self._max_delete_commit_ts

        try:
            for idx, row_id in enumerate(ids, start=1):
                deleted += self.cursor.execute(delete_sql, (row_id,))
                if idx % commit_interval == 0:
                    self.conn.commit()
                    max_commit_ts = self._max_commit_ts(max_commit_ts, self._last_commit_ts(self.cursor))

            if len(ids) % commit_interval != 0:
                self.conn.commit()
                max_commit_ts = self._max_commit_ts(max_commit_ts, self._last_commit_ts(self.cursor))
        except Exception as e:
            self.conn.rollback()
            log.warning("Failed to delete data from table: %s", e)
            raise

        self._max_delete_commit_ts = max_commit_ts
        return deleted, None

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
