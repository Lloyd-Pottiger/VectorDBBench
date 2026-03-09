import concurrent.futures
import io
import logging
from contextlib import contextmanager
from typing import Any

import pymysql

from ..api import VectorDB
from .config import TiDBIndexConfig

log = logging.getLogger(__name__)


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
        self.db_config = db_config
        self.case_config = db_case_config
        self.table_name = collection_name
        self.dim = dim
        self.conn = None  # To be inited by init()
        self.cursor = None  # To be inited by init()

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
        try:
            with self._get_connection() as (conn, cursor):
                cursor.execute(
                    f"""
                    CREATE TABLE {self.table_name} (
                        id BIGINT PRIMARY KEY,
                        embedding VECTOR({self.dim}) NOT NULL
                    );
                    """
                )
                conn.commit()
        except Exception as e:
            log.warning("Failed to create table: %s error: %s", self.table_name, e)
            raise

    def ready_to_load(self) -> bool:
        pass

    def optimize(self, data_size: int | None = None) -> None:
        if self.cursor is None or self.conn is None:
            raise RuntimeError("TiDB.optimize() must be called within `with self.init():`")

        # SPFRESH vector index is currently read-only: build the index only after all data is loaded.
        # TiDB blocks on this DDL until the index creation completes, so we do not need to poll TiFlash system tables.
        index_param = self.case_config.index_param()
        metric_fn = index_param["metric_fn"]
        metric_suffix = (
            self.case_config.metric_type.value.lower() if getattr(self.case_config, "metric_type", None) else "unknown"
        )
        index_name = f"idx_vec_{metric_suffix}"

        sql = (
            f"ALTER TABLE {self.table_name} "
            f"ADD VECTOR INDEX {index_name} (({metric_fn}(embedding))) USING SPFRESH"
        )
        log.info("Start building SPFRESH vector index with DDL: %s", sql)
        self.cursor.execute(sql)  # noqa: S608
        self.conn.commit()
        log.info("SPFRESH vector index build finished successfully.")

    def _insert_embeddings_serial(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        offset: int,
        size: int,
    ) -> Exception:
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
        except Exception as e:
            log.warning("Failed to insert data into table: %s", e)
            raise

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        **kwargs: Any,
    ) -> tuple[int, Exception]:
        workers = 10
        # Avoid exceeding MAX_ALLOWED_PACKET (default=64MB)
        max_batch_size = 64 * 1024 * 1024 // 24 // self.dim
        batch_size = len(embeddings) // workers
        batch_size = min(batch_size, max_batch_size)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = []
            for i in range(0, len(embeddings), batch_size):
                offset = i
                size = min(batch_size, len(embeddings) - i)
                future = executor.submit(self._insert_embeddings_serial, embeddings, metadata, offset, size)
                futures.append(future)
            done, pending = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_EXCEPTION)
            executor.shutdown(wait=False)
            for future in done:
                future.result()
            for future in pending:
                future.cancel()
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
