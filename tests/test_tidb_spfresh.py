from types import SimpleNamespace, TracebackType
from typing import Any
from unittest.mock import patch

import pytest

from vectordb_bench.backend.clients.api import MetricType
from vectordb_bench.backend.clients.tidb.config import TiDBIndexConfig
from vectordb_bench.backend.clients.tidb.tidb import (
    MAX_ALLOWED_PACKET_BYTES,
    TiDB,
)
from vectordb_bench.backend.task_runner import CaseRunner


class FakeCursor:
    def __init__(self, fetchone_results: list[tuple[int]] | None = None):
        self.execute_calls: list[tuple[str, Any]] = []
        self.fetchone_results = list(fetchone_results or [])

    def execute(self, sql: str, params: Any = None):
        self.execute_calls.append((sql, params))

    def fetchone(self):
        if not self.fetchone_results:
            return None
        return self.fetchone_results.pop(0)


class FakeConnection:
    def __init__(self):
        self.committed = False

    def commit(self):
        self.committed = True


def make_tidb(worker_debug_url: str = "http://worker:5678", dim: int = 3) -> TiDB:
    return TiDB(
        dim=dim,
        db_config={
            "host": "127.0.0.1",
            "port": 4000,
            "user": "root",
            "password": "",
            "database": "test",
            "ssl_verify_cert": False,
            "ssl_verify_identity": False,
            "worker_debug_url": worker_debug_url,
        },
        db_case_config=TiDBIndexConfig(metric_type=MetricType.L2),
    )


class TestTiDBSPFresh:

    def test_insert_embeddings_keeps_existing_worker_sharding_when_batches_are_small(self):
        tidb = make_tidb()
        embeddings = [[] for _ in range(200)]
        metadata = list(range(200))
        insert_calls: list[tuple[int, int]] = []

        def capture_insert_call(
            _embeddings: list[list[float]],
            _metadata: list[int],
            offset: int,
            size: int,
        ) -> None:
            insert_calls.append((offset, size))

        with patch.object(tidb, "_insert_embeddings_serial", side_effect=capture_insert_call):
            insert_count, error = tidb.insert_embeddings(embeddings=embeddings, metadata=metadata)

        assert (insert_count, error) == (len(metadata), None)
        assert sorted(insert_calls) == [(offset, 20) for offset in range(0, len(metadata), 20)]

    def test_insert_embeddings_keeps_existing_worker_sharding_for_larger_loads(self):
        tidb = make_tidb()
        embeddings = [[] for _ in range(6000)]
        metadata = list(range(6000))
        insert_calls: list[tuple[int, int]] = []

        def capture_insert_call(
            _embeddings: list[list[float]],
            _metadata: list[int],
            offset: int,
            size: int,
        ) -> None:
            insert_calls.append((offset, size))

        with patch.object(tidb, "_insert_embeddings_serial", side_effect=capture_insert_call):
            insert_count, error = tidb.insert_embeddings(embeddings=embeddings, metadata=metadata)

        assert (insert_count, error) == (len(metadata), None)
        assert sorted(insert_calls) == [(offset, 600) for offset in range(0, len(metadata), 600)]

    def test_insert_embeddings_caps_transactions_by_max_allowed_packet_bytes(self):
        tidb = make_tidb(dim=30_000)
        embeddings = [[] for _ in range(1000)]
        metadata = list(range(1000))
        insert_calls: list[tuple[int, int]] = []
        expected_batch_size = MAX_ALLOWED_PACKET_BYTES // 24 // tidb.dim

        def capture_insert_call(
            _embeddings: list[list[float]],
            _metadata: list[int],
            offset: int,
            size: int,
        ) -> None:
            insert_calls.append((offset, size))

        with patch.object(tidb, "_insert_embeddings_serial", side_effect=capture_insert_call):
            insert_count, error = tidb.insert_embeddings(embeddings=embeddings, metadata=metadata)

        assert expected_batch_size < len(metadata) // 10
        assert (insert_count, error) == (len(metadata), None)
        assert sorted(insert_calls) == [
            (offset, min(expected_batch_size, len(metadata) - offset))
            for offset in range(0, len(metadata), expected_batch_size)
        ]

    def test_insert_embeddings_tracks_max_commit_ts_across_worker_sessions(self):
        tidb = make_tidb()
        embeddings = [[] for _ in range(6000)]
        metadata = list(range(6000))

        def capture_insert_call(
            _embeddings: list[list[float]],
            _metadata: list[int],
            offset: int,
            _size: int,
        ) -> int:
            return 1000 + offset

        with patch.object(tidb, "_insert_embeddings_serial", side_effect=capture_insert_call):
            insert_count, error = tidb.insert_embeddings(embeddings=embeddings, metadata=metadata)

        assert (insert_count, error) == (len(metadata), None)
        assert tidb._max_insert_commit_ts == 6400

    def test_insert_embeddings_does_not_probe_incremental_backlog_during_load(self):
        tidb = make_tidb()
        tidb.cursor = FakeCursor()
        tidb.conn = FakeConnection()
        batch_size = tidb._max_insert_rows_per_transaction()
        total_rows = batch_size * 20

        with (
            patch.object(tidb, "_insert_embeddings_serial", return_value=None),
            patch.object(tidb, "_resolve_table_id", side_effect=AssertionError("should not probe table id")),
            patch.object(tidb, "_resolve_index_id", side_effect=AssertionError("should not probe index id")),
            patch.object(
                tidb,
                "_fetch_incremental_lifecycle",
                side_effect=AssertionError("should not fetch incremental lifecycle during load"),
            ),
        ):
            insert_count, error = tidb.insert_embeddings(
                embeddings=[[] for _ in range(total_rows)],
                metadata=list(range(total_rows)),
            )

        assert (insert_count, error) == (total_rows, None)

    def test_insert_embeddings_serial_reads_commit_ts_from_writer_session(self):
        tidb = make_tidb()
        cursor = FakeCursor(fetchone_results=[("123456",)])
        conn = FakeConnection()

        class ConnectionContext:
            def __enter__(self_inner) -> tuple[FakeConnection, FakeCursor]:
                return conn, cursor

            def __exit__(
                self_inner,
                exc_type: type[BaseException] | None,
                exc: BaseException | None,
                tb: TracebackType | None,
            ) -> bool:
                return False

        with patch.object(tidb, "_get_connection", return_value=ConnectionContext()):
            commit_ts = tidb._insert_embeddings_serial(embeddings=[[1.0, 2.0, 3.0]], metadata=[7], offset=0, size=1)

        assert commit_ts == 123456

    def test_create_table_inlines_spfresh_vector_index(self):
        tidb = make_tidb()
        cursor = FakeCursor()
        conn = FakeConnection()

        class ConnectionContext:
            def __enter__(self_inner) -> tuple[FakeConnection, FakeCursor]:
                return conn, cursor

            def __exit__(
                self_inner,
                exc_type: type[BaseException] | None,
                exc: BaseException | None,
                tb: TracebackType | None,
            ) -> bool:
                return False

        with patch.object(tidb, "_get_connection", return_value=ConnectionContext()):
            tidb._create_table()

        assert conn.committed is True
        sql, _ = cursor.execute_calls[0]
        assert "CREATE TABLE vector_bench_test" in sql
        assert "VECTOR INDEX idx_embedding_spfresh_l2 ((vec_l2_distance(embedding))) USING SPFRESH" in sql

    def test_optimize_waits_for_recorded_insert_barrier(self):
        tidb = make_tidb()
        tidb._max_insert_commit_ts = 123456
        cursor = FakeCursor(fetchone_results=[(42,), (84,)])
        tidb.cursor = cursor
        tidb.conn = FakeConnection()

        with patch.object(tidb, "_wait_for_incremental_catch_up") as wait_mock:
            tidb.optimize()

        wait_mock.assert_called_once_with(barrier_ts=123456, table_id=42, index_id=84)

    def test_load_train_data_restores_tidb_load_state_from_insert_runner(self):
        class FakeDB:
            def __init__(self):
                self.imported_states: list[dict[str, int] | None] = []

            def need_normalize_cosine(self) -> bool:
                return False

            def import_load_state(self, state: dict[str, int] | None) -> None:
                self.imported_states.append(state)

        case_runner = CaseRunner.construct(
            run_id="run",
            config=None,
            ca=SimpleNamespace(
                dataset=SimpleNamespace(data=SimpleNamespace(metric_type=MetricType.L2)),
                filters=None,
                load_timeout=None,
            ),
            status=None,
            dataset_source=None,
            db=FakeDB(),
        )

        with patch("vectordb_bench.backend.task_runner.SerialInsertRunner") as runner_cls:
            runner_cls.return_value.run.return_value = (1000, {"max_insert_commit_ts": 123456})
            case_runner._load_train_data()

        assert case_runner.db.imported_states == [{"max_insert_commit_ts": 123456}]

    def test_wait_for_incremental_catch_up_succeeds_after_polling(self):
        tidb = make_tidb()
        responses = [
            [],
            [
                {
                    "table_id": 42,
                    "index_id": 84,
                    "runtime_state": "running",
                    "applied_base_ts": 100,
                    "global_resolved_ts": 120,
                    "buffered_events": 8,
                    "lag_seconds": 2.5,
                    "last_error": "",
                }
            ],
            [
                {
                    "table_id": 42,
                    "index_id": 84,
                    "runtime_state": "running",
                    "applied_base_ts": 149,
                    "global_resolved_ts": 151,
                    "buffered_events": 0,
                    "lag_seconds": 0.2,
                    "last_error": "",
                }
            ],
        ]

        with (
            patch.object(tidb, "_fetch_incremental_lifecycle", side_effect=responses),
            patch(
                "vectordb_bench.backend.clients.tidb.tidb.time.sleep",
            ),
        ):
            tidb._wait_for_incremental_catch_up(
                barrier_ts=150,
                table_id=42,
                index_id=84,
                timeout_seconds=5,
                registration_timeout_seconds=2,
                poll_interval_seconds=0.01,
            )

    def test_wait_for_incremental_catch_up_succeeds_when_applied_base_ts_crosses_barrier_without_new_debug_fields(self):
        tidb = make_tidb()

        with (
            patch.object(
                tidb,
                "_fetch_incremental_lifecycle",
                return_value=[
                    {
                        "table_id": 42,
                        "index_id": 84,
                        "runtime_state": "running",
                        "applied_base_ts": 151,
                        "lag_seconds": 0,
                        "last_error": "",
                    }
                ],
            ),
            patch("vectordb_bench.backend.clients.tidb.tidb.time.sleep"),
        ):
            tidb._wait_for_incremental_catch_up(
                barrier_ts=150,
                table_id=42,
                index_id=84,
                timeout_seconds=1,
                registration_timeout_seconds=1,
                poll_interval_seconds=0.01,
            )

    def test_wait_for_incremental_catch_up_fails_on_worker_error(self):
        tidb = make_tidb()
        with (
            patch.object(
                tidb,
                "_fetch_incremental_lifecycle",
                return_value=[
                    {
                        "table_id": 42,
                        "index_id": 84,
                        "runtime_state": "running",
                        "applied_base_ts": 100,
                        "global_resolved_ts": 120,
                        "lag_seconds": 2.5,
                        "last_error": "boom",
                    }
                ],
            ),
            pytest.raises(RuntimeError, match="last_error"),
        ):
            tidb._wait_for_incremental_catch_up(
                barrier_ts=150,
                table_id=42,
                index_id=84,
                timeout_seconds=1,
                registration_timeout_seconds=1,
                poll_interval_seconds=0.01,
            )

    def test_wait_for_incremental_catch_up_fails_on_rebuild_state(self):
        tidb = make_tidb()
        with (
            patch.object(
                tidb,
                "_fetch_incremental_lifecycle",
                return_value=[
                    {
                        "table_id": 42,
                        "index_id": 84,
                        "runtime_state": "snapshot_rebuild",
                        "applied_base_ts": 100,
                        "global_resolved_ts": 120,
                        "lag_seconds": 2.5,
                        "last_error": "",
                    }
                ],
            ),
            pytest.raises(RuntimeError, match="rebuild-like"),
        ):
            tidb._wait_for_incremental_catch_up(
                barrier_ts=150,
                table_id=42,
                index_id=84,
                timeout_seconds=1,
                registration_timeout_seconds=1,
                poll_interval_seconds=0.01,
            )

    def test_fetch_incremental_lifecycle_uses_json_endpoint(self):
        tidb = make_tidb(worker_debug_url="http://worker:5678/base")
        payload = b'[{"table_id": 1, "index_id": 2, "applied_base_ts": 3}]'

        class Response:
            def __enter__(self_inner) -> "Response":
                return self_inner

            def __exit__(
                self_inner,
                exc_type: type[BaseException] | None,
                exc: BaseException | None,
                tb: TracebackType | None,
            ) -> bool:
                return False

            def read(self_inner) -> bytes:
                return payload

        with patch("vectordb_bench.backend.clients.tidb.tidb.urlopen", return_value=Response()) as urlopen_mock:
            rows = tidb._fetch_incremental_lifecycle()

        assert rows == [{"table_id": 1, "index_id": 2, "applied_base_ts": 3}]
        assert urlopen_mock.call_args.args[0] == "http://worker:5678/base/debug/spfresh/incremental_lifecycle"
