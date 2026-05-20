from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

from vectordb_bench.backend.clients.api import MetricType
from vectordb_bench.backend.clients.tidb.config import TiDBIndexConfig
from vectordb_bench.backend.clients.tidb.tidb import TiDB
from vectordb_bench.backend.task_runner import CaseRunner
from vectordb_bench.cli.cli import parse_task_stages
from vectordb_bench.models import TaskStage


class FakeConn:
    def __init__(self):
        self.commits = 0
        self.rollbacks = 0

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


class FakeCursor:
    def __init__(self, existing_ids=None):
        self.statements = []
        self.existing_ids = set(existing_ids or [])
        self._search_results = []

    def execute(self, sql, params=None):
        self.statements.append((sql, params))
        if sql == "DELETE FROM vector_bench_test WHERE id = %s":
            row_id = params[0]
            if row_id in self.existing_ids:
                self.existing_ids.remove(row_id)
                return 1
            return 0
        if "SELECT id FROM vector_bench_test" in sql:
            self._search_results = sorted(self.existing_ids)[:1]
            return len(self._search_results)
        return 1

    def fetchall(self):
        return [(row_id,) for row_id in self._search_results]


class FakeRowCursor:
    def __init__(self, rows):
        self.rows = list(rows)
        self.statements = []
        self._last_row = None

    def execute(self, sql, params=None):
        self.statements.append((sql, params))
        if self.rows:
            self._last_row = self.rows.pop(0)
        else:
            self._last_row = None
        return 1 if self._last_row is not None else 0

    def fetchone(self):
        return self._last_row


def test_parse_task_stages_supports_delete_only():
    stages = parse_task_stages(
        drop_old=False,
        load=False,
        search_serial=False,
        search_concurrent=False,
        delete=True,
    )

    assert stages == [TaskStage.DELETE]


def test_parse_task_stages_supports_build_only():
    stages = parse_task_stages(
        drop_old=False,
        load=False,
        search_serial=False,
        search_concurrent=False,
        build=True,
    )

    assert stages == [TaskStage.BUILD]


def test_parse_task_stages_rejects_build_with_load():
    error = None
    try:
        parse_task_stages(
            drop_old=True,
            load=True,
            search_serial=False,
            search_concurrent=False,
            build=True,
        )
    except RuntimeError as exc:
        error = str(exc)

    if error is None:
        raise AssertionError("Expected parse_task_stages to reject build+load combination")
    assert "Build-only cannot be combined with loading data" in error


def test_parse_task_stages_rejects_delete_with_search():
    try:
        parse_task_stages(
            drop_old=False,
            load=False,
            search_serial=True,
            search_concurrent=False,
            delete=True,
        )
    except RuntimeError as exc:
        assert "Delete benchmark cannot be combined with search stages" in str(exc)
    else:
        raise AssertionError("Expected parse_task_stages to reject delete+search combination")


def test_tidb_optimize_skips_spfresh_wait_when_index_is_disabled():
    tidb = TiDB(
        dim=3,
        db_config={},
        db_case_config=TiDBIndexConfig(metric_type=MetricType.COSINE, build_spfresh_index=False),
        collection_name="vector_bench_test",
    )
    tidb._max_insert_commit_ts = 12345
    tidb.conn = FakeConn()
    tidb.cursor = FakeCursor()

    tidb.optimize()

    assert tidb.cursor.statements == []
    assert tidb.conn.commits == 0


def test_tidb_create_table_can_inline_spfresh_index():
    tidb = TiDB(
        dim=3,
        db_config={},
        db_case_config=TiDBIndexConfig(metric_type=MetricType.COSINE),
        collection_name="vector_bench_test",
    )
    conn = FakeConn()
    cursor = FakeCursor()

    @contextmanager
    def fake_get_connection():
        yield conn, cursor

    tidb._get_connection = fake_get_connection

    tidb._create_table()

    assert len(cursor.statements) == 1
    create_table_sql = " ".join(cursor.statements[0][0].split())
    assert "CREATE TABLE vector_bench_test" in create_table_sql
    assert "embedding VECTOR(3) NOT NULL" in create_table_sql
    assert "VECTOR INDEX idx_embedding_spfresh_cosine ((vec_cosine_distance(embedding))) USING SPFRESH" in (
        create_table_sql
    )
    assert conn.commits == 1


def test_tidb_create_table_can_skip_spfresh_index():
    tidb = TiDB(
        dim=3,
        db_config={},
        db_case_config=TiDBIndexConfig(metric_type=MetricType.COSINE, build_spfresh_index=False),
        collection_name="vector_bench_test",
    )
    conn = FakeConn()
    cursor = FakeCursor()

    @contextmanager
    def fake_get_connection():
        yield conn, cursor

    tidb._get_connection = fake_get_connection

    tidb._create_table()

    create_table_sql = " ".join(cursor.statements[0][0].split())
    assert "CREATE TABLE vector_bench_test" in create_table_sql
    assert "VECTOR INDEX" not in create_table_sql
    assert conn.commits == 1


def test_tidb_optimize_waits_for_insert_commit_barrier():
    tidb = TiDB(
        dim=3,
        db_config={},
        db_case_config=TiDBIndexConfig(metric_type=MetricType.COSINE),
        collection_name="vector_bench_test",
    )
    tidb._max_insert_commit_ts = 12345
    tidb.conn = FakeConn()
    tidb.cursor = FakeCursor()
    waited = []

    def fake_wait(barrier_ts: int):
        waited.append(barrier_ts)

    tidb._wait_for_spfresh_ready = fake_wait
    tidb.optimize()

    assert waited == [12345]
    assert tidb._max_insert_commit_ts is None


def test_tidb_build_only_optimize_waits_for_current_tso_barrier():
    tidb = TiDB(
        dim=3,
        db_config={},
        db_case_config=TiDBIndexConfig(metric_type=MetricType.COSINE),
        collection_name="vector_bench_test",
    )
    tidb.conn = FakeConn()
    tidb.cursor = FakeRowCursor(rows=[None, ("34567",)])
    waited = []

    def fake_wait(barrier_ts: int):
        waited.append(barrier_ts)

    tidb._wait_for_spfresh_ready = fake_wait
    tidb.optimize()

    assert waited == [34567]
    assert tidb.cursor.statements == [("BEGIN", None), ("SELECT @@tidb_current_ts", None)]
    assert tidb.conn.commits == 1


def test_tidb_waits_for_post_delete_commit_barrier():
    tidb = TiDB(
        dim=3,
        db_config={},
        db_case_config=TiDBIndexConfig(metric_type=MetricType.COSINE),
        collection_name="vector_bench_test",
    )
    tidb._max_delete_commit_ts = 23456
    waited = []

    @contextmanager
    def fake_init():
        yield

    def fake_wait(barrier_ts: int, timeout_seconds: float, poll_interval_seconds: float):
        waited.append((barrier_ts, timeout_seconds, poll_interval_seconds))

    tidb.init = fake_init
    tidb._wait_for_spfresh_ready = fake_wait

    barrier_ts = tidb.wait_spfresh_post_delete_catchup(timeout=3.0, poll_interval=0.25)

    assert barrier_ts == 23456
    assert waited == [(23456, 3.0, 0.25)]
    assert tidb._max_delete_commit_ts is None


def test_tidb_delete_embeddings_executes_one_delete_per_id():
    tidb = TiDB(
        dim=3,
        db_config={},
        db_case_config=TiDBIndexConfig(metric_type=MetricType.COSINE, delete_commit_interval=2),
        collection_name="vector_bench_test",
    )
    tidb.conn = FakeConn()
    tidb.cursor = FakeCursor(existing_ids=[1, 2, 3, 4, 5])
    tidb._last_commit_ts = lambda _cursor: tidb.conn.commits * 100

    deleted, error = tidb.delete_embeddings(ids=[1, 2, 3, 4, 5])

    assert error is None
    assert deleted == 5
    assert len(tidb.cursor.statements) == 5
    assert all("DELETE FROM vector_bench_test WHERE id = %s" == sql for sql, _ in tidb.cursor.statements)
    assert [params for _, params in tidb.cursor.statements] == [(1,), (2,), (3,), (4,), (5,)]
    assert tidb.conn.commits == 3
    assert tidb.conn.rollbacks == 0
    assert tidb.export_delete_state() == {"max_delete_commit_ts": 300}


def test_tidb_delete_then_search_returns_empty_result():
    tidb = TiDB(
        dim=3,
        db_config={},
        db_case_config=TiDBIndexConfig(metric_type=MetricType.COSINE, delete_commit_interval=10),
        collection_name="vector_bench_test",
    )
    tidb.conn = FakeConn()
    tidb.cursor = FakeCursor(existing_ids=[1, 2, 3])
    tidb._last_commit_ts = lambda _cursor: tidb.conn.commits * 100

    deleted, error = tidb.delete_embeddings(ids=[1, 2, 3])
    results = tidb.search_embedding(query=[0.1, 0.2, 0.3], k=1)

    assert error is None
    assert deleted == 3
    assert results == []


def test_delete_data_restores_tidb_delete_state_from_delete_runner():
    class FakeDB:
        def __init__(self):
            self.imported_states = []

        def import_delete_state(self, state):
            self.imported_states.append(state)

    fake_db = FakeDB()
    case_runner = CaseRunner.construct(
        run_id="run",
        config=SimpleNamespace(db_case_config=SimpleNamespace(delete_timeout=None)),
        ca=SimpleNamespace(dataset=None, optimize_timeout=10.0),
        status=None,
        dataset_source=None,
        db=fake_db,
    )

    with patch("vectordb_bench.backend.task_runner.SerialDeleteRunner") as runner_cls:
        runner_cls.return_value.run.return_value = (1000, {"max_delete_commit_ts": 23456})
        case_runner._delete_data()

    assert fake_db.imported_states == [{"max_delete_commit_ts": 23456}]


class FakeDeleteSearchDB:
    def __init__(self, search_responses=None):
        self.events = []
        self.queries = []
        self.filters = []
        self.search_responses = list(search_responses or [[]])

    def need_normalize_cosine(self):
        return False

    @contextmanager
    def init(self):
        yield

    def prepare_filter(self, filters):
        self.filters.append(filters)

    def search_embedding(self, query, k=100):
        self.events.append("search")
        self.queries.append((query, k))
        if self.search_responses:
            return self.search_responses.pop(0)
        return []


class FakeCatchupDeleteSearchDB(FakeDeleteSearchDB):
    def __init__(self, search_responses=None):
        super().__init__(search_responses)
        self.catchup_calls = []

    def wait_spfresh_post_delete_catchup(self, timeout, poll_interval):
        self.events.append("catchup")
        self.catchup_calls.append((timeout, poll_interval))
        return 123


class FailingDeleteSearchDB(FakeDeleteSearchDB):
    def search_embedding(self, query, k=100):
        raise RuntimeError("tidb search error")


def test_case_runner_verify_search_after_delete_records_immediate_stale_then_eventual_empty():
    fake_db = FakeDeleteSearchDB(search_responses=[[42], []])
    dataset = SimpleNamespace(
        test_data=[[0.1, 0.2, 0.3]],
        data=SimpleNamespace(metric_type=MetricType.COSINE),
    )
    case_runner = CaseRunner.construct(
        run_id="run",
        config=SimpleNamespace(
            db_case_config=SimpleNamespace(
                delete_search_wait_timeout=0.1,
                delete_search_poll_interval=0.01,
            )
        ),
        ca=SimpleNamespace(dataset=dataset, filters="non_filter"),
        status=None,
        dataset_source=None,
        db=fake_db,
    )

    immediate_results, final_results, settle_duration, poll_count, deadline_exceeded = (
        case_runner._verify_search_after_delete()
    )

    assert immediate_results == [42]
    assert final_results == []
    assert settle_duration >= 0
    assert poll_count == 2
    assert deadline_exceeded is False
    assert fake_db.queries == [([0.1, 0.2, 0.3], 1), ([0.1, 0.2, 0.3], 1)]
    assert fake_db.filters == ["non_filter", "non_filter"]


def test_case_runner_waits_for_spfresh_catchup_before_post_delete_search():
    fake_db = FakeCatchupDeleteSearchDB(search_responses=[[]])
    dataset = SimpleNamespace(
        test_data=[[0.1, 0.2, 0.3]],
        data=SimpleNamespace(metric_type=MetricType.COSINE),
    )
    case_runner = CaseRunner.construct(
        run_id="run",
        config=SimpleNamespace(
            db_case_config=SimpleNamespace(
                delete_search_wait_timeout=0.1,
                delete_search_poll_interval=0.01,
            )
        ),
        ca=SimpleNamespace(dataset=dataset, filters="non_filter"),
        status=None,
        dataset_source=None,
        db=fake_db,
    )

    immediate_results, final_results, settle_duration, poll_count, deadline_exceeded = (
        case_runner._verify_search_after_delete()
    )

    assert immediate_results == []
    assert final_results == []
    assert settle_duration >= 0
    assert poll_count == 1
    assert deadline_exceeded is False
    assert fake_db.events == ["catchup", "search"]
    assert fake_db.catchup_calls == [(0.1, 0.01)]


def test_case_runner_verify_search_after_delete_marks_deadline_exceeded_when_results_stay_non_empty():
    fake_db = FakeDeleteSearchDB(search_responses=[[42], [42], [42]])
    dataset = SimpleNamespace(
        test_data=[[0.1, 0.2, 0.3]],
        data=SimpleNamespace(metric_type=MetricType.COSINE),
    )
    case_runner = CaseRunner.construct(
        run_id="run",
        config=SimpleNamespace(
            db_case_config=SimpleNamespace(
                delete_search_wait_timeout=0.02,
                delete_search_poll_interval=0.01,
            )
        ),
        ca=SimpleNamespace(dataset=dataset, filters="non_filter"),
        status=None,
        dataset_source=None,
        db=fake_db,
    )

    immediate_results, final_results, settle_duration, poll_count, deadline_exceeded = (
        case_runner._verify_search_after_delete()
    )

    assert immediate_results == [42]
    assert final_results == [42]
    assert settle_duration >= 0
    assert poll_count >= 2
    assert deadline_exceeded is True


def test_case_runner_perf_case_raises_when_delete_verification_deadline_expires():
    case_runner = CaseRunner.construct(
        run_id="run",
        config=SimpleNamespace(stages=[TaskStage.DELETE]),
        ca=SimpleNamespace(dataset=None),
        status=None,
        dataset_source=None,
        db=None,
    )
    object.__setattr__(case_runner, "_delete_data", lambda: (0, 0.0))
    object.__setattr__(case_runner, "_verify_search_after_delete", lambda: ([42], [42], 0.02, 2, True))

    try:
        case_runner._run_perf_case()
    except RuntimeError as exc:
        assert "did not become empty before the deadline" in str(exc)
    else:
        raise AssertionError("Expected delete benchmark to fail when final ANN results stay non-empty")


def test_case_runner_perf_case_runs_build_only_stage():
    optimize_calls = []
    case_runner = CaseRunner.construct(
        run_id="run",
        config=SimpleNamespace(stages=[TaskStage.BUILD]),
        ca=SimpleNamespace(dataset=None),
        status=None,
        dataset_source=None,
        db=None,
    )
    object.__setattr__(case_runner, "_optimize", lambda: optimize_calls.append("optimize") or 1.23456)

    metric = case_runner._run_perf_case(drop_old=False)

    assert optimize_calls == ["optimize"]
    assert metric.optimize_duration == 1.2346


def test_case_runner_verify_search_after_delete_raises_immediately_on_query_error():
    fake_db = FailingDeleteSearchDB()
    dataset = SimpleNamespace(
        test_data=[[0.1, 0.2, 0.3]],
        data=SimpleNamespace(metric_type=MetricType.COSINE),
    )
    case_runner = CaseRunner.construct(
        run_id="run",
        config=SimpleNamespace(
            db_case_config=SimpleNamespace(
                delete_search_wait_timeout=0.1,
                delete_search_poll_interval=0.01,
            )
        ),
        ca=SimpleNamespace(dataset=dataset, filters="non_filter"),
        status=None,
        dataset_source=None,
        db=fake_db,
    )

    try:
        case_runner._verify_search_after_delete()
    except RuntimeError as exc:
        assert "Post-delete ANN search failed" in str(exc)
    else:
        raise AssertionError("Expected delete benchmark to stop immediately on ANN query error")
