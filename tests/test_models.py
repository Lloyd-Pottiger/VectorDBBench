import logging
import json

import pytest

from vectordb_bench.models import (
    TaskConfig, CaseConfig,
    CaseResult, TestResult,
    Metric, CaseType
)
from vectordb_bench.backend.clients import (
    DB,
    IndexType,
)

from vectordb_bench import config


log = logging.getLogger("vectordb_bench")


class TestModels:
    @pytest.mark.skip("runs locally")
    def test_test_result(self):
        result = CaseResult(
            task_config=TaskConfig(
                db=DB.Milvus,
                db_config=DB.Milvus.config(),
                db_case_config=DB.Milvus.case_config_cls(index=IndexType.Flat)(),
                case_config=CaseConfig(case_id=CaseType.Performance10M),
            ),
            metrics=Metric(),
        )

        test_result = TestResult(run_id=10000, results=[result])
        test_result.flush()

        with pytest.raises(ValueError):
            result = TestResult.read_file('nosuchfile.json')

    def test_test_result_read_write(self):
        result_dir = config.RESULTS_LOCAL_DIR
        for json_file in result_dir.rglob("result*.json"):
            res = TestResult.read_file(json_file)
            res.flush()

    def test_test_result_merge(self):
        result_dir = config.RESULTS_LOCAL_DIR
        all_results = []

        first_result = None
        for json_file in result_dir.glob("*.json"):
            res = TestResult.read_file(json_file)

            for cr in res.results:
                all_results.append(cr)

            if not first_result:
                first_result = res

        tr = TestResult(
            run_id=first_result.run_id,
            task_label="standard",
            results=all_results,
        )
        tr.flush()

    def test_test_result_display(self):
        result_dir = config.RESULTS_LOCAL_DIR
        for json_file in result_dir.rglob("result*.json"):
            log.info(json_file)
            res = TestResult.read_file(json_file)
            res.display()

    def test_test_result_read_file_accepts_empty_common_db_metadata(self, tmp_path):
        result = CaseResult(
            task_config=TaskConfig(
                db=DB.TiDB,
                db_config=DB.TiDB.config_cls(password="", host="127.0.0.1", port=4000, db_name="test"),
                db_case_config=DB.TiDB.case_config_cls()(),
                case_config=CaseConfig(case_id=CaseType.Performance1024D1M),
            ),
            metrics=Metric(),
        )
        test_result = TestResult(run_id="compat-run", task_label="compat", results=[result])

        json_file = tmp_path / "result_compat_tidb.json"
        json_file.write_text(test_result.json(), encoding="utf-8")

        raw_result = json.loads(json_file.read_text(encoding="utf-8"))
        raw_db_config = raw_result["results"][0]["task_config"]["db_config"]
        assert raw_db_config["version"] == ""
        assert raw_db_config["note"] == ""

        loaded = TestResult.read_file(json_file)

        assert loaded.results[0].task_config.db_config.version == ""
        assert loaded.results[0].task_config.db_config.note == ""
