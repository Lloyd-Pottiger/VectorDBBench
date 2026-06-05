import concurrent
import hashlib
import logging
import re
import time
import traceback
from enum import Enum, auto

import numpy as np
import psutil

from ..base import BaseModel
from ..metric import Metric, OptimizeResult
from ..models import PerformanceTimeoutError, TaskConfig, TaskStage
from . import utils
from .cases import Case, CaseLabel, StreamingPerformanceCase
from .clients import DB, MetricType, api
from .data_source import DatasetSource
from .runner import (
    MultiProcessingSearchRunner,
    ReadWriteRunner,
    SerialDeleteRunner,
    SerialInsertRunner,
    SerialSearchRunner,
)

log = logging.getLogger(__name__)


class RunningStatus(Enum):
    PENDING = auto()
    FINISHED = auto()


class CaseRunner(BaseModel):
    """DataSet, filter_rate, db_class with db config

    Fields:
        run_id(str): run_id of this case runner,
            indicating which task does this case belong to.
        config(TaskConfig): task configs of this case runner.
        ca(Case): case for this case runner.
        status(RunningStatus): RunningStatus of this case runner.

        db(api.VectorDB): The vector database for this case runner.
    """

    run_id: str
    config: TaskConfig
    ca: Case
    status: RunningStatus
    dataset_source: DatasetSource

    db: api.VectorDB | None = None
    test_emb: list[list[float]] | None = None
    serial_search_runner: SerialSearchRunner | None = None
    search_runner: MultiProcessingSearchRunner | None = None
    final_search_runner: MultiProcessingSearchRunner | None = None
    read_write_runner: ReadWriteRunner | None = None

    def __eq__(self, obj: any):
        if isinstance(obj, CaseRunner):
            return (
                self.ca.label == CaseLabel.Performance
                and self.config.db == obj.config.db
                and self.config.db_case_config == obj.config.db_case_config
                and self.ca.dataset == obj.ca.dataset
            )
        return False

    def __hash__(self) -> int:
        """Hash method to maintain consistency with __eq__ method."""
        return hash(
            (
                self.ca.label,
                self.config.db,
                self.config.db_case_config,
                self.ca.dataset,
            )
        )

    def display(self) -> dict:
        c_dict = self.ca.dict(
            include={
                "label": True,
                "name": True,
                "filters": True,
                "dataset": {
                    "data": {
                        "name": True,
                        "size": True,
                        "dim": True,
                        "metric_type": True,
                        "label": True,
                    },
                },
            },
        )
        c_dict["db"] = self.config.db_name
        return c_dict

    @property
    def normalize(self) -> bool:
        assert self.db
        return self.db.need_normalize_cosine() and self.ca.dataset.data.metric_type == MetricType.COSINE

    def init_db(self, drop_old: bool = True) -> None:
        db_cls = self.config.db.init_cls
        # Compose a compact, case-unique collection/table name for Doris to avoid cross-case interference
        collection_name = None
        try:
            if self.config.db == DB.Doris:
                # Primary identifier = case-type enum name from CLI (e.g., Performance768D10M)
                case_type_name = self.config.case_config.case_id.name
                base = f"{case_type_name.lower()}"
                # Sanitize to [a-z0-9_]
                base = re.sub(r"[^a-z0-9_]+", "_", base).strip("_")
                # Cap to 63 chars; add short hash if truncated
                if len(base) > 63:
                    h = hashlib.md5(base.encode(), usedforsecurity=False).hexdigest()[:6]
                    base = f"{base[:(63-7)]}_{h}"
                collection_name = base
        except Exception:
            # If anything goes wrong, fall back silently; Doris will use its default name logic
            collection_name = None

        # Check if collection_name is in the db_config (e.g., for Zilliz, Milvus)
        db_config_dict = self.config.db_config.to_dict()
        if "collection_name" in db_config_dict and not collection_name:
            collection_name = db_config_dict.pop("collection_name")

        self.db = db_cls(
            dim=self.ca.dataset.data.dim,
            db_config=db_config_dict,
            db_case_config=self.config.db_case_config,
            drop_old=drop_old,
            with_scalar_labels=self.ca.with_scalar_labels,
            **({"collection_name": collection_name} if collection_name else {}),
        )

    def _pre_run(self, drop_old: bool = True):
        try:
            self.init_db(drop_old)
            if self.ca.label != CaseLabel.Performance or self._requires_perf_dataset_prepare():
                self.ca.dataset.prepare(self.dataset_source, filters=self.ca.filters)
            else:
                log.info("Dataset preparation skipped")
        except ModuleNotFoundError as e:
            log.warning(f"pre run case error: please install client for db: {self.config.db}, error={e}")
            raise e from None

    def _requires_perf_dataset_prepare(self) -> bool:
        return any(
            stage in self.config.stages
            for stage in (
                TaskStage.LOAD,
                TaskStage.DELETE,
                TaskStage.SEARCH_SERIAL,
                TaskStage.SEARCH_CONCURRENT,
            )
        )

    def run(self, drop_old: bool = True) -> Metric:
        log.info("Starting run")

        self._pre_run(drop_old)

        if self.ca.label == CaseLabel.Load:
            return self._run_capacity_case()
        if self.ca.label == CaseLabel.Performance:
            return self._run_perf_case(drop_old)
        if self.ca.label == CaseLabel.Streaming:
            return self._run_streaming_case()
        msg = f"unknown case type: {self.ca.label}"
        log.warning(msg)
        raise ValueError(msg)

    def _run_capacity_case(self) -> Metric:
        """run capacity cases

        Returns:
            Metric: the max load count
        """
        assert self.db is not None
        log.info("Start capacity case")
        try:
            runner = SerialInsertRunner(
                self.db,
                self.ca.dataset,
                self.normalize,
                self.ca.filters,
                self.ca.load_timeout,
            )
            count = runner.run_endlessness()
        except Exception as e:
            log.warning(f"Failed to run capacity case, reason = {e}")
            raise e from None
        else:
            log.info(f"Capacity case loading dataset reaches VectorDB's limit: max capacity = {count}")
            return Metric(max_load_count=count)

    def _run_perf_case(self, drop_old: bool = True) -> Metric:
        """run performance cases

        Returns:
            Metric: load_duration/delete_duration, post-delete checks, recall,
                serial_latency_p99, and qps
        """

        log.info("Start performance case")
        try:
            m = Metric()
            if drop_old:
                self._run_perf_load_stage(m)
            if TaskStage.BUILD in self.config.stages and TaskStage.LOAD not in self.config.stages:
                self._run_perf_build_only_stage(m)
            if TaskStage.DELETE in self.config.stages:
                self._run_perf_delete_stage(m)
            if TaskStage.SEARCH_SERIAL in self.config.stages or TaskStage.SEARCH_CONCURRENT in self.config.stages:
                self._run_perf_search_stages(m)

        except Exception as e:
            log.warning(f"Failed to run performance case, reason = {e}")
            traceback.print_exc()
            raise e from None
        else:
            log.info(f"Performance case got result: {m}")
            return m

    def _run_perf_load_stage(self, metric: Metric) -> None:
        if TaskStage.LOAD not in self.config.stages:
            log.info("Data loading skipped")
            return

        if self._is_tidb_spfresh_split_load():
            self._run_spfresh_split_load(metric)
            log.info(
                "Finish SPFRESH split load, insert_duration=%s, "
                "spfresh_build_duration=%s, spfresh_incremental_catchup_duration=%s, "
                "load_duration=%s",
                metric.insert_duration,
                metric.spfresh_build_duration,
                metric.spfresh_incremental_catchup_duration,
                metric.load_duration,
            )
        else:
            self._run_standard_load_stage(metric)

        log.info(
            f"Finish loading the entire dataset into VectorDB,"
            f" insert_duration={metric.insert_duration}, optimize_duration={metric.optimize_duration}"
            f" load_duration(insert + optimize) = {metric.load_duration}"
        )

    def _run_standard_load_stage(self, metric: Metric) -> None:
        load_start = time.perf_counter()
        _, load_dur = self._load_train_data()
        optimize_result = self._coerce_optimize_result(self._optimize())
        if self._is_tidb_spfresh_inline_load():
            incremental_catchup_dur = time.perf_counter() - load_start
            optimize_result.spfresh_incremental_catchup_duration = incremental_catchup_dur
            optimize_result.optimize_duration = optimize_result.spfresh_build_duration + incremental_catchup_dur
            load_duration = optimize_result.optimize_duration
        else:
            load_duration = load_dur + optimize_result.optimize_duration

        metric.insert_duration = round(load_dur, 4)
        self._record_optimize_result(metric, optimize_result)
        metric.load_duration = round(load_duration, 4)

    def _run_perf_build_only_stage(self, metric: Metric) -> None:
        build_result = self._optimize()
        self._record_optimize_result(metric, build_result)
        log.info(f"Finish building VectorDB index, optimize_duration={metric.optimize_duration}")

    def _run_perf_delete_stage(self, metric: Metric) -> None:
        _, delete_dur = self._delete_data()
        metric.delete_duration = round(delete_dur, 4)
        log.info(f"Finish deleting the entire dataset from VectorDB, delete_duration={delete_dur}")
        (
            immediate_results,
            final_results,
            settle_duration,
            poll_count,
            deadline_exceeded,
        ) = self._verify_search_after_delete()
        metric.delete_search_immediate_results = immediate_results
        metric.delete_search_immediate_result_count = len(immediate_results)
        metric.delete_search_final_results = final_results
        metric.delete_search_final_result_count = len(final_results)
        metric.delete_search_settle_duration = round(settle_duration, 4)
        metric.delete_search_poll_count = poll_count
        log.info(
            "Post-delete ANN search finished, immediate_count=%s, immediate_results=%s, "
            "final_count=%s, final_results=%s, settle_duration=%s, polls=%s",
            metric.delete_search_immediate_result_count,
            metric.delete_search_immediate_results,
            metric.delete_search_final_result_count,
            metric.delete_search_final_results,
            metric.delete_search_settle_duration,
            metric.delete_search_poll_count,
        )
        if deadline_exceeded:
            self._raise_post_delete_verification_error(final_results)

    def _run_perf_search_stages(self, metric: Metric) -> None:
        self._init_search_runner()
        if TaskStage.SEARCH_CONCURRENT in self.config.stages:
            search_results = self._conc_search()
            (
                metric.qps,
                metric.conc_num_list,
                metric.conc_qps_list,
                metric.conc_latency_p99_list,
                metric.conc_latency_p95_list,
                metric.conc_latency_avg_list,
            ) = search_results
        if TaskStage.SEARCH_SERIAL in self.config.stages:
            search_results = self._serial_search()
            metric.recall, metric.ndcg, metric.serial_latency_p99, metric.serial_latency_p95 = search_results

    def _run_streaming_case(self) -> Metric:
        log.info("Start streaming case")
        try:
            self._init_read_write_runner()
            m = self.read_write_runner.run_read_write()
        except Exception as e:
            log.warning(f"Failed to run streaming case, reason = {e}")
            traceback.print_exc()
            raise e from None
        else:
            log.info(f"Streaming case got result: {m}")
            return m

    def _tidb_spfresh_build_mode(self) -> str | None:
        if getattr(self.config, "db", None) != DB.TiDB:
            return None
        db_case_config = getattr(self.config, "db_case_config", None)
        if not getattr(db_case_config, "build_spfresh_index", False):
            return None
        mode = getattr(db_case_config, "spfresh_build_mode", "inline")
        return getattr(mode, "value", mode)

    def _is_tidb_spfresh_inline_load(self) -> bool:
        return self._tidb_spfresh_build_mode() == "inline"

    def _is_tidb_spfresh_split_load(self) -> bool:
        return self._tidb_spfresh_build_mode() == "split"

    def _spfresh_split_counts(self) -> tuple[int, int]:
        ratio = float(getattr(self.config.db_case_config, "spfresh_split_ratio", 0.8))
        total = int(self.ca.dataset.data.size)
        if total < 2:
            raise ValueError("SPFRESH split load requires at least two rows")
        base_count = int(total * ratio)
        base_count = min(max(1, base_count), total - 1)
        return base_count, total - base_count

    @staticmethod
    def _coerce_optimize_result(result: OptimizeResult | float | None, wall_duration: float | None = None):
        if isinstance(result, OptimizeResult):
            return result
        if isinstance(result, int | float):
            return OptimizeResult(optimize_duration=float(result))
        return OptimizeResult(optimize_duration=0.0 if wall_duration is None else wall_duration)

    def _record_optimize_result(self, metric: Metric, result: OptimizeResult | float | None) -> None:
        optimize_result = self._coerce_optimize_result(result)
        metric.optimize_duration = round(optimize_result.optimize_duration, 4)
        metric.spfresh_build_duration = round(optimize_result.spfresh_build_duration, 4)
        metric.spfresh_incremental_catchup_duration = round(
            optimize_result.spfresh_incremental_catchup_duration,
            4,
        )

    def _run_spfresh_split_load(self, metric: Metric) -> None:
        base_count, delta_count = self._spfresh_split_counts()

        _, base_insert_dur = self._load_train_data(limit=base_count)
        build_result = self._coerce_optimize_result(self._build_spfresh_index())

        delta_start = time.perf_counter()
        _, delta_insert_dur = self._load_train_data(start_offset=base_count, limit=delta_count)
        wait_result = self._coerce_optimize_result(self._wait_spfresh_incremental_catchup())
        incremental_catchup_dur = time.perf_counter() - delta_start

        spfresh_build_dur = build_result.spfresh_build_duration
        optimize_dur = spfresh_build_dur + incremental_catchup_dur

        metric.spfresh_base_insert_duration = round(base_insert_dur, 4)
        metric.spfresh_delta_insert_duration = round(delta_insert_dur, 4)
        metric.insert_duration = round(base_insert_dur + delta_insert_dur, 4)
        metric.spfresh_build_duration = round(spfresh_build_dur, 4)
        metric.spfresh_incremental_catchup_duration = round(incremental_catchup_dur, 4)
        metric.optimize_duration = round(optimize_dur, 4)
        metric.load_duration = round(base_insert_dur + optimize_dur, 4)

        if wait_result.spfresh_incremental_catchup_duration:
            log.info(
                "SPFRESH split post-insert wait duration=%s, end-to-end incremental catch-up duration=%s",
                wait_result.spfresh_incremental_catchup_duration,
                incremental_catchup_dur,
            )

    @utils.time_it
    def _load_train_data(self, start_offset: int = 0, limit: int | None = None):
        """Insert train data and get the insert_duration"""
        try:
            runner = SerialInsertRunner(
                self.db,
                self.ca.dataset,
                self.normalize,
                self.ca.filters,
                self.ca.load_timeout,
                start_offset=start_offset,
                limit=limit,
            )
            count, load_state = runner.run()
        except Exception as e:
            raise e from None
        else:
            if hasattr(self.db, "import_load_state"):
                self.db.import_load_state(load_state)
            return count
        finally:
            runner = None

    @utils.time_it
    def _delete_data(self):
        """Delete train data and get the delete_duration"""
        try:
            delete_timeout = getattr(self.config.db_case_config, "delete_timeout", None)
            if delete_timeout is None:
                delete_timeout = self.ca.optimize_timeout
            runner = SerialDeleteRunner(
                self.db,
                self.ca.dataset,
                delete_timeout,
            )
            count, delete_state = runner.run()
        except Exception as e:
            raise e from None
        else:
            if hasattr(self.db, "import_delete_state"):
                self.db.import_delete_state(delete_state)
            return count
        finally:
            runner = None

    def _get_delete_search_query(self) -> list[float]:
        assert self.db is not None

        if self.normalize:
            query = np.array(self.ca.dataset.test_data[0])
            query = query / np.linalg.norm(query)
            return query.tolist()
        return self.ca.dataset.test_data[0]

    def _search_once_after_delete(self, search_query: list[float]) -> list[int]:
        assert self.db is not None
        try:
            with self.db.init():
                self.db.prepare_filter(self.ca.filters)
                return self.db.search_embedding(search_query, k=1)
        except Exception as e:
            log.exception("Post-delete ANN search failed, aborting delete benchmark")
            raise RuntimeError("Post-delete ANN search failed") from e

    @staticmethod
    def _raise_post_delete_verification_error(final_results: list[int]) -> None:
        msg = f"Post-delete ANN verification did not become empty before the deadline, final_results={final_results}"
        raise RuntimeError(msg)

    def _wait_post_delete_ann_ready(self, wait_timeout: float, poll_interval: float) -> None:
        assert self.db is not None

        wait_spfresh_post_delete_catchup = getattr(self.db, "wait_spfresh_post_delete_catchup", None)
        if wait_spfresh_post_delete_catchup is None:
            return

        try:
            barrier_ts = wait_spfresh_post_delete_catchup(wait_timeout, poll_interval)
        except Exception as e:
            log.exception("Post-delete SPFRESH catch-up failed before ANN search")
            raise RuntimeError("Post-delete SPFRESH catch-up failed before ANN search") from e

        if barrier_ts is not None:
            log.info(
                "SPFRESH incremental apply caught up before post-delete ANN search, delete_commit_barrier_ts=%s",
                barrier_ts,
            )

    def _verify_search_after_delete(self) -> tuple[list[int], list[int], float, int, bool]:
        """Record immediate post-delete ANN results, then wait for eventual empty results."""
        search_query = self._get_delete_search_query()
        wait_timeout = float(getattr(self.config.db_case_config, "delete_search_wait_timeout", 30.0))
        poll_interval = float(getattr(self.config.db_case_config, "delete_search_poll_interval", 1.0))

        self._wait_post_delete_ann_ready(wait_timeout, poll_interval)

        immediate_results = self._search_once_after_delete(search_query)
        deadline = time.perf_counter() + wait_timeout
        poll_count = 1
        final_results = immediate_results
        start = time.perf_counter()

        while final_results and time.perf_counter() < deadline:
            time.sleep(poll_interval)
            final_results = self._search_once_after_delete(search_query)
            poll_count += 1

        deadline_exceeded = bool(final_results)
        return immediate_results, final_results, time.perf_counter() - start, poll_count, deadline_exceeded

    def _serial_search(self) -> tuple[float, float, float, float]:
        """Performance serial tests, search the entire test data once,
        calculate the recall, serial_latency_p99, serial_latency_p95

        Returns:
            tuple[float, float, float, float]: recall, ndcg, serial_latency_p99, serial_latency_p95
        """
        try:
            results, _ = self.serial_search_runner.run()
        except Exception as e:
            log.warning(f"search error: {e!s}, {e}")
            self.stop()
            raise e from e
        else:
            return results

    def _conc_search(self):
        """Performance concurrency tests, search the test data endlessness
        for 30s in several concurrencies

        Returns:
            float: the largest qps in all concurrencies
        """
        try:
            return self.search_runner.run()
        except Exception as e:
            log.warning(f"search error: {e!s}, {e}")
            raise e from None
        finally:
            self.stop()

    @utils.time_it
    def _db_method_task(self, method_name: str) -> OptimizeResult | None:
        with self.db.init():
            method = getattr(self.db, method_name)
            if method_name == "optimize":
                return method(data_size=self.ca.dataset.data.size)
            return method()

    def _run_db_method(self, method_name: str) -> OptimizeResult:
        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._db_method_task, method_name)
            try:
                raw_result, wall_duration = future.result(timeout=self.ca.optimize_timeout)
                return self._coerce_optimize_result(raw_result, wall_duration)
            except TimeoutError as e:
                log.warning(f"VectorDB {method_name} timeout in {self.ca.optimize_timeout}")
                for pid, _ in executor._processes.items():
                    psutil.Process(pid).kill()
                raise PerformanceTimeoutError from e
            except Exception as e:
                log.warning(f"VectorDB {method_name} error: {e}")
                raise e from None

    def _optimize(self) -> OptimizeResult:
        return self._run_db_method("optimize")

    def _build_spfresh_index(self) -> OptimizeResult:
        return self._run_db_method("build_spfresh_index")

    def _wait_spfresh_incremental_catchup(self) -> OptimizeResult:
        return self._run_db_method("wait_spfresh_incremental_catchup")

    def _init_search_runner(self):
        if self.normalize:
            test_emb = np.stack(self.ca.dataset.test_data)
            test_emb = test_emb / np.linalg.norm(test_emb, axis=1)[:, np.newaxis]
            self.test_emb = test_emb.tolist()
        else:
            self.test_emb = self.ca.dataset.test_data

        gt_df = self.ca.dataset.gt_data

        if TaskStage.SEARCH_SERIAL in self.config.stages:
            self.serial_search_runner = SerialSearchRunner(
                db=self.db,
                test_data=self.test_emb,
                ground_truth=gt_df,
                filters=self.ca.filters,
                k=self.config.case_config.k,
            )
        if TaskStage.SEARCH_CONCURRENT in self.config.stages:
            self.search_runner = MultiProcessingSearchRunner(
                db=self.db,
                test_data=self.test_emb,
                filters=self.ca.filters,
                concurrencies=self.config.case_config.concurrency_search_config.num_concurrency,
                duration=self.config.case_config.concurrency_search_config.concurrency_duration,
                concurrency_timeout=self.config.case_config.concurrency_search_config.concurrency_timeout,
                k=self.config.case_config.k,
            )

    def _init_read_write_runner(self):
        ca: StreamingPerformanceCase = self.ca
        self.read_write_runner = ReadWriteRunner(
            db=self.db,
            dataset=ca.dataset,
            insert_rate=ca.insert_rate,
            search_stages=ca.search_stages,
            optimize_after_write=ca.optimize_after_write,
            read_dur_after_write=ca.read_dur_after_write,
            concurrencies=ca.concurrencies,
            k=self.config.case_config.k,
            normalize=self.normalize,
        )

    def stop(self):
        if self.search_runner:
            self.search_runner.stop()


DATA_FORMAT = " %-14s | %-12s %-20s %7s | %-10s"
TITLE_FORMAT = (" %-14s | %-12s %-20s %7s | %-10s") % (
    "DB",
    "CaseType",
    "Dataset",
    "Filter",
    "task_label",
)


class TaskRunner(BaseModel):
    run_id: str
    task_label: str
    case_runners: list[CaseRunner]

    def num_cases(self) -> int:
        return len(self.case_runners)

    def num_finished(self) -> int:
        return self._get_num_by_status(RunningStatus.FINISHED)

    def set_finished(self, idx: int) -> None:
        self.case_runners[idx].status = RunningStatus.FINISHED

    def _get_num_by_status(self, status: RunningStatus) -> int:
        return sum([1 for c in self.case_runners if c.status == status])

    def display(self) -> None:
        fmt = [TITLE_FORMAT]
        fmt.append(DATA_FORMAT % ("-" * 11, "-" * 12, "-" * 20, "-" * 7, "-" * 7))

        for f in self.case_runners:
            filters = f.ca.filters.filter_rate

            ds_str = f"{f.ca.dataset.data.name}-{f.ca.dataset.data.label}-{utils.numerize(f.ca.dataset.data.size)}"
            fmt.append(
                DATA_FORMAT
                % (
                    f.config.db_name,
                    f.ca.label.name,
                    ds_str,
                    filters,
                    self.task_label,
                ),
            )

        tmp_logger = logging.getLogger("no_color")
        for f in fmt:
            tmp_logger.info(f)
