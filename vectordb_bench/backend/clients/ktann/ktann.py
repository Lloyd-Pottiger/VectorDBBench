"""Thin pickleable adapter; all database ownership stays in the Rust process."""

import json
import operator
import os
import socket
import struct
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ...filter import FilterOp
from ...payload import PayloadProfile
from ..api import PartialInsertError, VectorDB

if TYPE_CHECKING:
    from .config import KTANNCaseConfig

MAX_FRAME = 8 << 20
MAX_BATCH = 50


def _label_bytes(value: str) -> bytes:
    """Validate the native String boundary without coercion or normalization."""
    if not isinstance(value, str):
        raise TypeError("label must be a string")
    encoded = value.encode("utf-8")
    if len(encoded) > 1024:
        raise ValueError("label exceeds 1024 UTF-8 bytes")
    return encoded


def _encode_insert(fields: dict) -> bytes:
    """Encode the bounded insert body once, without decimal float text."""
    if set(fields) not in ({"ids", "vectors"}, {"ids", "vectors", "labels"}):
        raise ValueError("insert requires IDs and vectors")
    ids = [operator.index(value) for value in fields["ids"]]
    if not 0 < len(ids) <= MAX_BATCH or any(not -(1 << 63) <= value < (1 << 63) for value in ids):
        raise ValueError("invalid insert IDs or batch size")
    with np.errstate(over="raise", invalid="raise"):
        vectors = np.asarray(fields["vectors"], dtype="<f4")
    if vectors.ndim != 2 or vectors.shape[0] != len(ids) or vectors.shape[1] == 0:
        raise ValueError("IDs/vectors shape mismatch")
    labels = fields.get("labels")
    tail = bytearray([labels is not None])
    if labels is not None:
        if len(labels) != len(ids):
            raise ValueError("IDs/labels shape mismatch")
        for label in labels:
            encoded = _label_bytes(label)
            tail.extend(struct.pack("!H", len(encoded)))
            tail.extend(encoded)
    if 11 + len(ids) * 8 + vectors.nbytes + len(tail) > MAX_FRAME:
        raise ValueError("bridge frame exceeds 8 MiB")
    if not np.isfinite(vectors).all():
        raise ValueError("nonfinite vector")
    return (b"KTI" + struct.pack("!II", len(ids), vectors.shape[1])
            + struct.pack(f"<{len(ids)}q", *ids) + vectors.tobytes(order="C") + tail)


class BridgeError(RuntimeError):
    """A terminal wire/KTANN error. Unknown inserts must never be auto-replayed."""


class Connection:
    """One sequential connection, opened inside the current worker only."""

    def __init__(self, path: str, *, collect_metrics: bool = False):
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.socket.settimeout(3605)
        try:
            self.socket.connect(path)
        except BaseException:
            self.socket.close()
            raise
        self.lock = threading.Lock()
        self.collect_metrics = collect_metrics
        if collect_metrics:
            self.calls = 0
            self.elapsed = self.encode = self.decode = self.ktann = 0.0
            self.latency_buckets = [0] * 64
            self.first_insert_ns = self.last_search_ns = None

    def close(self):
        self.socket.close()

    def _read(self, size: int):
        result = bytearray()
        while len(result) < size:
            chunk = self.socket.recv(size - len(result))
            if not chunk:
                raise BridgeError("bridge disconnected; insert outcome may be unknown")
            result.extend(chunk)
        return result

    def request(self, op: str, **fields):
        if self.collect_metrics and op == "insert" and self.first_insert_ns is None:
            self.first_insert_ns = time.monotonic_ns()
        measure = self.collect_metrics and op == "search"
        started = time.perf_counter() if measure else 0.0
        data = (_encode_insert(fields) if op == "insert" else
                json.dumps({"op": op, **fields}, separators=(",", ":"), allow_nan=False).encode())
        encoded = time.perf_counter() if measure else 0.0
        if len(data) > MAX_FRAME:
            raise ValueError("bridge frame exceeds 8 MiB")
        with self.lock:
            self.socket.sendall(struct.pack("!I", len(data)) + data)
            (size,) = struct.unpack("!I", self._read(4))
            if not 0 < size <= MAX_FRAME:
                raise BridgeError("invalid response length")
            data = self._read(size)
            decode_started = time.perf_counter() if measure else 0.0
            response = json.loads(data)
            finished = time.perf_counter() if measure else 0.0
            if not response.get("ok"):
                raise BridgeError(str(response.get("error")))
            result = response["result"]
            if measure:
                self.last_search_ns = time.monotonic_ns()
                self.calls += 1
                self.elapsed += finished - started
                self.encode += encoded - started
                self.decode += finished - decode_started
                self.ktann += result["ktann_seconds"]
                bucket = min(63, max(0, int((finished - started) * 1e9).bit_length()))
                self.latency_buckets[bucket] += 1
            return result

    def report(self):
        return {
            "label": "KTANN plus benchmark bridge — client companion",
            "first_insert_monotonic_ns": self.first_insert_ns,
            "last_search_monotonic_ns": self.last_search_ns,
            "pid": os.getpid(),
            "search_calls": self.calls,
            "round_trip_seconds_sum": self.elapsed,
            "python_encode_seconds": self.encode,
            "python_decode_seconds": self.decode,
            "ktann_seconds_sum": self.ktann,
            "bridge_ipc_and_queue_seconds": max(0, self.elapsed - self.ktann),
            "latency_log2_nanoseconds_histogram": self.latency_buckets,
        }


class KTANN(VectorDB):
    supported_filter_types = [FilterOp.NonFilter, FilterOp.NumGE, FilterOp.StrEqual]

    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: "KTANNCaseConfig",
        collection_name: str = "vdbbench",
        drop_old: bool = False,
        with_scalar_labels: bool = False,
        **kwargs,
    ):
        if kwargs.get("multitenant_tenant_labels"):
            raise ValueError("KTANN bridge supports single-tenant IDs-only ANN")
        if not drop_old:
            raise ValueError("KTANN benchmark requires a fresh bridge and full load (drop_old=True)")
        self.name = "KTANN plus benchmark bridge"
        self._filter = None
        self._with_scalar_labels = with_scalar_labels
        self.dim = dim
        self.db_config = dict(db_config)
        self._connection = None
        self._owner_pid = None
        metric = str(db_case_config.metric_type)
        if metric not in {"L2", "COSINE"}:
            raise ValueError("only L2 and COSINE are supported")
        connection = Connection(self.db_config["socket_path"])
        try:
            connection.request(
                "reset",
                dimension=dim,
                metric=metric,
                with_scalar_labels=with_scalar_labels,
                dataset=self.db_config["dataset"],
                **db_case_config.index_param(),
                **db_case_config.search_param(),
            )
        finally:
            connection.close()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_connection"] = None
        state["_owner_pid"] = None
        return state

    @contextmanager
    def init(self):
        if self._connection is not None:
            raise RuntimeError("nested init is unsupported")
        companion_dir = self.db_config.get("companion_dir")
        self._connection = Connection(self.db_config["socket_path"], collect_metrics=bool(companion_dir))
        self._owner_pid = os.getpid()
        try:
            yield
        finally:
            connection, self._connection = self._connection, None
            self._owner_pid = None
            connection.close()
            if companion_dir and (connection.calls or connection.first_insert_ns is not None):
                directory = Path(companion_dir)
                directory.mkdir(parents=True, exist_ok=True)
                target = directory / f"client-{os.getpid()}-{uuid.uuid4().hex}.json"
                report = connection.report()
                report["with_scalar_labels"] = self._with_scalar_labels
                report["filter_at_close"] = self._filter
                target.write_text(json.dumps(report, indent=2))

    def _request(self, op: str, **fields):
        if self._connection is None or self._owner_pid != os.getpid():
            raise RuntimeError("use init() inside each worker process")
        return self._connection.request(op, **fields)

    def prepare_filter(self, filters: dict):
        """Replace the entire predicate only after validating the new condition."""
        if filters.type == FilterOp.NonFilter:
            predicate = None
        elif filters.type == FilterOp.NumGE and filters.int_field == "id":
            if isinstance(filters.int_value, bool):
                raise ValueError("filter threshold must be an integer, not bool")
            threshold = operator.index(filters.int_value)
            if not -(1 << 63) <= threshold < (1 << 63):
                raise ValueError("filter threshold must be a signed 64-bit integer")
            predicate = {"type": "num_ge", "value": threshold}
        elif filters.type == FilterOp.StrEqual and filters.label_field == "labels":
            if not self._with_scalar_labels:
                raise ValueError("label filter requires scalar labels at reset")
            _label_bytes(filters.label_value)
            predicate = {"type": "str_equal", "value": filters.label_value}
        else:
            raise ValueError("only numeric id >= threshold and string labels == value filters are supported")
        self._filter = predicate

    def insert_embeddings(
        self, embeddings: list[list[float]], metadata: list[int], labels_data: list[str] | None = None, **kwargs
    ):
        inserted = 0
        # Local validation failures must return the same confirmed-prefix error as wire failures.
        try:
            if kwargs.get("tenant") is not None:
                raise ValueError("tenants are unsupported")  # noqa: TRY301
            if len(embeddings) != len(metadata):
                raise ValueError("IDs/vectors length mismatch")  # noqa: TRY301
            if (labels_data is not None) != self._with_scalar_labels:
                raise ValueError("labels must match the schema selected at reset")  # noqa: TRY301
            if labels_data is not None:
                if len(labels_data) != len(metadata):
                    raise ValueError("IDs/labels shape mismatch")  # noqa: TRY301
                for label in labels_data:
                    _label_bytes(label)
            # Split at the client boundary; each wire batch is a bounded atomic import.
            for offset in range(0, len(metadata), MAX_BATCH):
                ids = metadata[offset : offset + MAX_BATCH]
                vectors = embeddings[offset : offset + MAX_BATCH]
                if any(len(row) != self.dim for row in vectors):
                    raise ValueError("wrong vector dimension")  # noqa: TRY301
                labels = labels_data[offset : offset + MAX_BATCH] if labels_data is not None else None
                result = self._request("insert", ids=ids, vectors=vectors, labels=labels)
                if result["inserted"] != len(ids):
                    raise BridgeError("incomplete insert response")  # noqa: TRY301
                inserted += result["inserted"]
        except Exception as error:
            # The upstream loader recognizes non_retryable, preserving known prefixes.
            return inserted, PartialInsertError(str(error), inserted_count=inserted, cause=error)
        return inserted, None

    def optimize(self, data_size: int | None = None):
        if data_size is None:
            raise ValueError("optimize requires the expected dataset size")
        self._request("optimize", records=int(data_size))

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        payload_profile: PayloadProfile = PayloadProfile.IDS_ONLY,
        tenant: str | None = None,
    ):
        if payload_profile != PayloadProfile.IDS_ONLY or tenant is not None:
            raise ValueError("only IDs-only, single-tenant search is supported")
        return self._request("search", vector=[float(v) for v in query], k=int(k), filter=self._filter)["ids"]
