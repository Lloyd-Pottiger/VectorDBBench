# KTANN benchmark client

The KTANN adapter uses an explicitly launched Rust Unix-socket bridge. Build and
start it following the [KTANN benchmark guide](https://github.com/Lloyd-Pottiger/ktann/blob/main/benchmarks/README.md#vectordbbench).
Update the client and bridge together: their private protocol is unversioned.

Supported filters are `NonFilter`, `NumGE` on signed 64-bit `id`, and `StrEqual`
on UTF-8 string `labels`. Filtering uses native KTANN predicates, not result
post-processing. The CLI exposes Cohere 1M unfiltered/numeric cases, custom
unfiltered ANN cases, and `LabelFilterPerformanceCase` for CohereMedium with its
registered official label percentages. Multitenant and scalar payload retrieval
are not supported.

```sh
vectordbbench ktann --socket-path /tmp/ktann-bench.sock \
  --dataset-identity cohere-1m-labels --case-type LabelFilterPerformanceCase \
  --dataset-with-size-type 'Medium Cohere (768dim, 1M)' \
  --label-percentage 0.01 --load-concurrency 1 --insert-batch-size 50 --k 100 \
  --refinement-rounds 2 --refinement-memory-bytes 12884901888
```

The 12 GiB refinement admission budget covers this 1M workload; retained-input
accounting alone requires about 9.57 GiB before partition plans. It is not an
RSS reservation, and tree/serving budgets are separate. A 1 GiB refinement budget
is insufficient.

Here 0.01 means approximately 1% match `labels == "label_1p"`; use the official
`neighbors_labels_label_1p.parquet`, not numeric ground truth. DatasetManager
resolves the official files. The loader aligns `scalar_labels.parquet` with
record IDs and supplies `labels_data`. The adapter checks alignment, UTF-8 byte
limits (1024 per value), field names and types, then slices labels with the same
50-record boundaries as IDs and vectors. Empty strings are valid, with no case
folding or Unicode normalization. Invalid filter preparation raises without
replacing the last valid filter; callers must not ignore that error.

`with_scalar_labels` selects the schema at reset. Label data is required for every
row of a label-enabled index; an ID-only index rejects label inserts and queries.
Build a new index to change this setting. Every search sends a complete mutually
exclusive predicate, so label/numeric/unfiltered transitions cannot leave stale
conditions. The bridge owns schema validation and native predicate execution.

Binary insert frames have the existing `KTI`, big-endian u32 count/dimension,
little-endian i64 IDs and row-major float32 vectors, then a mandatory 0/1 label
presence byte. Present labels follow in row order as big-endian u16 byte lengths
and UTF-8 bytes. The whole frame remains bounded by 8 MiB. There is no automatic
replay after an unknown insert outcome.

Keep canonical recall/QPS/latency results, bridge and client companion reports,
the effective configuration, source revisions, dataset/ground-truth provenance
and environment together. `filter_at_close` in a client report describes only
the final filter state. Never present existing numeric results as label results
or imply a backend was measured when only its process tests ran.

Tests use the current bridge binary and isolated backend storage:

```sh
python -m unittest discover -s tests -p test_ktann_filters.py
KTANN_BRIDGE_BIN=/path/to/ktann-vdbbench-bridge KTANN_TEST_BACKEND=rocksdb \
  python -m unittest discover -s tests -p test_ktann_bridge.py
KTANN_BRIDGE_BIN=/path/to/ktann-vdbbench-bridge KTANN_TEST_BACKEND=foundationdb \
  python -m unittest discover -s tests -p test_ktann_bridge.py
```

FoundationDB needs the native client library and a running local cluster. Without
`KTANN_BRIDGE_BIN`, process tests skip and do not establish backend correctness.
