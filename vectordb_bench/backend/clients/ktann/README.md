# KTANN benchmark adapter

The adapter connects to a separately launched KTANN bridge over a Unix socket.
One bridge owns the Runtime and backend across VectorDBBench worker processes.
It supports full-load, single-tenant, IDs-only L2 and cosine cases, including
Cohere 1M numeric `id >= threshold` filters. `Performance768D1M1P` excludes 1%
of IDs; `Performance768D1M99P` excludes 99%. Label filters are unsupported.
Search options inherit KTANN defaults; `--leaf-beam` overrides the traversal
beam. Leaf Entry scans have no independent entry budget.

## Run

Build the bridge in the KTANN checkout:

```sh
cargo build --release -p ktann-benchmarks --bin ktann-vdbbench-bridge
# FoundationDB additionally requires --all-features and its native client/cluster.
target/release/ktann-vdbbench-bridge --backend rocksdb \
  --socket /tmp/ktann-bench.sock --database /tmp/ktann-bench-db \
  --report /tmp/ktann-bridge.json --bulk-workspace /tmp/ktann-bulk
```

In another terminal, from VectorDBBench with its dependencies installed:

```sh
pip install -e .
vectordbbench ktann --socket-path /tmp/ktann-bench.sock \
  --dataset-identity cohere-1m --case-type Performance768D1M \
  --load-concurrency 1 --insert-batch-size 50 --k 100
```

Use a fresh bridge, a new absolute `--bulk-workspace` directory, and a dedicated
backend location for each case. All imports use Bulk Build; there is no online
`batch_mutate` mode. Optimize seals the source, constructs and publishes the
index, then runs offline refinement before search is enabled.

Refinement defaults to two rounds and a 32 GiB input limit. Configure it with
`--refinement-rounds 0..5` and `--refinement-input-bytes BYTES`. Zero rounds still
refreshes centroids and loads the input; it is not a skip switch. The input limit
covers loaded vectors, IDs, centroids and topology, not additional planning memory
or total RSS. Exceeding it fails Optimize and leaves search disabled. Successful
repeated Optimize calls do not repeat construction or refinement.

Canonical insert time measures input capture; Optimize time includes Bulk Build,
refinement and readiness checks. Native reports include the requested refinement
settings and a separate `phases.refinement_seconds` measurement. Stop it with
Ctrl-C after the run; shutdown removes the benchmark index and socket. For
protocol details and FoundationDB setup, see the
[KTANN benchmark documentation](https://github.com/Lloyd-Pottiger/ktann/blob/main/benchmarks/README.md#vectordbbench-interoperability).

Canonical results are produced by VectorDBBench. Optional `--companion-dir DIR`
collects additional per-client timings; without it, client timing collection
and diagnostic file writes are disabled. Native bridge diagnostics are written
to its `--report` path.

## Tests

Build the bridge, then run from this checkout:

```sh
export KTANN_BRIDGE_BIN=/path/to/ktann/target/release/ktann-vdbbench-bridge
python -m unittest discover -s tests -p 'test_ktann_bridge.py' -v
KTANN_TEST_BACKEND=foundationdb python -m unittest discover \
  -s tests -p 'test_ktann_bridge.py' -v
```


Insert batches use binary float32 frames (at most 50 records), while control and
search requests remain JSON. Update this client and the bridge together; reset
now carries refinement controls. Vectors are converted and checked once with
NumPy. Receipt writes canonical source frames and sort runs; it does not commit
searchable batches. Duplicate IDs may fail at Optimize. Unknown outcomes are never
automatically replayed. Source snapshots remain caller-owned after shutdown.
