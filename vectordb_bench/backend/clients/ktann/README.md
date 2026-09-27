# KTANN benchmark adapter

The adapter connects to a separately launched KTANN bridge over a Unix socket.
One bridge owns the Runtime and backend across VectorDBBench worker processes.
It supports full-load, unfiltered, single-tenant, IDs-only L2 and cosine cases.
Search options inherit KTANN defaults; `--leaf-beam` overrides the traversal
beam. Leaf Entry scans have no independent entry budget.

## Run

Build the bridge in the KTANN checkout:

```sh
cargo build --release -p ktann-benchmarks --bin ktann-vdbbench-bridge
# FoundationDB additionally requires --all-features and its native client/cluster.
target/release/ktann-vdbbench-bridge --backend rocksdb \
  --socket /tmp/ktann-bench.sock --database /tmp/ktann-bench-db \
  --report /tmp/ktann-bridge.json
```

In another terminal, from VectorDBBench with its dependencies installed:

```sh
pip install -e .
vectordbbench ktann --socket-path /tmp/ktann-bench.sock \
  --dataset-identity cohere-1m --case-type Performance768D1M \
  --load-concurrency 1 --insert-batch-size 50 --k 100
```

Use a fresh bridge and dedicated backend location for each case. Stop it with
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
