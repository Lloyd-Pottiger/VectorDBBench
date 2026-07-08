# Agent Instructions

## TiDB SPFRESH Makefile Targets

Before choosing a TiDB SPFRESH benchmark scenario, inspect the live Makefile in
this repository:

```sh
sed -n '1,180p' Makefile
```

The Makefile invokes the repository-local console script:

```make
VECTORDBBENCH := $(CURDIR)/.venv/bin/vectordbbench
```

Do not assume an activated shell environment controls the benchmark CLI. Verify
or repair `.venv` before running Makefile targets:

```sh
test -x .venv/bin/vectordbbench
.venv/bin/vectordbbench tidb --help >/dev/null
```

The `*-local` targets currently connect to TiDB on `127.0.0.1:4000` as `root`
with an empty password. The 1M targets use database `test`; the 10M targets use
database `test10m`.

Common target semantics:

| Target | Use |
| --- | --- |
| `load-search-1m-local` | 1M inline SPFRESH load, index creation, serial search, and concurrent search. |
| `load-search-1m-non-inline-local` | 1M load plus non-inline SPFRESH build. |
| `load-search-1m-split-local` | 1M load plus split SPFRESH build. |
| `build` | Build-only path on an existing 1M table; skips load/drop and concurrent search. |
| `search-1m-local` | Search-only path on an existing 1M table and index. |
| `load-search-10m-local` | 10M inline SPFRESH load/search in database `test10m`. |
| `load-search-10m-non-inline-local` | 10M load plus non-inline SPFRESH build. |
| `load-search-10m-split-local` | 10M load plus split SPFRESH build. |
| `search-10m-local` | Search-only path on an existing 10M table and index. |
| `delete-plain-1m-local` | Delete benchmark without building SPFRESH index. |
| `delete-spfresh-1m-local` | Delete benchmark with SPFRESH index. |
| `delete-compare-1m-local` | Runs both delete targets and compares results. |

Targets ending their recipe with `$(ARGS)` pass `ARGS` verbatim to
`vectordbbench tidb`. Use `ARGS` for TiDB SPFRESH options that are intentionally
not hard-coded in the target.

Example inline or non-inline 1M run:

```sh
make load-search-1m-local \
  ARGS='--spfresh-vector-index-param max_partition_size=256,write_beam_size=8,search_beam_size=320,oversample_factor=1'
```

Example build-only run:

```sh
make build \
  ARGS='--spfresh-vector-index-param max_partition_size=256,write_beam_size=8'
```

`--spfresh-vector-index-param` is passed through as the raw TiDB
`VECTOR_INDEX_PARAM` value. Create/build parameters such as
`min_partition_size`, `max_partition_size`, `rot_algorithm`, and
`write_beam_size` must be supplied when the index is created or built. Search
parameters such as `search_beam_size`, `oversample_factor`, and `read_only` are
persisted on the index and may be changed later for search-only probes.

Do not use `write_beam_size` as a search-only option in `ALTER INDEX ... SET
VECTOR_INDEX_PARAM`.

Use these tuned search values when comparing established partition sizes:

| `max_partition_size` | `search_beam_size` | `oversample_factor` |
| ---: | ---: | ---: |
| 128 | 512 | 1.25 |
| 256 | 320 | 1 |
| 384 | 224 | 1 |
| 512 | 192 | 1 |
| 1024 | 104 | 1 |

`search-*` targets reuse the existing index and cannot change create-time
parameters. To compare different partition or write-beam settings, recreate or
rebuild the index with a `load-search-*` target or `build`.

