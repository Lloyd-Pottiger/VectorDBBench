VENV_BIN := $(CURDIR)/.venv/bin
PYTHON := $(VENV_BIN)/python
VECTORDBBENCH := $(VENV_BIN)/vectordbbench

unittest:
	PYTHONPATH=$(CURDIR) $(PYTHON) -m pytest tests/test_dataset.py::TestDataSet::test_download_small -svv

format:
	PYTHONPATH=$(CURDIR) $(PYTHON) -m black vectordb_bench
	PYTHONPATH=$(CURDIR) $(PYTHON) -m ruff check vectordb_bench --fix

lint:
	PYTHONPATH=$(CURDIR) $(PYTHON) -m black vectordb_bench --check
	PYTHONPATH=$(CURDIR) $(PYTHON) -m ruff check vectordb_bench

load-search-1m-local:
	$(VECTORDBBENCH) tidb --host 127.0.0.1 --port 4000 --username root --password '' --db-name test --task-label tidb-spfresh --case-type Performance768D1M $(ARGS)

load-search-1m-non-inline-local:
	$(VECTORDBBENCH) tidb --host 127.0.0.1 --port 4000 --username root --password '' --db-name test --task-label tidb-spfresh-non-inline --case-type Performance768D1M --spfresh-build-mode non-inline $(ARGS)

load-search-1m-split-local:
	$(VECTORDBBENCH) tidb --host 127.0.0.1 --port 4000 --username root --password '' --db-name test --task-label tidb-spfresh-split --case-type Performance768D1M --spfresh-build-mode split --spfresh-split-ratio 0.8 $(ARGS)

build:
	$(VECTORDBBENCH) tidb --host 127.0.0.1 --port 4000 --username root --password '' --db-name test --task-label tidb-spfresh-build --case-type Performance768D1M --skip-load --skip-drop-old --build --skip-search-concurrent --spfresh-build-mode non-inline $(ARGS)

search-1m-local:
	$(VECTORDBBENCH) tidb --host 127.0.0.1 --port 4000 --username root --password '' --db-name test --task-label tidb-spfresh --case-type Performance768D1M --skip-load --skip-drop-old

load-search-10m-local:
	$(VECTORDBBENCH) tidb --host 127.0.0.1 --port 4000 --username root --password '' --db-name test10m --task-label tidb-spfresh-10m --case-type Performance768D10M $(ARGS)

load-search-10m-non-inline-local:
	$(VECTORDBBENCH) tidb --host 127.0.0.1 --port 4000 --username root --password '' --db-name test10m --task-label tidb-spfresh-10m-non-inline --case-type Performance768D10M --spfresh-build-mode non-inline $(ARGS)

load-search-10m-split-local:
	$(VECTORDBBENCH) tidb --host 127.0.0.1 --port 4000 --username root --password '' --db-name test10m --task-label tidb-spfresh-10m-split --case-type Performance768D10M --spfresh-build-mode split --spfresh-split-ratio 0.8 $(ARGS)

search-10m-local:
	$(VECTORDBBENCH) tidb --host 127.0.0.1 --port 4000 --username root --password '' --db-name test10m --task-label tidb-spfresh-10m --case-type Performance768D10M --skip-load --skip-drop-old

delete-plain-1m-local:
	$(VECTORDBBENCH) tidb --host 127.0.0.1 --port 4000 --username root --password '' --db-name test --task-label tidb-spfresh-delete-plain --case-type Performance768D1M --delete --skip-search-serial --skip-search-concurrent --skip-build-spfresh-index

delete-spfresh-1m-local:
	$(VECTORDBBENCH) tidb --host 127.0.0.1 --port 4000 --username root --password '' --db-name test --task-label tidb-spfresh-delete-indexed --case-type Performance768D1M --delete --skip-search-serial --skip-search-concurrent $(ARGS)

delete-compare-1m-local:
	$(MAKE) delete-plain-1m-local
	$(MAKE) delete-spfresh-1m-local
	$(PYTHON) scripts/compare_tidb_delete_results.py

load-search-1m-remote:
	$(VECTORDBBENCH) tidb --host 10.2.12.79 --port 9090 --username root --password '' --db-name test --task-label tidb-spfresh --case-type Performance768D1M $(ARGS)

search-1m-remote:
	$(VECTORDBBENCH) tidb --host 10.2.12.79 --port 9090 --username root --password '' --db-name test --task-label tidb-spfresh --case-type Performance768D1M --skip-load --skip-drop-old

load-search-10m-remote:
	$(VECTORDBBENCH) tidb --host 10.2.12.79 --port 9090 --username root --password '' --db-name test10m --task-label tidb-spfresh-10m --case-type Performance768D10M $(ARGS)

search-10m-remote:
	$(VECTORDBBENCH) tidb --host 10.2.12.79 --port 9090 --username root --password '' --db-name test10m --task-label tidb-spfresh-10m --case-type Performance768D10M --skip-load --skip-drop-old

load-search-100m-remote:
	$(VECTORDBBENCH) tidb --host 10.2.12.79 --port 9090 --username root --password '' --db-name test100m --task-label tidb-spfresh-100m --case-type Performance768D10M $(ARGS)

search-100m-remote:
	$(VECTORDBBENCH) tidb --host 10.2.12.79 --port 9090 --username root --password '' --db-name test100m --task-label tidb-spfresh-100m --case-type Performance768D10M --skip-load --skip-drop-old
