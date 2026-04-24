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
	$(VECTORDBBENCH) tidb --host 127.0.0.1 --port 4000 --username root --password '' --db-name test --task-label tidb-spfresh --case-type Performance768D1M

search-1m-local:
	$(VECTORDBBENCH) tidb --host 127.0.0.1 --port 4000 --username root --password '' --db-name test --task-label tidb-spfresh --case-type Performance768D1M --skip-load --skip-drop-old

load-search-1m-remote:
	$(VECTORDBBENCH) tidb --host 10.2.12.79 --port 9090 --username root --password '' --db-name test --task-label tidb-spfresh --case-type Performance768D1M

search-1m-remote:
	$(VECTORDBBENCH) tidb --host 10.2.12.79 --port 9090 --username root --password '' --db-name test --task-label tidb-spfresh --case-type Performance768D1M --skip-load --skip-drop-old

load-search-10m-remote:
	$(VECTORDBBENCH) tidb --host 10.2.12.79 --port 9090 --username root --password '' --db-name test10m --task-label tidb-spfresh-10m --case-type Performance768D10M

search-10m-remote:
	$(VECTORDBBENCH) tidb --host 10.2.12.79 --port 9090 --username root --password '' --db-name test10m --task-label tidb-spfresh-10m --case-type Performance768D10M --skip-load --skip-drop-old

load-search-100m-remote:
	$(VECTORDBBENCH) tidb --host 10.2.12.79 --port 9090 --username root --password '' --db-name test100m --task-label tidb-spfresh-100m --case-type Performance768D10M

search-100m-remote:
	$(VECTORDBBENCH) tidb --host 10.2.12.79 --port 9090 --username root --password '' --db-name test100m --task-label tidb-spfresh-100m --case-type Performance768D10M --skip-load --skip-drop-old
