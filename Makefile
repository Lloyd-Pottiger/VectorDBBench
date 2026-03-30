unittest:
	PYTHONPATH=`pwd` python3 -m pytest tests/test_dataset.py::TestDataSet::test_download_small -svv

format:
	PYTHONPATH=`pwd` python3 -m black vectordb_bench
	PYTHONPATH=`pwd` python3 -m ruff check vectordb_bench --fix

lint:
	PYTHONPATH=`pwd` python3 -m black vectordb_bench --check
	PYTHONPATH=`pwd` python3 -m ruff check vectordb_bench

load-search-1m-local:
	vectordbbench tidb --host 127.0.0.1 --port 4123 --username root --password '' --db-name test --task-label tidb-spfresh --case-type Performance768D1M

search-1m-local:
	vectordbbench tidb --host 127.0.0.1 --port 4123 --username root --password '' --db-name test --task-label tidb-spfresh --case-type Performance768D1M --skip-load --skip-drop-old

load-search-1m-remote:
	vectordbbench tidb --host 10.2.12.79 --port 9090 --username root --password '' --db-name test --task-label tidb-spfresh --case-type Performance768D1M

search-1m-remote:
	vectordbbench tidb --host 10.2.12.79 --port 9090 --username root --password '' --db-name test --task-label tidb-spfresh --case-type Performance768D1M --skip-load --skip-drop-old

load-search-10m-remote:
	vectordbbench tidb --host 10.2.12.79 --port 9090 --username root --password '' --db-name test10m --task-label tidb-spfresh-10m --case-type Performance768D10M

search-10m-remote:
	vectordbbench tidb --host 10.2.12.79 --port 9090 --username root --password '' --db-name test10m --task-label tidb-spfresh-10m --case-type Performance768D10M --skip-load --skip-drop-old

load-search-100m-remote:
	vectordbbench tidb --host 10.2.12.79 --port 9090 --username root --password '' --db-name test100m --task-label tidb-spfresh-100m --case-type Performance768D10M

search-100m-remote:
	vectordbbench tidb --host 10.2.12.79 --port 9090 --username root --password '' --db-name test100m --task-label tidb-spfresh-100m --case-type Performance768D10M --skip-load --skip-drop-old
