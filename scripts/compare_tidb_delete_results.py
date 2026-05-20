import json
from pathlib import Path


RESULT_DIR = Path("vectordb_bench/results/TiDB")
TASK_LABELS = {
    "plain": "tidb-spfresh-delete-plain",
    "indexed": "tidb-spfresh-delete-indexed",
}


def latest_result(task_label: str) -> Path:
    files = sorted(
        RESULT_DIR.glob(f"result_*_{task_label}_tidb.json"),
        key=lambda path: path.stat().st_mtime,
    )
    if not files:
        msg = f"missing result file for {task_label} under {RESULT_DIR}"
        raise SystemExit(msg)
    return files[-1]


def load_metrics(task_label: str) -> dict:
    result_path = latest_result(task_label)
    payload = json.loads(result_path.read_text())
    return payload["results"][0]["metrics"]


def main() -> None:
    plain = load_metrics(TASK_LABELS["plain"])
    indexed = load_metrics(TASK_LABELS["indexed"])

    plain_delete = plain["delete_duration"]
    indexed_delete = indexed["delete_duration"]
    diff = indexed_delete - plain_delete
    ratio = (plain_delete / indexed_delete) if indexed_delete else float("inf")

    print("Delete benchmark comparison")
    print(f"  no index delete_duration      : {plain_delete}s")
    print(f"  with SPFRESH delete_duration  : {indexed_delete}s")
    print(f"  difference(indexed - plain)   : {diff}s")
    print(f"  plain/indexed ratio           : {ratio}")


if __name__ == "__main__":
    main()
