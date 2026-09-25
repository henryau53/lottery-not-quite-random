from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.lottery_not_quite_random.utils import load_json


def main():
    records = load_json(Path("./data/dlt/raw/draws.json"))

    results = []
    for r in records:
        results.extend(r.get("prizeLevelList"))

    table = pa.Table.from_pylist(results)
    pq.write_table(table, "output.parquet")


if __name__ == "__main__":
    main()
