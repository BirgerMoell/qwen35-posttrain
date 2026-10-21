#!/usr/bin/env python3
"""Re-apply the current category rules (and the meta-answer filter) to a built openeurollm-tool-use
dataset and its pre-tokenized pool in place; rows are matched by id. Updates manifest totals."""
import json
import sys
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from tooluse_adapters import categorize, check_not_meta

dataset_dir, pool_dir = Path(sys.argv[1]), Path(sys.argv[2])
moved, dropped, totals = Counter(), Counter(), Counter()
for path in sorted((dataset_dir / "data").glob("*.parquet")):
    t = pq.read_table(path)
    rows = t.to_pylist()
    keep, cats = [], []
    for r in rows:
        row = {"tools": json.loads(r["tools"]), "messages": r["messages"]}
        try:
            check_not_meta(row)
        except ValueError:
            keep.append(False); cats.append(r["category"]); dropped[r["source"]] += 1
            continue
        new = categorize(row)
        if new != r["category"]:
            moved[(r["category"], new)] += 1
        keep.append(True); cats.append(new)
    t = t.set_column(t.schema.get_field_index("category"), "category", pa.array(cats)).filter(pa.array(keep))
    pq.write_table(t, path, compression="zstd")
    cat_by_id = dict(zip(t["id"].to_pylist(), t["category"].to_pylist()))
    pool_path = pool_dir / path.name
    pt = pq.read_table(pool_path)
    ids = pt["id"].to_pylist()
    mask = [i in cat_by_id for i in ids]
    pt = pt.filter(pa.array(mask))
    pt = pt.set_column(pt.schema.get_field_index("category"), "category", pa.array([cat_by_id[i] for i in pt["id"].to_pylist()]))
    pq.write_table(pt, pool_path, compression="zstd")
    totals.update(f"cat:{c}" for c in t["category"].to_pylist())
    totals["kept"] += t.num_rows
    totals["tokens"] += pc.sum(pc.list_value_length(pt["input_ids"])).as_py() or 0
for d in (dataset_dir, pool_dir):
    m = json.loads((d / "manifest.json").read_text())
    m["totals"] = {**{k: v for k, v in m["totals"].items() if k.startswith("mode:")}, **dict(totals)}
    m["relabel"] = {"moved": {f"{a}->{b}": n for (a, b), n in moved.items()}, "dropped_meta_answers": dict(dropped)}
    (d / "manifest.json").write_text(json.dumps(m, indent=2, ensure_ascii=False) + "\n")
print(json.dumps({"moved": {f"{a}->{b}": n for (a, b), n in moved.most_common()}, "dropped": dict(dropped), "totals": dict(totals)}, indent=1))
