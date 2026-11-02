#!/bin/bash
#SBATCH --job-name=stage4b-assemble
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
# Stage 4b DPO data: merge pre-tokenized pair sets (each built with the v2.1 template and the same tokenizer) plus a
# fixed sample of general Stage-4 pairs into one DPO_DATA directory with a merged manifest.
#   OUT=<dir> PARTS="<pretok dir> ..." [GENERAL=4000] sbatch slurm/stage4b_assemble_leonardo.sbatch
set -euo pipefail
[[ -n ${OELLM_WORK:-} ]] || source "${OELLM_ENV:-$HOME/.oellm/leonardo.env}"
C=/leonardo_work/OELLM_prod2026/container_images/post-training-25.10-transformers-olmo3-patch.sif
PY=$OELLM_SCRATCH/oellm-rl/tmax-reproduction/.venv/bin/python
: "${OUT:?}" "${PARTS:?}"
singularity exec -B /leonardo_work,/leonardo_scratch "$C" "$PY" - "$OUT" "${GENERAL:-4000}" "$OELLM_SCRATCH/oellm-dpo/pretok-v1" $PARTS <<'PY'
import json, random, sys
from pathlib import Path
import pyarrow.parquet as pq
out, n_general, general, parts = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3]), [Path(p) for p in sys.argv[4:]]
out.mkdir(parents=True, exist_ok=True)
for old in out.glob("*.parquet"):
    old.unlink()
manifests = [json.loads((p / "manifest.json").read_text()) for p in parts + [general]]
keys = {(m["template_sha256"], m["tokenizer_json_sha256"], m["max_length"]) for m in manifests}
assert len(keys) == 1, f"template / tokenizer / max_length differ: {keys}"
merged = {**{k: manifests[0][k] for k in ("kind", "max_length", "template", "template_sha256", "tokenizer_json_sha256", "dropped_tag_ids")},
          "parts": [str(p) for p in parts], "splits": {"train": {"by_source": {}, "kept": 0}, "validation": {"by_source": {}, "kept": 0}}}
for i, p in enumerate(parts):
    for split in ("train", "validation"):
        for j, shard in enumerate(sorted(p.glob(f"{split}-*.parquet"))):
            t = pq.read_table(shard)
            pq.write_table(t, out / f"{split}-{i:02d}{j:02d}.parquet", compression="zstd")
            merged["splits"][split]["kept"] += t.num_rows
        for src, c in json.loads((p / "manifest.json").read_text())["splits"].get(split, {}).get("by_source", {}).items():
            merged["splits"][split]["by_source"][src] = c
if n_general:
    t = pq.read_table(sorted(general.glob("train-*.parquet"))[0])
    t = t.take(sorted(random.Random(20261103).sample(range(t.num_rows), n_general)))
    pq.write_table(t, out / "train-9900-general.parquet", compression="zstd")
    merged["splits"]["train"]["by_source"]["pretok-v1 general sample"] = {"staged": n_general, "kept": n_general}
    merged["splits"]["train"]["kept"] += n_general
(out / "manifest.json").write_text(json.dumps(merged, indent=2) + "\n")
print(json.dumps({s: merged["splits"][s]["kept"] for s in merged["splits"]}), {s: c["kept"] for s, c in merged["splits"]["train"]["by_source"].items()})
PY
