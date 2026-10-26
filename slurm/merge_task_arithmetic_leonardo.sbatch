#!/bin/bash
#SBATCH --job-name=oellm-taskarith
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# Task arithmetic (RL program §7, "combine specialists"): out = BASE + sum_i w_i * (MODEL_i - BASE), tensor by tensor in
# fp32, saved bf16 with BASE's shard layout, config and tokenizer, plus TEMPLATE as chat_template.jinja. Every model
# must share BASE's parameter names and shapes (same base, tokenizer and architecture). A model given as an RL run's
# outputs directory resolves to its newest step_* checkpoint at run time.
#   BASE=<dir> DELTAS="1.0:<dir or run outputs dir> 1.0:<dir>" DST=<dir> [TEMPLATE=<jinja>] sbatch ...
set -euo pipefail
[[ -n ${OELLM_WORK:-} ]] || source "${OELLM_ENV:-$HOME/.oellm/leonardo.env}"  # site settings: account, paths (leonardo.env.example)
container=/leonardo_work/OELLM_prod2026/container_images/post-training-25.10-transformers-olmo3-patch.sif
: "${BASE:?}" "${DELTAS:?}" "${DST:?}"
TEMPLATE=${TEMPLATE:-$OELLM_WORK/sft-migration-preflight/oellm_qwen3_two_stage_v2_1.jinja}
resolved=""
for item in $DELTAS; do
  weight=${item%%:*}; path=${item#*:}
  if ! ls "$path"/*.safetensors >/dev/null 2>&1; then
    path=$(ls -d "$path"/*_checkpoints/step_* | sort -t_ -k2 -V | tail -1)
  fi
  resolved="$resolved $weight:$path"
done
echo "base=$BASE deltas=$resolved"
srun singularity exec --nv -B /leonardo_work,/leonardo_scratch "$container" python - "$BASE" "$DST" "$TEMPLATE" $resolved <<'PY'
import json, os, shutil, sys
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file

base, dst, template = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
deltas = [(float(item.split(":", 1)[0]), Path(item.split(":", 1)[1])) for item in sys.argv[4:]]

def weight_map(path):
    index = path / "model.safetensors.index.json"
    if index.exists():
        return json.loads(index.read_text())["weight_map"]
    with safe_open(str(path / "model.safetensors"), framework="pt") as handle:
        return {key: "model.safetensors" for key in handle.keys()}

handles = {}
def tensor(path, mapping, key):
    shard = path / mapping[key]
    if shard not in handles:
        handles[shard] = safe_open(str(shard), framework="pt", device="cpu")
    return handles[shard].get_tensor(key)

base_map = weight_map(base)
maps = [weight_map(path) for _, path in deltas]
for (_, path), mapping in zip(deltas, maps):
    if set(mapping) != set(base_map):
        raise SystemExit(f"{path}: parameter names differ from the base")
dst.mkdir(parents=True, exist_ok=True)
for name in ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json"]:
    if (base / name).exists():
        shutil.copy2(base / name, dst / name)
shutil.copy2(template, dst / "chat_template.jinja")
by_shard = {}
for key, shard in base_map.items():
    by_shard.setdefault(shard, []).append(key)
norms = {str(path): 0.0 for _, path in deltas}
for shard, keys in sorted(by_shard.items()):
    out = {}
    for key in keys:
        b = tensor(base, base_map, key).float()
        acc = b.clone()
        for (w, path), mapping in zip(deltas, maps):
            t = tensor(path, mapping, key)
            if t.shape != b.shape:
                raise SystemExit(f"{key}: shape {tuple(t.shape)} vs {tuple(b.shape)} in {path}")
            d = t.float() - b
            norms[str(path)] += float(d.pow(2).sum())
            acc += w * d
        out[key] = acc.to(torch.bfloat16)
    save_file(out, str(dst / shard), metadata={"format": "pt"})
    print("wrote", shard, len(keys), flush=True)
record = {"base": str(base), "deltas": [{"weight": w, "model": str(p), "delta_l2": norms[str(p)] ** 0.5} for w, p in deltas],
          "formula": "base + sum_i w_i * (model_i - base)", "template": str(template)}
(dst / "merge.json").write_text(json.dumps(record, indent=1) + "\n")
print(json.dumps(record, indent=1))
PY
