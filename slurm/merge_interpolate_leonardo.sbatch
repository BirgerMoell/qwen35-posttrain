#!/bin/bash
#SBATCH --job-name=oellm-merge
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# Linear weight interpolation (roadmap merging policy, DPO fallback): out = (1 - ALPHA) * BASE + ALPHA * TUNED,
# tensor by tensor in fp32, saved bf16 with TUNED's shard layout, config, tokenizer and chat template. Both models
# must have identical parameter names and shapes (same architecture and tokenizer).
#   BASE=<dir> TUNED=<dir> ALPHA=0.5 DST=<dir> sbatch slurm/merge_interpolate_leonardo.sbatch
set -euo pipefail
[[ -n ${OELLM_WORK:-} ]] || source "${OELLM_ENV:-$HOME/.oellm/leonardo.env}"  # site settings: account, paths (leonardo.env.example)
container=/leonardo_work/OELLM_prod2026/container_images/post-training-25.10-transformers-olmo3-patch.sif
: "${BASE:?}" "${TUNED:?}" "${ALPHA:?}" "${DST:?}"
srun singularity exec --nv -B /leonardo_work,/leonardo_scratch "$container" python - "$BASE" "$TUNED" "$ALPHA" "$DST" <<'PY'
import json, shutil, sys
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file

base, tuned, alpha, dst = Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3]), Path(sys.argv[4])
assert 0.0 <= alpha <= 1.0
def index(path):
    return json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
base_map, tuned_map = index(base), index(tuned)
if set(base_map) != set(tuned_map):
    raise SystemExit(f"parameter names differ: {sorted(set(base_map) ^ set(tuned_map))[:5]}")
dst.mkdir(parents=True, exist_ok=True)
for name in ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
             "model.safetensors.index.json"]:
    if (tuned / name).exists():
        shutil.copy2(tuned / name, dst / name)
handles = {}
def tensor(root, mapping, key):
    shard = root / mapping[key]
    if shard not in handles:
        handles[shard] = safe_open(str(shard), framework="pt", device="cpu")
    return handles[shard].get_tensor(key)
by_shard = {}
for key, shard in tuned_map.items():
    by_shard.setdefault(shard, []).append(key)
for shard, keys in sorted(by_shard.items()):
    out = {}
    for key in keys:
        b, t = tensor(base, base_map, key), tensor(tuned, tuned_map, key)
        if b.shape != t.shape:
            raise SystemExit(f"{key}: shape {tuple(b.shape)} vs {tuple(t.shape)}")
        out[key] = torch.lerp(b.float(), t.float(), alpha).to(torch.bfloat16)
    save_file(out, str(dst / shard), metadata={"format": "pt"})
    print("wrote", shard, len(keys), flush=True)
(dst / "merge.json").write_text(json.dumps({"base": str(base), "tuned": str(tuned), "alpha": alpha,
                                            "formula": "(1 - alpha) * base + alpha * tuned"}, indent=1) + "\n")
print("merged ->", dst)
PY
