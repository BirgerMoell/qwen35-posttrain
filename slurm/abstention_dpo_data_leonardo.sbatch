#!/bin/bash
#SBATCH --job-name=abstention-dpo-data
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
# Abstention DPO data (hallucination plan E4; scripts/build_abstention_dpo.py): eu_qa train + fabricated prompts,
# k=8 default-mode samples from POLICY, self-consistency pairs, pre-tokenized with the v2.1 template, plus GENERAL
# pre-tokenized Stage-4 pairs from pretok-v1 (same tokenizer and template) so DPO does not train on abstention alone.
#   POLICY=<model dir> WORK=<dir> [QA=0 (all)] [FABRICATED=600] [GENERAL=3000] [REFERENCE_WORK=<parent work dir>]
#   [KNOWN_SHARE=0.3333] sbatch slurm/abstention_dpo_data_leonardo.sbatch
# A WORK that already holds prompts.jsonl (e.g. copied from the parent's) is sampled as is.
set -euo pipefail
[[ -n ${OELLM_WORK:-} ]] || source "${OELLM_ENV:-$HOME/.oellm/leonardo.env}"
root=$OELLM_WORK/sft-migration-preflight
C=/leonardo_work/OELLM_prod2026/container_images/post-training-25.10-transformers-olmo3-patch.sif
PY=$OELLM_SCRATCH/oellm-rl/tmax-reproduction/.venv/bin/python
: "${POLICY:?}" "${WORK:?}"
template=$root/oellm_qwen3_two_stage_v2_1.jinja
cmp "$POLICY/chat_template.jinja" "$template"   # the DPO pipeline renders with v2.1; sample with the same template
run() { singularity exec --nv -B /leonardo_work,/leonardo_scratch "$C" env HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 \
          PYTHONPATH=$OELLM_SCRATCH/oellm-rl/oellm-rlvr/src "$PY" "$@"; }
mkdir -p "$WORK"
[[ -s $WORK/prompts.jsonl ]] || run "$root/build_abstention_dpo.py" prompts --work "$WORK" \
  --rl-qa "$OELLM_SCRATCH/oellm-rl/data/eu-tasks/rl_qa.parquet" --eval "$root/data/hallu-v1/eval.jsonl" \
  --qa "${QA:-0}" --fabricated "${FABRICATED:-600}"
[[ -s $WORK/samples.jsonl ]] || run "$root/build_abstention_dpo.py" sample --work "$WORK" --policy "$POLICY" --k 8
run "$root/build_abstention_dpo.py" pairs --work "$WORK" ${REFERENCE_WORK:+--reference-work "$REFERENCE_WORK"} \
  --known-share "${KNOWN_SHARE:-0.3333}"
rm -rf "$WORK/pretok"
run "$root/build_oellm_dpo_v1_pretok.py" --staged "$WORK/staged" --tokenizer "$POLICY" --template "$template" \
  --out "$WORK/pretok" --workers 8
# General pairs: a fixed random sample of pretok-v1 train rows (tokenizer and template hashes must match).
[[ ${GENERAL:-3000} == 0 ]] || run - "$OELLM_SCRATCH/oellm-dpo/pretok-v1" "$WORK/pretok" "${GENERAL:-3000}" <<'PY'
import json, sys, random
from pathlib import Path
import pyarrow.parquet as pq
src, dst, n = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
a, b = json.loads((src / "manifest.json").read_text()), json.loads((dst / "manifest.json").read_text())
assert (a["template_sha256"], a["tokenizer_json_sha256"]) == (b["template_sha256"], b["tokenizer_json_sha256"])
table = pq.read_table(sorted(src.glob("train-*.parquet"))[0])
idx = sorted(random.Random(20261103).sample(range(table.num_rows), n))
pq.write_table(table.take(idx), dst / "train-9000-general.parquet", compression="zstd")
b["splits"]["train"]["by_source"]["pretok-v1 general sample"] = {"staged": n, "kept": n}
b["splits"]["train"]["kept"] += n
b["general_sample"] = {"source": "pretok-v1 train shard 0", "rows": n, "seed": 20261103}
(dst / "manifest.json").write_text(json.dumps(b, indent=2) + "\n")
print("train rows", b["splits"]["train"]["kept"], "validation rows", b["splits"]["validation"]["kept"])
PY
