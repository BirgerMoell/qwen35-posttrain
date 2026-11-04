#!/bin/bash
#SBATCH --job-name=mopd-kl
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=02:30:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
# MOPD same-origin check (scripts/mopd_kl_probe.py): the student samples on a per-domain prompt mix, every teacher
# scores the same tokens; reports the per-token reverse KL per domain and mode.
#   STUDENT=<dir> TEACHERS="name=dir ..." [N=48] sbatch slurm/mopd_kl_probe_leonardo.sbatch
set -euo pipefail
[[ -n ${OELLM_WORK:-} ]] || source "${OELLM_ENV:-$HOME/.oellm/leonardo.env}"
root=$OELLM_WORK/sft-migration-preflight
C=/leonardo_work/OELLM_prod2026/container_images/post-training-25.10-transformers-olmo3-patch.sif
PY=$OELLM_SCRATCH/oellm-rl/tmax-reproduction/.venv/bin/python
: "${STUDENT:?}" "${TEACHERS:?}"
out=$root/eval/mopd-kl/${NAME:-e8}
mkdir -p "$out"
run() { singularity exec --nv -B /leonardo_work,/leonardo_scratch "$C" env HF_HUB_OFFLINE=1 "$PY" "$@"; }
[[ -s $out/prompts.jsonl ]] || run - "$OELLM_SCRATCH" "$out/prompts.jsonl" "${N:-48}" <<'PY'
import json, random, sys
import pyarrow.parquet as pq
scratch, dst, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
rng = random.Random(20261104)
rows = pq.read_table(f"{scratch}/oellm-rl/data/e8/train-mixed.parquet").to_pylist()
out = []
groups = {"eu_doc": "documents", "code_stdio": "code", "tool_call": "tools", "math": "math",
          "multilingual_math": "math", "ifeval_direct": "instruction_following", "ifeval_think": "instruction_following"}
by = {}
for r in rows:
    domain = groups.get(r["dataset"]) or ("factual_qa" if r["dataset"] == "eu_qa" and not r["tools"] else None)
    if domain:
        by.setdefault(domain, []).append(r)
for domain, items in sorted(by.items()):
    rng.shuffle(items)
    for r in items[:n]:
        msgs = [{"role": m["role"], "content": m["content"]} for m in r["messages"]]
        out.append({"domain": domain, "mode": "default" if r["messages"][-1]["rl_mode"] == "default" else "thinking",
                    "messages": msgs})
for domain, path in (("safety", f"{scratch}/oellm-dpo/safety-v1/prompts.jsonl"),
                     ("identity", f"{scratch}/oellm-dpo/identity-v2/prompts.jsonl")):
    items = [json.loads(l) for l in open(path, encoding="utf-8")]
    rng.shuffle(items)
    for r in items[:n]:
        out.append({"domain": domain, "mode": "default", "messages": [{"role": "user", "content": r["prompt"]}]})
with open(dst, "w", encoding="utf-8") as f:
    f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in out)
print("prompts", len(out), {d: sum(r["domain"] == d for r in out) for d in sorted({r["domain"] for r in out})})
PY
[[ -s $out/student.jsonl ]] || run "$root/mopd_kl_probe.py" gen --student "$STUDENT" --prompts "$out/prompts.jsonl" --out "$out"
for t in $TEACHERS; do run "$root/mopd_kl_probe.py" score --teacher "$t" --out "$out"; done
run "$root/mopd_kl_probe.py" report --out "$out"
