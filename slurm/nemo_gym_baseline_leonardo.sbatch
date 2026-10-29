#!/bin/bash
#SBATCH --job-name=nemo-gym-baseline
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
# Baseline of a checkpoint on NVIDIA NeMo Gym RL environments (HF "rl-environment" datasets), scored by NeMo Gym's own
# resources servers, to pick environments with a useful learning signal (20-60 % success) for later RL windows.
#   GPU 0: vLLM OpenAI server with the policy (served as "agora", direct mode, hermes tool parser).
#   GPU 1: Ollama with the judge gemma4:31b (EUROPA judge digest), OpenAI-compatible API, for the environments that use
#          an LLM judge (VerifIF: SysBench / CFBench; abstention).
# Environments (sample sizes via N_*): workplace_assistant (multi-step tools, validation split), Structured Outputs v2
# (direct_generation: JSON/YAML/XML schemas), SysBench (system-message following), CFBench (constraint following;
# en/ar/hi/zh/ja/ko), QA-Abstention (\boxed{[IDK]} when unsure).
# NeMo Gym (github.com/NVIDIA-NeMo/Gym) is installed in $GYM with per-server .venv links to its own venv (offline nodes:
# skip_venv_if_present). Data was downloaded to $GYM/oellm-baseline/data on the login node.
#   POLICY=<model dir> NAME=<run name> sbatch slurm/nemo_gym_baseline_leonardo.sbatch
set -euo pipefail
[[ -n ${OELLM_WORK:-} ]] || source "${OELLM_ENV:-$HOME/.oellm/leonardo.env}"  # site settings: account, paths (leonardo.env.example)
: "${POLICY:?policy model dir}" "${NAME:?run name}"
C=/leonardo_work/OELLM_prod2026/container_images/post-training-25.10-transformers-olmo3-patch.sif
VLLM_PY=$OELLM_SCRATCH/oellm-rl/tmax-reproduction/.venv/bin/python
tools=$OELLM_WORK/tools
GYM=$OELLM_SCRATCH/nemo-gym
B=$GYM/oellm-baseline
OUT=$B/runs/$NAME
mkdir -p "$OUT"
N_WA=${N_WA:-200} N_SO=${N_SO:-300} N_SYS=${N_SYS:-200} N_CF=${N_CF:-200} N_ABS=${N_ABS:-300}
JREV=sha256:6316f0629137b426c9d9b853ffc4c8209589f30ee39aebede6285096c0ff47e7
export HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 UV_OFFLINE=1 RAY_USAGE_STATS_ENABLED=0

# 1. Inputs: a fixed random sample per environment, agent_ref pointed at the agent we run.
singularity exec -B /leonardo_work,/leonardo_scratch "$C" python3 - "$B" "$OUT" "$N_WA" "$N_SO" "$N_SYS" "$N_CF" "$N_ABS" <<'PY'
import json, random, sys
import pyarrow.parquet as pq
B, OUT = sys.argv[1], sys.argv[2]
n_wa, n_so, n_sys, n_cf, n_abs = map(int, sys.argv[3:8])
rng = random.Random(20261028)
def sample(rows, n, agent, path):
    rows = list(rows); rng.shuffle(rows); rows = rows[:n]
    for r in rows:
        r["agent_ref"] = {"type": "responses_api_agents", "name": agent}
    with open(f"{OUT}/{path}", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(path, len(rows))
jl = lambda p: [json.loads(l) for l in open(f"{B}/data/{p}")]
sample(jl("workplace_assistant_validation.jsonl"), n_wa, "workplace_assistant_simple_agent", "in_workplace_assistant.jsonl")
sample(jl("sysbench.jsonl"), n_sys, "verifif_simple_agent", "in_sysbench.jsonl")
sample(jl("cfbench.jsonl"), n_cf, "verifif_simple_agent", "in_cfbench.jsonl")
sample(jl("abstention.jsonl"), n_abs, "abstention_simple_agent", "in_abstention.jsonl")
so = pq.read_table(f"{B}/data/so_direct.parquet").to_pylist()
sample(so, n_so, "structured_outputs_simple_agent", "in_structured_outputs.jsonl")
PY

# 2. Policy server (GPU 0) and judge (GPU 1).
# The API server imports OpenCV, whose loader prepends its package dir to sys.path; a *spawned* engine-core process
# inherits that path and resolves `import typing` to cv2/typing (libxcb ImportError). Fork the engine instead.
cd "$OUT"
CUDA_VISIBLE_DEVICES=0 singularity exec --nv -B /leonardo_work,/leonardo_scratch "$C" \
  env PYTHONPATH=$OELLM_WORK/europa/pylibs VLLM_WORKER_MULTIPROC_METHOD=fork "$VLLM_PY" -m vllm.entrypoints.openai.api_server \
  --model "$POLICY" --served-model-name agora --host 127.0.0.1 --port 8000 --dtype bfloat16 --max-model-len 32768 \
  --gpu-memory-utilization 0.88 --enable-auto-tool-choice --tool-call-parser hermes --enforce-eager > "$OUT/vllm.log" 2>&1 &
# Ollama 0.32 uses the Vulkan backend here, which ignores CUDA_VISIBLE_DEVICES, sees both GPUs and picks a 262K default
# context: loading the judge then fails next to vLLM on GPU 0 (HTTP 500 on every OpenAI-style judge call). Hide GPU 0
# from Vulkan and cap the context (judge prompts are short).
export OLLAMA_MODELS=$tools/ollama-models-baseline OLLAMA_HOST=127.0.0.1:11434 OLLAMA_NUM_PARALLEL=8 OLLAMA_KEEP_ALIVE=-1
export OLLAMA_CONTEXT_LENGTH=16384 GGML_VK_VISIBLE_DEVICES=1
CUDA_VISIBLE_DEVICES=1 $tools/ollama-0.32.8/bin/ollama serve > "$OUT/ollama.log" 2>&1 &
until curl -sf http://127.0.0.1:11434/api/version > /dev/null; do sleep 2; done
got=$(curl -s http://127.0.0.1:11434/api/tags | python3 -c "import json,sys; print([m['digest'] for m in json.load(sys.stdin)['models'] if m['name']=='gemma4:31b'][0])")
[[ "sha256:$got" == "$JREV" ]] || { echo "judge digest $got != $JREV" >&2; exit 2; }
curl -s http://127.0.0.1:11434/api/generate -d '{"model":"gemma4:31b","prompt":"ok","stream":false,"options":{"num_predict":1}}' > /dev/null
for i in $(seq 1 120); do curl -sf http://127.0.0.1:8000/v1/models > /dev/null && break; sleep 10; done
curl -sf http://127.0.0.1:8000/v1/models > /dev/null || { echo "vLLM did not come up" >&2; tail -30 "$OUT/vllm.log" >&2; exit 3; }
echo "servers up $(date)"

# 3. One NeMo Gym evaluation per environment. A custom input file needs --no-serve, so the Gym servers (resources
# server, agent, model proxy) are started per environment with `gym env start`, awaited via the head server's
# /readyz (port 11000), used by `gym eval run --no-serve`, then stopped.
cd "$GYM"
run_env() {  # name config agent input
  local name=$1 config=$2 agent=$3 input=$4
  echo "== $name $(date)"
  setsid .venv/bin/gym env start --config "$config" --config "$B/oellm.yaml" --model-type vllm_model \
    > "$OUT/serve_$name.log" 2>&1 &
  local pgid=$!
  local ok=0
  for i in $(seq 1 120); do
    curl -sf http://127.0.0.1:11000/readyz > /dev/null && { ok=1; break; }
    kill -0 $pgid 2> /dev/null || break
    sleep 5
  done
  if [[ $ok == 1 ]]; then
    .venv/bin/gym eval run --no-serve --agent "$agent" --input "$OUT/$input" --output "$OUT/rollouts_$name.jsonl" \
      --num-repeats 1 --concurrency 16 --temperature 0.6 --top-p 0.95 --max-output-tokens 4096 --no-health-check \
      > "$OUT/gym_$name.log" 2>&1 || echo "gym eval failed for $name (see $OUT/gym_$name.log)"
    tail -25 "$OUT/gym_$name.log"
  else
    echo "Gym servers for $name did not become ready (see $OUT/serve_$name.log)"; tail -20 "$OUT/serve_$name.log"
  fi
  kill -TERM -- -$pgid 2> /dev/null || true
  for i in $(seq 1 30); do curl -sf http://127.0.0.1:11000/readyz > /dev/null || break; sleep 2; done
  kill -KILL -- -$pgid 2> /dev/null || true
}
run_env workplace_assistant resources_servers/workplace_assistant/configs/workplace_assistant.yaml workplace_assistant_simple_agent in_workplace_assistant.jsonl
run_env structured_outputs resources_servers/structured_outputs/configs/structured_outputs_json_yaml_xml_v1.yaml structured_outputs_simple_agent in_structured_outputs.jsonl
run_env sysbench resources_servers/verifif/configs/verifif.yaml verifif_simple_agent in_sysbench.jsonl
run_env cfbench resources_servers/verifif/configs/verifif.yaml verifif_simple_agent in_cfbench.jsonl
run_env abstention resources_servers/abstention/configs/abstention.yaml abstention_simple_agent in_abstention.jsonl

# 4. Summary: mean reward per environment (and per language / category where present).
python3 - "$OUT" <<'PY'
import json, glob, os, sys, collections
out = sys.argv[1]
summary = {}
for path in sorted(glob.glob(f"{out}/rollouts_*.jsonl")):
    name = os.path.basename(path)[len("rollouts_"):-len(".jsonl")]
    if name.endswith("_materialized_inputs"):
        continue
    rows = [json.loads(l) for l in open(path)]
    # Rows NeMo Gym's own schema check rejects (dataset/verifier version mismatch, e.g. CFBench instructions missing a
    # required argument) are data errors, not model failures: count them separately.
    invalid = [r for r in rows if "'instruction': 'schema_validation', 'status': 'Failed'" in str(r.get("validation_results"))]
    rows = [r for r in rows if r not in invalid]
    rewards = [r.get("reward") for r in rows if isinstance(r.get("reward"), (int, float))]
    by_cat = collections.defaultdict(list)
    for r in rows:
        if isinstance(r.get("reward"), (int, float)):
            by_cat[str(r.get("category") or r.get("domain") or r.get("schema_type") or "-")].append(r["reward"])
    summary[name] = {"n": len(rows), "scored": len(rewards), "invalid_rows": len(invalid), "mean_reward": sum(rewards) / max(1, len(rewards)),
                     "solved_share": sum(1 for x in rewards if x >= 0.999) / max(1, len(rewards)),
                     "by_category": {k: round(sum(v) / len(v), 3) for k, v in sorted(by_cat.items())}}
json.dump(summary, open(f"{out}/summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
PY
