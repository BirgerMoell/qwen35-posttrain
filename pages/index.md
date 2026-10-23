# OpenEuroLLM 9B post-training

How the OpenEuroLLM 9B 256K base model is post-trained into a multilingual assistant on the Leonardo supercomputer:
**Stage 1 Instruct SFT → Stage 2 Reasoning SFT → Stage 3 Tool-use SFT → Stage 4 DPO → Stage 5 RL**.
The documents below are the working record of the run. They are updated as it progresses, and are written so the
pipeline can be reproduced.

| Document | What it contains |
|---|---|
| [Plan and decisions](OELLM_POST_TRAINING_ROADMAP.md) | goals, status dashboard, dated decisions with reasons, stage plans, evaluation protocol, budget, risks, next steps |
| [Playbook](OELLM_POST_TRAINING_PLAYBOOK.md) | the reproducible record: data, configs, hashes, job IDs, costs, results, pitfalls, exact commands |
| [RL run program](OELLM_RL_RUN_PROGRAM.md) | Stage 5 run by run: questions, run cards, results, stop rules, merging policy |
| [Stage 5 data and rewards](OELLM_STAGE5_RL_PLAN.md) | RL data pools with licenses, rewards, difficulty windows, environment tiers |

Code: [`qwen35-posttrain`](https://github.com/BirgerMoell/qwen35-posttrain) (SFT/DPO data builders, trainers, Slurm
launchers, evaluation scripts) and [`oellm-rlvr`](https://github.com/BirgerMoell/oellm-rlvr) (RL control plane).
Related benchmark: [EUROPA Reasoning](https://birgermoell.github.io/europa-reasoning/).

Paths in the documents use placeholders (`$WORK`, `$SCRATCH`, `<account>`); model checkpoints are private until
evaluation and license review.
