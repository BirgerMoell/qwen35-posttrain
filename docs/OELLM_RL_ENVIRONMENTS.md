# RL environments: quality bar and roadmap

Status 2026-11-04. How we build, qualify and add RL environments for the OpenEuroLLM 9B assistant. It draws on
three sources:

- our own incidents in Stage 5 ([RL run program](OELLM_RL_RUN_PROGRAM.md));
- the [svenska-rl environment contract](https://github.com/BirgerMoell/svenska-rl/blob/main/docs/environment-contract.md);
- the Hugging Face guide [*The ultimate guide to RL environments*](https://huggingface.co/spaces/AdithyaSK/rl-environments-guide)
  (by AdithyaSK, May 2026), which compares six environment frameworks: OpenEnv, ORS, NeMo Gym, Verifiers, SkyRL Gym
  and GEM.

## 1. Why environments matter now

The guide's starting point is that RL gains now come mainly from the breadth and quality of environments, not from
the algorithm. Qwen3.5 attributes most of its post-training gain over Qwen3 to "extensive scaling of virtually all
RL tasks and environments".

For us, two further points follow:

- **The environment decides what the model learns, flaws included.** Our worst RL result (W3) was an environment
  defect, not an optimiser problem. On the eu_qa pool, always answering `UNKNOWN` scored better than trying, and
  W3 learned exactly that.
- **Environments are where experts are trained.** Under the multi-teacher distillation plan (MOPD, run program
  §10.7), each environment family trains one expert, and the student copies that expert token by token. A
  rewardable shortcut therefore spreads through distillation.

## 2. What an environment is (our vocabulary)

Following the guide's anatomy, an environment has five parts:

| Part | Question | Ours |
|---|---|---|
| **Tasks** | what is the model asked, with what hidden ground truth | parquet rows: `messages`, `ground_truth`, `dataset`, optional `tools` |
| **Tools** | what can it call, and what comes back | per-row `tools` column; servers on 127.0.0.1 per node (`code_api`, `wiki_search`) |
| **Harness** | who drives the turns, and the budget | the trainer (TMAX `grpo_fast`): generate → parse Hermes calls → execute → append observation; `max_steps` |
| **Reward** | how the result is scored | verifiers in oellm-rlvr (`compat.patch_*`), chosen per row by `dataset` |
| **Episode state** | what persists across turns | tool-server sessions (none for search; the sandbox for code) |

The guide sorts frameworks by deployment: an **HTTP service** (OpenEnv, ORS, NeMo Gym) or **in-process**
(Verifiers, SkyRL Gym, GEM). Its rule is HTTP for sandboxed or stateful execution, in-process for pure-Python
scoring. We already do both: verifiers run in-process in the vLLM actors, and the code sandbox and Wikipedia
search run as per-node HTTP services.

## 3. Current inventory

| Environment (`dataset`) | Type | Reward | Runs | Used in | Known issues |
|---|---|---|---|---|---|
| `multilingual_math`, `math` | typed answer | 1 / 0 exact | in-process | W1a–E8 | rows run out (E8 reuses 78 W1a rows) |
| `math_direct_shaped` | default-mode answer | shaped | in-process | A3, W2 | — |
| `ifeval_direct`, `ifeval_think` | verifiable instructions | fraction of constraints met | in-process (NLTK) | W2–E8 | 389 scoring errors until NLTK data was offline (W3) |
| `code_stdio` | program | 1 only if all hidden tests pass | HTTP `code_api`, contained | A5, W2–E8 | — |
| `tool_call` | single-turn call | name / JSON / argument match | in-process | A4b, W2–E8 | single turn only; no execution or recovery |
| `eu_doc` | typed answer over a document | exact / numeric | in-process | W3, E8 | — |
| `eu_qa` | calibrated factual QA | +1 / 0 abstain / −1 | in-process | W3, E8 | **W3 collapse: abstaining was optimal** |
| `eu_qa` + `wiki_search` | multi-turn search | +1 / 0 / −1 on the final turn | HTTP search per node | E8 | default mode only (parser postfix) |
| svenska-rl (8 office envs) | typed answer and tool episodes | end-state checks | in-process (`svrl`) | not yet in a run | Swedish only |

## 4. Quality bar (environment contract for oellm-rlvr)

Every environment passes these checks **before it enters a training pool**. They extend the svenska-rl
qualification (v0.2) with what Stage 5 taught us. Q1–Q6 run on CPU in CI; Q7 and Q8 need one cheap baseline job.

| # | Check | Pass when | Incident it would have prevented |
|---|---|---|---|
| Q1 | **Reference solves it** | the gold answer, also after reasoning text and inside both modes' formats, earns the maximum reward on every task | — |
| Q2 | **Shortcut probes** (below) | every probe earns a mean reward ≤ 0.3 of the reward range **and below the parent policy's mean** | **W3**: always `UNKNOWN` scored 0, W1a's mean was −0.42 |
| Q3 | **Verifier audit** | 50 graded rollouts per environment and language checked by hand (or a pinned judge, spot-checked); false positives and negatives ≤ 2 % | answer-line leakage from retrieved text (fixed before E8) |
| Q4 | **Coverage** | every row has a registered verifier; verifier error rate ≤ 2 % on a dry run, with no network or download at score time | the W1a/W2 reward-coverage bug (§10.5); NLTK on offline nodes |
| Q5 | **Determinism and splits** | seeded generation; ≥ 90 % unique tasks; train / dev / eval disjoint; no eval id in any pool | release checklist 7.1 |
| Q6 | **Train/serve parity** | prompt, tool schemas and observation text are token-identical to the release chat template; tool budget equal to the eval's | thinking-mode search postfix (kept out of E8) |
| Q7 | **Difficulty** | pass-rate profile on the parent (k = 8): enough prompts with in-group variance; the window is recorded | saturated pools in RL-0b |
| Q8 | **Learnability (L1–L3)** | L1 baseline recorded; L3 (held-out +5 points, lower 95 % bound > 0) before it counts as a proven environment | — |
| Q9 | **Coverage of languages and modes** | stated per language and per mode; no language silently missing | — |
| Q10 | **Observability** | per-environment reward histogram, tool-call stats and sample rollouts logged every update | finding the E8 canary signal took manual decoding |

**Shortcut probes (Q2)**, applied wherever they make sense:
- empty answer, or no answer line;
- always abstain (`UNKNOWN`, `INFORMATION SAKNAS`);
- hedging, i.e. listing several candidates;
- copying a number or name from the prompt, or listing them all;
- an answer in the wrong mode or format (think tags in default mode);
- the maximum-length ramble;
- tool episodes:
  - finishing immediately;
  - only reading until the step limit;
  - calling every tool;
  - repeating the same call;
- search: answering with the first result's title.

**Ternary rewards need the policy comparison in Q2.** Under +1 / 0 / −1, "always abstain" earns 0. That is
harmless only if the policy's expected reward on the pool is above 0. So the pool must contain enough questions
the parent knows: the "known guard" in E8, 448 of 1,088 QA rows. Check it from the parent's pass-rate profile
before training.

## 5. Architecture stance

- **Keep the trainer; standardise the environment interface.** The guide's conclusion is that every framework can
  express the same environment; what differs is how it plugs into the trainer. We stay on TMAX for RL and verl for
  distillation, and write new tool environments in one shape both can use:
  - a stateless HTTP service with a session id, JSON-schema tools and an end-state `check`;
  - aligned with OpenEnv/MCP or ORS, so environments also run under SkyRL (T2) and in eval harnesses.
- **Run the HTTP services on the training nodes, on 127.0.0.1**, as we do for `code_api` and `wiki_search`. The
  guide's benchmarks show the server is rarely the bottleneck; execution (sandbox start, search) is.
- **External hubs are candidates, not sources.**
  - The hubs: HF Spaces (4k+ MCP environments), PrimeIntellect's Environments Hub, openreward.ai (330+).
  - Most of these environments are English-only, of unknown quality, with mixed licences.
  - An external environment enters a pool only after the Q1–Q10 checks, a licence check, and translation or
    localisation where that makes sense.
- **Rewards stay verifiable where possible.** LLM-judge rewards (writing quality, safety) get a pinned judge
  digest, rubric decomposition, shortcut probes against the judge, and a held-out stricter judge. Until then those
  behaviours stay in DPO (Stage 4b).

## 6. Roadmap (ordered by our measured gaps)

| # | Environment | Gap it targets | Work | Cost |
|---|---|---|---|---|
| E-1 | **Retro-qualify existing environments** (Q1–Q4 probes for eu_qa, eu_doc, tool_call, ifeval, math_direct) | prevent the next W3 | CPU, tests in oellm-rlvr | ~0 local-h |
| E-2 | **Multi-turn tools with injected errors**: mock services (calendar, ledger, mail, search) with timeouts, malformed results and business errors; end-state check; Kontoret as the world | EUROPA tool recovery 0.49 (target 0.8); held-out multi-step 0.76 | tool-episode environment in the §5 shape, profile, canary | ~50 local-h to qualify, then an expert run |
| E-3 | **svenska-rl → european-rl** ([design](OELLM_EUROPEAN_RL.md)): the 8 Kontoret environments in oellm-rlvr pools, then localised to more languages (VAT, holidays, formats per country) | eu_doc invoice/meeting variants at 0.07–0.36 | adapter from the `svrl` export; localisation | CPU + profile |
| E-4 | **Search in thinking mode** | E8 trains default mode only | mode-aware observation postfix in the parser patch, token test | small |
| E-5 | **Long-context documents** (16K–256K): eu_doc generators over long, noisy files, with the needle at controlled depth | 256K base; post-training at ≤ 10K | generator + NIAH-style probes | profile + expert run |
| E-6 | **Multilingual verifiable instructions** (IFEval-style constraints per language) | strict formats; answer language | translate constraints, language-aware checkers | CPU + profile |
| E-7 | Judge-based environments (writing quality, refusal calibration) | EUROPA practical communication; over-refusal | §5 judge safeguards first | later |

Each new environment enters training first as a narrow **expert run** (one environment family, its own recipe)
and is integrated through distillation (run program §10.7). It joins the mixed replay only once it has passed L3.
