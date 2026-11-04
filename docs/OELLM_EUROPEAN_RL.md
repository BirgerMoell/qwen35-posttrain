# european-rl: design for a European RL environment suite

Status 2026-11-04: repository created, M0 in progress ([github.com/BirgerMoell/european-rl](https://github.com/BirgerMoell/european-rl)). Decisions (11-04): a new public repo; all 36 OpenEuroLLM languages; LLM review first, marked as such. This document plans **european-rl**: verifiable RL environments for office,
document and agentic work in the European languages, grown out of
[svenska-rl](https://github.com/BirgerMoell/svenska-rl). It sits under the quality bar in
[RL environments](OELLM_RL_ENVIRONMENTS.md) (roadmap item E-3) and feeds the expert runs of the RL program
(§10.7).

Contents:
1. The gap
2. What labs train on
3. svenska-rl today: an honest audit
4. Design principles
5. Locale packs: what "European" means in data
6. Environment families
7. From one language to 36
8. Architecture
9. Milestones
10. Decisions needed

## 1. The gap

A survey of 2025–2026 post-training reports (sources in §11) finds:

- **Stateful agent and office environments are English-only.** Every one found is English: τ²-bench,
  AppWorld, NVIDIA's Workplace Assistant and Banking, TheAgentCompany, OfficeBench, Spreadsheet-RL and BFCL.
- **Nemotron 3 uses no multilingual RL environments.** Multilingual ability comes only from translated SFT data.
- **The Multilingual Reasoning Gym** (94 tasks, 14 languages, six of them European) states that there is no
  large-scale, intrinsically multilingual RLVR environment.
- **No environment encodes European locale semantics:**
  - VAT regimes and intra-EU reverse charge;
  - national ID and VAT-number formats;
  - SEPA/IBAN payments;
  - national holidays;
  - GDPR-type requests;
  - formality registers (du / Sie / vous / Lei).
- **Verifiers have language bias**, and cross-lingual chain-of-thought collapse is documented. Verifiers therefore
  need per-language audits.

**A European suite fills a real gap**, and it plays to what a 9B multilingual assistant should be good at.

## 2. What labs train on

| Family | Who reports it | Verification | Relevance for us |
|---|---|---|---|
| Math with answer checks | everyone | answer match | have it (baseline) |
| Code with unit tests | everyone | tests in a sandbox | have it |
| Instruction following / format | Qwen3, Kimi K2, GLM-4.5, Nemotron, OLMo 3, Tülu 3 | constraint checkers (+ judge for soft constraints) | have English-centric; **multilingual missing** |
| SWE repository repair | Qwen3-Coder, DeepSeek-V3.2, GLM, MiniMax, K2, INTELLECT-3 | fail-to-pass tests in Docker | costly at 9B; later |
| Search / deep research | DeepSeek (50K tasks, largest share), GLM, Kimi, MiniMax, INTELLECT-3 | unique verifiable answer | **E8 (running)** |
| **Tool use in synthetic stateful worlds** (database + tools + simulated user) | DeepSeek "general agent" (1,827 synthesised environments), Kimi K2 (20K+ synthetic tools, LLM world model), GLM, Nemotron Workplace/Banking, τ², AppWorld | end state / database diff, collateral-damage checks | **the core of european-rl** |
| Chat / helpfulness | Kimi, DeepSeek, Nemotron, OLMo 3, MiniMax | rubric GenRM | DPO for now |
| Logic puzzles | K2, MiniMax M1, INTELLECT-3, RLVE (400 procedural environments) | procedural | cheap filler |
| Safety / over-refusal | K2, Nemotron | judge / classifier | DPO (4b) for now |
| Long context, structured output, faithfulness | Nemotron, K2 | judge / schema | E-5 |
| Spreadsheets, terminal, GUI | emerging | recalculation / state | spreadsheets yes, GUI later |

**Patterns that recur in the strongest reports:**
- **Hard to solve, easy to verify** (Qwen3-Coder). DeepSeek keeps a synthesised environment only if a reference
  solution that may call nothing but the tools solves it (pass@100 > 0).
- **End-state verification with collateral-damage checks** (AppWorld, Nemotron). The world is a database, and the
  reward checks what changed and what must not have changed.
- **An LLM simulates tools or users, but the reward stays rule-based** (Kimi's world model, τ² dual control,
  GLM's user simulator). Judges filter data; they do not grade where a rule can.
- **Difficulty control:** drop always-pass and always-fail prompts; easy/normal/hard pools; a pass-rate target
  that moves from easy to hard (Nemotron); procedural difficulty knobs (RLVE); iterative "make it harder"
  synthesis (DeepSeek).
- **Robustness by perturbation:** vary tool schemas, system prompts and scaffolds (MiniMax M2/Forge).
- **Language hygiene as reward:** language-consistency reward (DeepSeek-V3.2, Magistral +0.1) and language-mixing
  penalties (MiniMax Forge, GLM "pathology RL").
- **Many environments trained jointly beat single-environment runs** (Nemotron: single-environment RL caused
  unrecoverable regressions elsewhere; RLVE: +3.37 vs +0.49). This is consistent with our plan:
  1. narrow expert runs per family;
  2. integration by distillation;
  3. joint replay.

## 3. svenska-rl today: an honest audit

**Strengths:**
- A solid contract v0.2: typed answers or tool episodes, seeded splits, ground truth from simulator state, no
  multiple choice.
- **Shortcut probes:** every probe scores 0 on all 8 environments (`svrl.cli check`, 11-04).
- A shared synthetic world (Kontoret) with real Swedish conventions: OCR/Luhn, bankgiro, VAT 25/12/6 %, holidays.

**Weaknesses that matter for training value:**
1. **Rule puzzles, not work.** Every business rule is spelled out in the prompt ("discount before VAT, round half
   up per line…"). The model practises arithmetic under explicit rules. Real work means knowing or looking up the
   convention.
2. **One template per environment.** The phrasing barely varies, so a model can learn the template rather than
   the skill.
3. **No difficulty evidence.** No L1 baseline exists (the only run summary is empty), so we do not know whether
   any environment sits in the learnable 10–90 % band for our model.
4. **Small and short.**
   - 8 environments.
   - The tool episodes have a few tools and short horizons.
   - No messy inputs (ambiguous e-mails, contradictory documents).
   - No user to ask, no cross-system workflows.
5. **Generator sloppiness.** One meeting task lists the same participant twice and repeats a busy slot. It is
   harmless to the answer, but it is a quality smell.
6. **Swedish hard-coded** in `kontoret` (VAT, holidays, IDs, formats). Translation alone cannot make it
   European.

**Verdict:** keep the contract, the probes and the world-simulator idea; redesign the tasks.

## 4. Design principles

1. **Localise the world, don't translate the text.** One European company group with subsidiaries in each
   country. Each country's tasks use its own laws of arithmetic: VAT rates, holidays, ID checksums, payment
   references, formats. Names, companies and addresses come from per-locale lists. They are never translated
   Swedish names.
2. **Rules live in documents.** VAT handling, rounding, approval limits and leave rules sit in a company handbook
   and policy documents, in the local language, reachable through a `search_policies` / `read_document` tool.
   The prompt states the task, not the rules. Tasks state a rule inline only at the easiest difficulty level.
3. **Hard to solve, easy to verify.** Every task has:
   - a programmatic checker on the end state;
   - a reference solver that uses only the public tools (the DeepSeek rule);
   - collateral-damage checks: no wrong e-mails, no wrong postings, no edits outside scope.
4. **Messy, realistic inputs.**
   - Requests arrive as e-mails or tickets with typos, mixed languages, missing fields and contradictions.
   - Some tasks are unanswerable or need a clarifying question; the right action is then to ask or to abstain,
     and the checker rewards that.
5. **Simulated colleagues and customers, rule-based reward.** A user simulator (an LLM with a hidden goal, in the
   user's language) drives multi-turn tasks, τ²-style. The reward is always the database end state, never the
   simulator's opinion. A rule-based simulator covers the cheap tiers.
6. **Perturbation for generalisation.** Per task, vary:
   - tool names and schemas (English or local-language identifiers, renamed fields);
   - system prompts;
   - document layouts (table / prose / key-value);
   - number and date formats.
7. **Language as part of the reward.** The answer must be in the user's language and register (du / Sie / vous /
   Lei per the policy). A language-consistency term (fastText/lingua) and a mixing penalty apply to every
   environment.
8. **Difficulty knobs, measured.** Each environment exposes levels:
   - rules inline → rules in documents;
   - 1 system → several;
   - clean → noisy;
   - answerable → ambiguous.
   An L1 baseline on our checkpoints (W2, E8) places each level. Pools keep levels with in-group variance and move
   from easy to hard as the policy improves (Nemotron's moving pass-rate target).
9. **The quality contract is the gate.** Contract v0.3 = svenska-rl v0.2 + the Q1–Q10 checks in
   [RL environments](OELLM_RL_ENVIRONMENTS.md). New probes:
   - "act in the wrong locale" (Swedish VAT on a German invoice);
   - "copy the policy text";
   - "e-mail everyone";
   - "ask a question every turn".

## 5. Locale packs: what "European" means in data

One pack per country (not per language: Belgium and Switzerland have several languages, Austria and Germany
share one). Each pack is data plus small functions, tested independently:

| Field | Examples |
|---|---|
| Languages and registers | `de-AT`, formal Sie in business mail; `fr-BE` / `nl-BE` |
| Money | currency (EUR, SEK, PLN, CZK, HUF, DKK, RON, BGN, …), decimal and grouping, rounding customs (e.g. Swedish öre / Finnish 5-cent cash rounding) |
| VAT | standard and reduced rates by category; reverse charge for intra-EU B2B; VAT-number format and checksum |
| Company and person IDs | organisation numbers, VAT IDs, synthetic personal IDs that cannot be real (svenska-rl's day + 60 trick generalised) |
| Payments | IBAN (ISO 13616 checksum), SEPA creditor reference RF (ISO 11649), national references (Swedish OCR, Finnish viitenumero, Belgian structured communication, Norwegian KID) |
| Calendar | public holidays incl. movable feasts and regional ones; working hours and lunch norms; time zone (WET / CET / EET) |
| Addresses and names | formats, postal codes, given and family names by country, company suffixes (AB, GmbH, SARL, S.p.A., sp. z o.o., Oy, BV) |
| Documents | invoice, credit note, payslip, leave request, meeting minutes, contract templates in the local style |

Tier 1 packs match our eu_qa languages: SE, DE, FR, ES, IT, NL, PL, PT, FI, plus AT, BE, IE and a pan-EU English
group office. Tier 2 covers the rest of the 36 languages in the OpenEuroLLM scope.

## 6. Environment families

Ordered by value for a 9B office assistant. Families 1–4 make up v1.

| # | Family | Interaction | Verification | Lab precedent |
|---|---|---|---|---|
| 1 | **Finance desk**: amount due with national VAT, credit notes and partial payments; intra-EU reverse charge; multi-currency; payment matching with IBAN/RF/national references | typed answer → tool episode (ledger) | exact cents; ledger end state | DeepSeek general agent, Nemotron Banking |
| 2 | **Office world**: mail, calendar, contacts, files, CRM/tickets, ledger. Tasks arrive as e-mails from colleagues ("move the board meeting, Anna is sick, tell everyone"); cross-country scheduling with holidays and time zones; injected tool errors | multi-turn tools (HTTP service, §8) | end state + collateral damage; recovery from injected errors | AppWorld, Nemotron Workplace, τ² |
| 3 | **Customer and citizen service** with a simulated user: returns, subscription changes, address change, GDPR access or erasure request, handled by a policy document; ask when information is missing | multi-turn with user simulator | database end state; policy adherence; language and register | τ²-bench, Kimi K2, GLM |
| 4 | **Multilingual instruction following**: constraints that exist only in some languages (formality, diacritics, locale number/date formats, word counts in compounding languages, required closing formula) plus IFEval-style constraints | single turn | rule checkers per language | Qwen3, GLM (151 constraint types), Nemotron |
| 5 | **Spreadsheets and tables**: compute, pivot, reconcile, find the error, from local-format CSV/xlsx | tools (sheet API) | recalculated cell values / table diff | Spreadsheet-RL, OfficeBench |
| 6 | **Documents and forms**: fill a form from a dossier; redact personal data by policy (national ID formats); extract structured JSON from a contract or payslip | typed / tools | field match; redaction F1 | Nemotron (structured output), svenska-rl |
| 7 | **Cross-lingual work**: a request in Polish, an attachment in German, a reply required in Polish; minutes summarised into a French action list | typed / tools | structured checks + language term | — (gap) |
| 8 | **Long dossiers** (16K–256K): many documents, the answer depends on two of them | typed | exact | Nemotron long-context QA |

The existing 8 svenska-rl environments become level-1 instances of families 1, 2 and 6 in the SE pack. Their
generators are generalised, not discarded.

## 7. From one language to 36

1. **Author templates once, in English, with slots.** Write several phrasings per task and register, plus the
   policy documents.
2. **Translate on Leonardo** with gemma4:31b (already served for our judges), into every target language:
   - back-translation check;
   - the OELLM quality gate (`grammatical`, `natural`, `language_match`) to drop bad renderings;
   - tier-1 languages: a native spot review of ≥ 50 renderings per environment family.
3. **Never translate data, only text.** Amounts, IDs, dates, names and companies come from the locale pack, so
   the answer stays exact in every language.
4. **Audit verifiers per language.** Q3 (50 hand-checked rollouts) runs per language for each family before that
   language enters a pool. Answer parsers accept local formats (1.234,50 €; 1 234,50 zł).
5. **Cross-locale combinations** (German company, Polish customer) come for free from the world model and test
   real EU work.

## 8. Architecture

- **One core package, `eurl`.** It grows out of `svrl`:
  - the world simulator (database of entities, documents, systems);
  - locale packs;
  - environments;
  - contract v0.3 and its probes;
  - exporters.
- **svenska-rl stays the Swedish community front.** It keeps its Swedish contribution flow on top of the SE pack.
- **Typed-answer environments export rows for oellm-rlvr pools** (`messages`, `ground_truth`, verifier name);
  scoring runs in-process, as today.
- **Tool and user environments run as a stateless HTTP service with session ids**, in the shape set in
  [RL environments §5](OELLM_RL_ENVIRONMENTS.md):
  - `reset(task_id)` → tools + first message;
  - `step(session, call)` → observation;
  - `check(session)` → reward.
  - One service per training node on 127.0.0.1, like `wiki_search`.
  - The same service serves TMAX RL, verl distillation, SkyRL (T2) and our evals.
- **The user simulator runs as a second local service.** A small vLLM model, or rules for the cheap tiers.

## 9. Milestones

| # | Milestone | Done when | Cost |
|---|---|---|---|
| M0 | Core and locale packs SE, DE, FR, PL, FI | packs pass their own tests (checksums, holidays vs official calendars); svenska-rl environments run on `eurl` unchanged | CPU, agent work |
| M1 | Family 1 (finance desk) and family 4 (multilingual IF) in tier-1 languages, with difficulty levels | contract v0.3 passes; L1 baselines on W2 and E8 per level and language | CPU + ~20 local-h baselines |
| M2 | Family 2 office world as an HTTP service, with injected errors | contract passes; canary inside RL (as the E8 canary) | ~30 local-h |
| M3 | First expert run on families 1, 2 and 4 from the current student | L3: held-out +5 points with a positive lower bound; the release checklist does not regress | ~300 local-h, **separate approval** |
| M4 | Family 3 (simulated users) and tier-2 locales | as M1–M2 | later |

Data and environments are public (Apache-2.0 code, synthetic data, no personal data). Models trained on them stay
private until release review.

## 10. Decisions (11-04)

Decided: a new public repository with the `eurl` core; coverage target all 36 languages; LLM review first, recorded per language as `llm`, native review later.

### Originally open

1. **Repository:** a new public GitHub repo `european-rl` with the `eurl` core, with svenska-rl depending on it.
   Or rename and extend svenska-rl.
2. **Tier-1 language list:** the eu_qa set above, or another order based on OpenEuroLLM priorities.
3. **Native review capacity for tier 1:** project partners per language, or LLM review only, marked as such.

## 11. Sources

- DeepSeek-V3.2 <https://arxiv.org/abs/2512.02556>
- Kimi K2 <https://arxiv.org/abs/2507.20534>
- Kimi K2.5 <https://arxiv.org/abs/2602.02276>
- Qwen3 <https://arxiv.org/abs/2505.09388>
- Qwen3-Coder <https://qwenlm.github.io/blog/qwen3-coder/>
- Qwen3.5 <https://www.alibabacloud.com/blog/qwen3-5-towards-native-multimodal-agents_602894>
- MiniMax-M1 <https://arxiv.org/abs/2506.13585>
- MiniMax M2 <https://www.minimax.io/news/aligning-to-what-rethinking-agent-generalization-in-m2>
- MiniMax Forge <https://minimax.io/news/forge-scalable-agent-rl-framework-and-algorithm>
- GLM-4.5 <https://arxiv.org/abs/2508.06471>
- Nemotron 3 Nano <https://arxiv.org/abs/2512.20848>
- Nemotron 3 Super RLVR <https://docs.nvidia.com/nemotron/latest/nemotron/super3/rl/rlvr.html>
- NeMo Gym <https://github.com/NVIDIA-NeMo/Gym>
- OLMo 3 <https://arxiv.org/abs/2512.13961>
- INTELLECT-3 <https://arxiv.org/abs/2512.16144>
- Apertus <https://arxiv.org/abs/2509.14233>
- Magistral <https://arxiv.org/abs/2506.10910>
- Multilingual Reasoning Gym <https://arxiv.org/abs/2603.10793>
- RLVE <https://arxiv.org/abs/2511.07317>
- τ²-bench <https://arxiv.org/abs/2506.07982>
- AppWorld <https://arxiv.org/abs/2407.18901>
- TheAgentCompany <https://arxiv.org/abs/2412.14161>
- Spreadsheet-RL <https://arxiv.org/abs/2605.22642>
- Multilingual verifier bias <https://arxiv.org/abs/2608.20362>
- Cross-lingual chain-of-thought collapse <https://arxiv.org/abs/2506.05850>
- HF RL environments guide <https://huggingface.co/spaces/AdithyaSK/rl-environments-guide>
