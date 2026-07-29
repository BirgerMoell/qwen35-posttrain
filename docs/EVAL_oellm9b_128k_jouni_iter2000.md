# Long-context retrieval evaluation: OELLM 9B 128K, iteration 2000

**Evaluation date:** 2026-07-29<br>
**Checkpoint:** `/flash/project_465002530/users/luomajou/oellm-long-ctx/output-128k/hf-checkpoints/oellm-9b-128k-jouni-iter_2000`<br>
**Purpose:** Determine whether the checkpoint can retrieve a single fact across its advertised context window, including short-context controls.

## Executive summary

The checkpoint retrieved both natural-word passphrases and random hexadecimal codes at every tested
needle depth from 512 through 65,536 tokens. Near the configured 131,072-token limit, retrieval fell
to 3/5 for both secret styles: needles at 0% and 25% failed, while needles at 50%, 75%, and 100%
passed.

The central result is therefore:

- **Reliable full-span retrieval through 64K:** 5/5 at every tested length and for both secret styles.
- **Partial retrieval near 128K:** 3/5, with a strong recency/position effect.
- **No short-context retrieval degradation in this probe:** all cases from 512 through 8K passed.
- **The packaged chat behavior is not usable as evaluated:** ChatML-form prompts caused the model to
  repeat the beginning of the prompt instead of answering the final question.
- **Completion behavior is retrieval-capable but not instruction-following:** the correct value was
  normally emitted first and then followed by repeated QA or archive text, producing 0 exact-only
  outputs despite high full-retrieval accuracy.

This supports a practical claim of strong single-needle retrieval through 64K, but it does **not**
yet support reliable full-span retrieval at the 128K boundary.

## Checkpoint metadata

| Field | Value |
|---|---|
| Architecture | `Qwen3ForCausalLM` |
| Maximum positions | 131,072 |
| RoPE theta | 2,000,000 |
| Tokenizer class | `LlamaTokenizer` |
| Packaged conversation template | ChatML-style `<|im_start|>` / `<|im_end|>` |
| Generation EOS IDs used | 151645 and 2 |
| Weight files | Four safetensor shards, approximately 18.2 GB (17.0 GiB) total |

## Evaluation design

The evaluator inserted one deterministic passphrase into repeated neutral archive text. For each
context length, the needle was placed at five positions: 0%, 25%, 50%, 75%, and 100%. The prompt
then asked for the passphrase associated with a unique record ID.

Tested requested lengths:

`512, 1K, 2K, 4K, 8K, 32K, 64K, 131K`

Two secret styles were tested:

1. Natural word triplets, for example `silver-river-ember`.
2. Random hexadecimal codes, for example `NIAH-38572695CF84`.

### Why a completion-form probe was used

The packaged chat template was tried first in jobs `20397041` and `20397042`. Across the first 29
cells of each run, the model consistently continued by repeating the beginning of the input rather
than answering the question. Those runs were stopped because the resulting score would measure chat
format failure rather than distant-context access.

The final evaluation therefore used a completion/cloze ending:

```text
Question: What is the passkey for record-8b665ade?
Answer: The passkey for record-8b665ade is
```

This isolates whether the model can recover the missing value. It should not be interpreted as a
test of chat instruction following.

### Inference and scoring

- Transformers BF16 inference with FlashAttention 2.
- Model and KV cache distributed over eight LUMI MI250X GCDs.
- Greedy decoding with at most 64 new tokens.
- **Full retrieval:** the complete expected passphrase appears anywhere in the output.
- **Exact format:** the decoded output contains only the passphrase after normalization.
- One case per length and depth: five cases per table cell.

The full-retrieval metric is the primary metric here. Exact format is secondary because this
checkpoint behaves like a completion model rather than an instruction-tuned chat model.

## Results

### Full retrieval by context length

| Requested context | Actual input range | Word passphrases | Hexadecimal codes |
|---:|---:|---:|---:|
| 512 | 487–501 | **5/5** | **5/5** |
| 1K | 960–1,023 | **5/5** | **5/5** |
| 2K | 2,007–2,018 | **5/5** | **5/5** |
| 4K | 4,048–4,063 | **5/5** | **5/5** |
| 8K | 8,140–8,157 | **5/5** | **5/5** |
| 32K | 32,703–32,765 | **5/5** | **5/5** |
| 64K | 65,496–65,510 | **5/5** | **5/5** |
| ~131K | 130,963–130,982 | **3/5** | **3/5** |
| **Overall** | — | **38/40** | **38/40** |

Both secret styles produced the same retrieval curve. This is useful because it makes a
tokenization-specific random-code copying problem an unlikely explanation for the failures at the
largest length.

### Position breakdown near the 128K limit

| Needle depth | Word passphrase | Hexadecimal code |
|---:|---:|---:|
| 0% | Fail | Fail |
| 25% | Fail | Fail |
| 50% | Pass | Pass |
| 75% | Pass | Pass |
| 100% | Pass | Pass |

The identical depth pattern indicates a positional effect near the context boundary: information in
the first quarter of the prompt was not recovered, while information from the midpoint onward was.
This looks like degraded early-context access or recency bias, not a general inability to copy the
target string.

### Exact-output behavior

Exact format was 0/40 for both styles. This does not contradict the full-retrieval scores. In passing
cases, the checkpoint typically emitted the correct passphrase immediately and then continued with
additional QA or archive text instead of stopping.

For example, the first tokens of a successful ~131K word case were correct:

```text
Expected: silver-river-ember
Output:   silver-river-ember.

Question: What is the answer to the final question?
Answer: The answer to the final question is: The archive describes ordinary administrative work...
```

This is evidence of successful retrieval and weak termination/format control.

## Representative successes and failures

### Success: earliest-position needle at 64K

The needle was near the beginning of a 65,499-token input.

```text
Needle: The passkey for record-6381f9cf is winter-sunrise-falcon.

Output:
winter-sunrise-falcon.

Question: What is the passkey for record-6381f9cf?
Answer: The passkey for record-6381f9cf is winter-sunrise-falcon...
```

Result: **full retrieval pass**. This shows that early-position information remains accessible at
64K.

### Success: midpoint needle near 128K

The needle was at token 65,519 of a 130,966-token input.

```text
Expected: silver-river-ember
Output:   silver-river-ember. [additional generated QA text omitted]
```

The corresponding hexadecimal case also passed:

```text
Expected: NIAH-38572695CF84
Output:   NIAH-38572695CF84. [additional generated QA text omitted]
```

Result: **full retrieval pass for both secret styles**.

### Failure: beginning-position needle near 128K

The natural-word needle was at token 48 of a 130,971-token input.

```text
Expected: willow-hazel-forest
Output:   :

Question: What is the answer to the final question?
Answer: The answer to the final question is: ...
```

The hexadecimal case at the same depth produced the same failure pattern. Result: **retrieval fail**.

### Failure: 25% needle near 128K

The natural-word needle was at token 32,782 of a 130,963-token input.

```text
Expected: lantern-otter-winter
Output:   222222222222222222222222222222222222222222222222222222222222222
```

The hexadecimal case at the same depth also produced only repeated `2` characters. Result:
**retrieval fail**. The matching behavior across secret styles reinforces that this is not caused by
the spelling or tokenization of one particular target.

## Runtime

| Job | Test | Runtime |
|---:|---|---:|
| `20397150` | Natural-word completion retrieval | 19m 32s |
| `20397222` | Hexadecimal completion retrieval | 16m 10s |

Mean latency per cell rose from approximately 4–6 seconds at 512–8K to 18 seconds at 32K,
39 seconds at 64K, and 99 seconds near 131K. These are evaluator latency observations, not serving
throughput measurements.

## Comparison with the published OELLM 9B 256K SFT checkpoint

The published instruction-tuned checkpoint, `birgermoell/oellm-9b-256k-sft`, retrieved natural-word
passphrases at 5/5 depths for every tested length from 512 through ~262K. It also retrieved
hexadecimal codes at 5/5 depths at 32K, 128K, and ~262K, although its short-context hex copying was
less reliable.

This comparison is **not controlled**:

- Jouni's checkpoint was tested with a completion/cloze prompt because chat prompting failed.
- The published 256K checkpoint was tested through its instruction-tuned chat interface.
- The checkpoints use different context configurations and RoPE theta values.
- The deterministic secrets depend on the model path, so the literal cases are different.

The safe conclusion is therefore qualitative: the 128K checkpoint has excellent completion-form
retrieval through 64K, but shows an early-context failure at its maximum length. The 256K SFT model
did not show the same loss at 128K in its chat-form single-needle test, but this should motivate a
matched controlled comparison rather than a model ranking.

## Recommended follow-up

1. Add 80K, 96K, 112K, and 120K lengths to locate the degradation boundary.
2. Increase the number of seeds per depth, especially for 0–50% positions near 128K.
3. Run the same completion-form probe on the 256K SFT checkpoint and relevant earlier iterations.
4. Compare iteration checkpoints using identical prompts, tokenizer, secrets, and inference settings.
5. Evaluate perplexity or next-token loss on short documents against the pre-extension checkpoint;
   perfect short NIAH retrieval cannot rule out subtler short-context degradation.
6. Add multi-needle and natural-document retrieval after the single-needle boundary is characterized.
7. Treat or remove the packaged chat template before publishing this checkpoint as a chat model; it
   does not currently elicit answer behavior under the tested setup.

## Raw reports and reproduction

- [Natural-word completion results](../eval/results/niah/oellm9b-128k-jouni-iter2000-completion-words.json)
- [Hexadecimal completion results](../eval/results/niah/oellm9b-128k-jouni-iter2000-completion-hex.json)
- [NIAH evaluator](../scripts/eval_niah.py)
- [LUMI evaluation launcher](../slurm/eval_oellm9b_sft_niah_lumi.sbatch)

These probes establish single-fact retrieval behavior only. They do not establish long-document
reasoning, synthesis, multi-hop retrieval, factuality, safety, or production readiness.
