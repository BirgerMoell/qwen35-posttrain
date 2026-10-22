#!/usr/bin/env python3
"""Re-fit the lm_head rows of the chat-tag tokens (IDs 11-18) with the rest of the model frozen.

Why: after Stage 3 the output rows of `<tool_call>` (11) and `</tool_call>` (12) were still nearly
identical (they start out near-equal and AdamW moves a weight by at most ~LR per step; 397 steps at
3e-6 cannot separate them), so the model emitted `<tool_call>` where `</tool_call>` belongs even though
all 160,990 supervised calls in the data were well formed. The hidden states carry the open/close
distinction; only the head rows cannot express it.

Method: collect final hidden states h at supervised positions of Stage-3 training rows (every tag
target plus a random sample of ordinary targets). With all other lm_head rows frozen, the softmax
denominator splits into a constant part (precomputed once per position) and the 8 tag logits, so the
exact full-vocabulary cross-entropy is optimized over only the 8 rows, with an L2 pull toward the
original rows. Evaluation uses the held-out Stage-3 validation rows: top-1 rate per tag target and the
rate at which a tag wrongly becomes top-1 at an ordinary position (before vs after). The repaired
model is written as a new directory; the input checkpoint is not modified.
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import shutil
from pathlib import Path

import pyarrow.parquet as pq
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM

TAGS = list(range(11, 19))
NAMES = {11: "<tool_call>", 12: "</tool_call>", 13: "<tools>", 14: "</tools>", 15: "<tool_response>",
         16: "</tool_response>", 17: "</think>", 18: "<think>"}


def rows_from(files, need, max_len, n, rng):
    out = []
    for f in files:
        t = pq.read_table(f, columns=["input_ids", "assistant_masks"]).to_pylist()
        rng.shuffle(t)
        out += [r for r in t if len(r["input_ids"]) <= max_len and any(x in r["input_ids"] for x in need)]
        if len(out) >= n:
            break
    return out[:n]


@torch.no_grad()
def collect(model, rows, other_per_row, rng):
    """Return hidden states, targets, and per-position lse/max over the non-tag vocabulary."""
    W = model.lm_head.weight
    rest = torch.ones(W.shape[0], dtype=torch.bool, device=W.device)
    rest[TAGS] = False
    H, T, LSE, MX, TGT_LOGIT = [], [], [], [], []
    for k, r in enumerate(rows):
        ids, mask = r["input_ids"], r["assistant_masks"]
        sup = [p for p in range(1, len(ids)) if mask[p]]
        tag_pos = [p for p in sup if ids[p] in TAGS]
        other = [p for p in sup if ids[p] not in TAGS]
        pos = tag_pos + rng.sample(other, min(other_per_row, len(other)))
        if not pos:
            continue
        h = model.model(torch.tensor([ids], device=W.device)).last_hidden_state[0]
        hp = h[[p - 1 for p in pos]]
        logits = (hp @ W.T).float()
        lr = logits[:, rest]
        tg = torch.tensor([ids[p] for p in pos], device=W.device)
        H.append(hp.float().cpu()); T.append(tg.cpu())
        LSE.append(torch.logsumexp(lr, -1).cpu()); MX.append(lr.max(-1).values.cpu())
        TGT_LOGIT.append(logits.gather(1, tg[:, None])[:, 0].cpu())
        if k % 200 == 0:
            print(f"  collected {k}/{len(rows)} rows", flush=True)
    return [torch.cat(x) for x in (H, T, LSE, MX, TGT_LOGIT)]


def metrics(Ws, H, T, MX):
    tag_logits = H @ Ws.T                      # N x 8
    best_tag_val, best_tag = tag_logits.max(-1)
    pred = torch.where(best_tag_val > MX, torch.tensor(TAGS)[best_tag], torch.full_like(T, -1))
    out = {}
    for t in TAGS:
        m = T == t
        if m.any():
            out[NAMES[t]] = {"n": int(m.sum()), "top1": float((pred[m] == t).float().mean())}
    ordinary = ~torch.isin(T, torch.tensor(TAGS))
    out["ordinary_positions"] = {"n": int(ordinary.sum()), "tag_false_top1": float((pred[ordinary] >= 0).float().mean())}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--mix", type=Path, required=True, help="pre-tokenized Stage-3 mix directory")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--train-rows", type=int, default=2500)
    ap.add_argument("--think-rows", type=int, default=500)
    ap.add_argument("--other-per-row", type=int, default=64)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--l2", type=float, default=0.1)
    a = ap.parse_args()
    rng = random.Random(1234)
    torch.manual_seed(1234)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="flash_attention_2",
                                                 device_map="cuda").eval()
    train_files = sorted(glob.glob(f"{a.mix}/train-*.parquet"))
    val_files = sorted(glob.glob(f"{a.mix}/validation-*.parquet"))
    train = rows_from(train_files, [11], 8000, a.train_rows, rng) + rows_from(train_files, [17], 8000, a.think_rows, rng)
    val = rows_from(val_files, [11, 17], 8000, 400, rng)
    print(f"train rows {len(train)}, validation rows {len(val)}", flush=True)
    Htr, Ttr, LSEtr, _, TLtr = collect(model, train, a.other_per_row, rng)
    Hva, Tva, _, MXva, _ = collect(model, val, a.other_per_row, rng)
    W0 = model.lm_head.weight[TAGS].detach().float().cpu()
    report = {"before": metrics(W0, Hva, Tva, MXva)}
    print("before:", json.dumps(report["before"]), flush=True)

    dev = "cuda"
    Htr, Ttr, LSEtr, TLtr = Htr.to(dev), Ttr.to(dev), LSEtr.to(dev), TLtr.to(dev)
    Ws = torch.nn.Parameter(W0.clone().to(dev))
    W0d = W0.to(dev)
    opt = torch.optim.Adam([Ws], lr=a.lr)
    tag_index = {t: i for i, t in enumerate(TAGS)}
    is_tag = torch.isin(Ttr, torch.tensor(TAGS, device=dev))
    col = torch.tensor([tag_index.get(int(t), 0) for t in Ttr.tolist()], device=dev)
    for step in range(a.steps + 1):
        tl = Htr @ Ws.T                                           # exact logits of the 8 tag rows
        lse = torch.logaddexp(LSEtr, torch.logsumexp(tl, -1))     # exact full-vocabulary normalizer
        target_logit = torch.where(is_tag, tl.gather(1, col[:, None])[:, 0], TLtr)
        ce = (lse - target_logit).mean()
        loss = ce + a.l2 * ((Ws - W0d) ** 2).sum()
        if step % 50 == 0:
            print(f"step {step} ce {ce.item():.4f} |dW| {(Ws - W0d).norm().item():.4f}", flush=True)
        if step == a.steps:
            break
        opt.zero_grad(); loss.backward(); opt.step()
    W1 = Ws.detach().cpu()
    report["after"] = metrics(W1, Hva, Tva, MXva)
    report["row_change_norm"] = {NAMES[t]: float((W1[i] - W0[i]).norm()) for i, t in enumerate(TAGS)}
    report["pair_separation_before_after"] = {
        f"{NAMES[a_]} vs {NAMES[b]}": [float((W0[a_ - 11] - W0[b - 11]).norm()), float((W1[a_ - 11] - W1[b - 11]).norm())]
        for a_, b in ((11, 12), (13, 14), (15, 16), (17, 18))}
    report["config"] = {k: str(v) for k, v in vars(a).items()}
    print("after:", json.dumps(report["after"]), flush=True)

    # write a new checkpoint directory; only the lm_head shard changes
    a.out.mkdir(parents=True, exist_ok=False)
    for f in a.model.iterdir():
        if f.is_file():
            shutil.copy2(f, a.out / f.name)
    shard = json.load(open(a.model / "model.safetensors.index.json"))["weight_map"]["lm_head.weight"]
    tensors = load_file(a.out / shard)
    w = tensors["lm_head.weight"]
    w[TAGS] = W1.to(w.dtype)
    save_file(tensors, a.out / shard, metadata={"format": "pt"})
    (a.out / "tag_head_repair.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
