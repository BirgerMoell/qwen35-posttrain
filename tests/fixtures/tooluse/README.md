# Tool-use adapter fixtures

Small unmodified samples (≤20 rows each) from public datasets, used only to test the adapters in
`scripts/tooluse_adapters.py`. Each file keeps its upstream license and attribution:

| File | Source (revision) | License |
|---|---|---|
| `eu_tooluse.jsonl` | birgermoell/oellm-eu-tooluse-v1 (b131d571) | Apache-2.0 |
| `when2call_sft.jsonl` | nvidia/When2Call `train/when2call_train_sft.jsonl` (0582f774) | CC-BY-4.0 |
| `nemotron_agentic.jsonl` | nvidia/Nemotron-SFT-Agentic-v2 `data/tool_calling.jsonl` (7c804833), first 8 rows | CC-BY-4.0 |
| `toucan.jsonl` | Agent-Ark/Toucan-1.5M `SFT/` (0df3cf37) | Apache-2.0 |
| `toolmind.jsonl` | Nanbeige/ToolMind `open_datasets/ToolACE-query.jsonl` (8020ed1c) | Apache-2.0 |
| `toolace.jsonl` | Team-ACE/ToolACE `data.json` (6bda777c) | Apache-2.0 |
