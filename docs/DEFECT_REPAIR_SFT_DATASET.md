# Defect Repair SFT Dataset

`oellm-eu-defect-repair-sft-v1` is a monolingual SFT repair shard for the
European-language generation defects observed after Qwen post-training.

It targets:

- Tier 1 floor failures: `is`, `ga`, `mt`, `et`
- Tier 2 small-model loop failures: `hr`, `sl`, `lt`, `lv`, `da`, `hu`, `sk`
- Bulgarian Russian leakage: `bg`
- Touch-up languages: `ro`, `pl`, `fi`

Build:

```bash
env HF_HOME=/private/tmp/hf-cache python3 scripts/build_defect_repair_sft.py
```

Validate:

```bash
python3 scripts/validate_defect_repair_sft.py
```

Upload:

```bash
python3 scripts/upload_defect_repair_sft_hf.py
```

The strict SFT files live under
`data/defect_repair/oellm-eu-defect-repair-sft-v1/data/*.jsonl` and contain
only:

```json
{"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}],"lang":"is"}
```

License and provenance are sidecars, not extra training columns:

- `manifest.json`
- `source_registry.json`
- `metadata/all_metadata.jsonl`
- `metadata/quality_report.json`

Version `v0.1.0` is built from native `wikimedia/wikipedia` snapshots. It avoids
raw MT and should be upweighted as a repair shard, not blended like ordinary
high-resource instruction data.
