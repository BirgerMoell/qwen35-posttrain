#!/bin/bash
#SBATCH --job-name=oellm256k-stage
#SBATCH --account=aifac_f02_352
#SBATCH --partition=lrd_all_serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=/leonardo_work/AIFAC_F02_352/pmoell00/sft-migration-preflight/logs/%j.out
#SBATCH --error=/leonardo_work/AIFAC_F02_352/pmoell00/sft-migration-preflight/logs/%j.err

set -euo pipefail

revision=783ab41b06d1937b10b5b8fde232051f7ffd2998
expected_sha=5dcf789668d1233b48c8e99bf3733f0e80abbde4a74de1974b0605108030378c
base_url="https://huggingface.co/openeurollm/oellm-9b-256k-theta64m-prelude-anneal300b/resolve/${revision}"
target=/leonardo_work/AIFAC_F02_352/pmoell00/sft-migration-preflight/base-256k-anneal300b
mkdir -p "$target"

curl --fail --location --retry 5 --connect-timeout 30 \
  --output "$target/config.json.part" "$base_url/config.json"
mv "$target/config.json.part" "$target/config.json"

if [[ -f "$target/pytorch_model.bin" ]]; then
  echo "$expected_sha  $target/pytorch_model.bin" | sha256sum --check
  echo "Verified existing 256K base weights"
  exit 0
fi

curl --fail --location --retry 5 --connect-timeout 30 \
  --continue-at - --output "$target/pytorch_model.bin.part" \
  "$base_url/pytorch_model.bin"
echo "$expected_sha  $target/pytorch_model.bin.part" | sha256sum --check
mv "$target/pytorch_model.bin.part" "$target/pytorch_model.bin"
echo "Verified and staged 256K base weights at $target"
