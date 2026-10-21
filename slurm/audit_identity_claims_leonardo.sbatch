#!/bin/bash
#SBATCH --job-name=oellm-identity-audit
#SBATCH --account=aifac_f02_352
#SBATCH --partition=lrd_all_serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=02:00:00
#SBATCH --output=/leonardo_work/AIFAC_F02_352/pmoell00/sft-migration-preflight/logs/%j.out
#SBATCH --error=/leonardo_work/AIFAC_F02_352/pmoell00/sft-migration-preflight/logs/%j.err
set -euo pipefail
root=/leonardo_work/AIFAC_F02_352/pmoell00/sft-migration-preflight
srun singularity exec -B /leonardo_work:/leonardo_work /leonardo_work/OELLM_prod2026/container_images/post-training-25.10-transformers-olmo3-patch.sif \
  python "$root/audit_identity_claims.py" --root "$root" --out "$root/reports/stage1-mix-v1-identity-claims.json"
