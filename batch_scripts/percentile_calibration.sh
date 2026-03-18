#!/bin/bash
#SBATCH --job-name=calibrate
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:full:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=normal

set -euo pipefail

module load devel/cuda/12.9

cd /hkfs/work/workspace_haic/scratch/unsvk-multimodal/Codebase_MasterThesis || exit 1
mkdir -p logs

# -------------------------
# Configuration (override via env vars before sbatch)
# -------------------------
TEXT_MAX_CHUNKS="${TEXT_MAX_CHUNKS:-500000}"
TEXT_STRATIFY_BY="${TEXT_STRATIFY_BY:-perturbation}" # none|dataset|perturbation
TEXT_SAMPLING_SEED="${TEXT_SAMPLING_SEED:-123}"

echo "[INFO] Text calibration sampling: max_chunks=${TEXT_MAX_CHUNKS} stratify_by=${TEXT_STRATIFY_BY} seed=${TEXT_SAMPLING_SEED}"

echo "[INFO] Collecting calibration quality scores from imported datasets"
pixi run python utils/collect_calibration_scores_from_manifests.py \
  --manifest-dir "data/calibration_data/manifests" \
  --modalities "audio,image,text" \
  --max-text-chunks "${TEXT_MAX_CHUNKS}" \
  --text-stratify-by "${TEXT_STRATIFY_BY}" \
  --text-sampling-seed "${TEXT_SAMPLING_SEED}" \
  --out-path "data/calibration_data/scores_imported_datasets.csv"

echo "[INFO] Building frozen percentile calibration JSON"
pixi run python utils/build_quality_percentile_calibration.py \
  --input-csv "data/calibration_data/scores_imported_datasets.csv" \
  --out-path "data/calibration_data/quality_percentile_calibration.json" \
  --modalities "audio,image,text"
