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

echo "[INFO] Collecting calibration quality scores from imported datasets"
pixi run python utils/collect_calibration_scores_from_manifests.py \
  --manifest-dir "data/calibration_data/manifests" \
  --modalities "audio,image,text" \
  --out-path "data/calibration_data/scores_imported_datasets.csv"

echo "[INFO] Building frozen percentile calibration JSON"
pixi run python utils/build_quality_percentile_calibration.py \
  --input-csv "data/calibration_data/scores_imported_datasets.csv" \
  --out-path "data/calibration_data/quality_percentile_calibration.json" \
  --modalities "audio,image,text"