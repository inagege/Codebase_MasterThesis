#!/bin/bash
#SBATCH --job-name=calib_1m_audio
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:full:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=normal

set -euo pipefail

module load devel/cuda/12.9

cd /hkfs/work/workspace_haic/scratch/unsvk-multimodal/Codebase_MasterThesis || exit 1
mkdir -p logs

# -------------------------
# Configuration (override via env vars before sbatch)
# -------------------------
CALIB_ROOT="${CALIB_ROOT:-data/calibration_data/noise_1m}"
MANIFEST_DIR="${MANIFEST_DIR:-${CALIB_ROOT}/manifests}"
MAX_FILES_PER_MODALITY="${MAX_FILES_PER_MODALITY:-1000000}"

SCORES_CSV="${SCORES_CSV:-${CALIB_ROOT}/scores_imported_datasets_1m_noise_audio.csv}"
CALIB_JSON="${CALIB_JSON:-${CALIB_ROOT}/quality_percentile_calibration_1m_noise_audio.json}"

echo "[INFO] 1M audio calibration settings:"
echo "[INFO]   manifest_dir=${MANIFEST_DIR}"
echo "[INFO]   max_files_per_modality=${MAX_FILES_PER_MODALITY}"
echo "[INFO]   scores_csv=${SCORES_CSV}"
echo "[INFO]   calib_json=${CALIB_JSON}"

echo "[INFO] Collecting audio quality scores (1M noisy sets)"
pixi run python utils/calibration/collect_calibration_scores_from_manifests.py \
  --manifest-dir "${MANIFEST_DIR}" \
  --modalities "audio" \
  --max-files-per-modality "${MAX_FILES_PER_MODALITY}" \
  --out-path "${SCORES_CSV}"

echo "[INFO] Building audio percentile calibration JSON for 1M noisy sets"
pixi run python utils/calibration/build_quality_percentile_calibration.py \
  --input-csv "${SCORES_CSV}" \
  --out-path "${CALIB_JSON}" \
  --modalities "audio"

echo "[INFO] Done."
echo "[INFO] new audio calibration JSON: ${CALIB_JSON}"

