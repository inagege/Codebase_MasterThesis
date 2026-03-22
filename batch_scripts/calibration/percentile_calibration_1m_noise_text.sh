#!/bin/bash
#SBATCH --job-name=calib_1m_text
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=48:00:00
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
MAX_TEXT_CHUNKS="${MAX_TEXT_CHUNKS:-1000000}"
TEXT_STRATIFY_BY="${TEXT_STRATIFY_BY:-none}" # none|dataset|perturbation
TEXT_SAMPLING_SEED="${TEXT_SAMPLING_SEED:-123}"

SCORES_CSV="${SCORES_CSV:-${CALIB_ROOT}/scores_imported_datasets_1m_noise_text.csv}"
CALIB_JSON="${CALIB_JSON:-${CALIB_ROOT}/quality_percentile_calibration_1m_noise_text.json}"

echo "[INFO] 1M text calibration settings:"
echo "[INFO]   manifest_dir=${MANIFEST_DIR}"
echo "[INFO]   max_files_per_modality=${MAX_FILES_PER_MODALITY}"
echo "[INFO]   max_text_chunks=${MAX_TEXT_CHUNKS}"
echo "[INFO]   text_stratify_by=${TEXT_STRATIFY_BY}"
echo "[INFO]   scores_csv=${SCORES_CSV}"
echo "[INFO]   calib_json=${CALIB_JSON}"

echo "[INFO] Collecting text quality scores (1M noisy sets)"
pixi run python utils/calibration/collect_calibration_scores_from_manifests.py \
  --manifest-dir "${MANIFEST_DIR}" \
  --modalities "text" \
  --max-files-per-modality "${MAX_FILES_PER_MODALITY}" \
  --max-text-chunks "${MAX_TEXT_CHUNKS}" \
  --text-stratify-by "${TEXT_STRATIFY_BY}" \
  --text-sampling-seed "${TEXT_SAMPLING_SEED}" \
  --out-path "${SCORES_CSV}"

echo "[INFO] Building text percentile calibration JSON for 1M noisy sets"
pixi run python utils/calibration/build_quality_percentile_calibration.py \
  --input-csv "${SCORES_CSV}" \
  --out-path "${CALIB_JSON}" \
  --modalities "text"

echo "[INFO] Done."
echo "[INFO] new text calibration JSON: ${CALIB_JSON}"

