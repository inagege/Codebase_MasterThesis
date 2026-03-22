#!/bin/bash
#SBATCH --job-name=analyze_quality
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=normal

set -euo pipefail

cd /hkfs/work/workspace_haic/scratch/unsvk-multimodal/Codebase_MasterThesis || exit 1
mkdir -p logs

SCORES_CSV="${SCORES_CSV:-data/calibration_data/noise_1m/scores_imported_datasets_1m_noise_audio.csv}"
CALIBRATION_JSON="${CALIBRATION_JSON:-data/calibration_data/noise_1m/quality_percentile_calibration_1m_noise_audio.json}"
OUT_DIR="${OUT_DIR:-analysis/quality_score_percentile_report}"
MODALITIES="${MODALITIES:-audio}"
PERCENTILES="${PERCENTILES:-1,5,10,25,50,75,90,95,99}"

CALIBRATION_ARGS=()
if [[ -f "${CALIBRATION_JSON}" ]]; then
  CALIBRATION_ARGS=(--calibration-json "${CALIBRATION_JSON}")
else
  echo "[WARN] Calibration JSON not found at ${CALIBRATION_JSON}; deriving percentiles from input scores."
fi

pixi run python analysis/analyze_quality_score_distributions.py \
  --scores-csv "${SCORES_CSV}" \
  --out-dir "${OUT_DIR}" \
  --modalities "${MODALITIES}" \
  --percentiles "${PERCENTILES}" \
  "${CALIBRATION_ARGS[@]}"
