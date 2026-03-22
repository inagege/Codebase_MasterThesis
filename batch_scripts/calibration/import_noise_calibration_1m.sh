#!/bin/bash
#SBATCH --job-name=calib_1m_prepare
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=normal

set -euo pipefail

cd /hkfs/work/workspace_haic/scratch/unsvk-multimodal/Codebase_MasterThesis || exit 1
mkdir -p logs

# -------------------------
# Configuration (override via env vars before sbatch)
# -------------------------
CALIB_ROOT="${CALIB_ROOT:-data/calibration_data/noise_1m}"
GENERATED_ROOT="${GENERATED_ROOT:-${CALIB_ROOT}/generated_sources}"
MODALITIES="${MODALITIES:-audio}" # audio,image,text

SEED="${SEED:-123}"
CLEAN_LINK_MODE="${CLEAN_LINK_MODE:-symlink}" # symlink|copy
OVERWRITE_GENERATED="${OVERWRITE_GENERATED:-0}" # 1=yes, 0=no
IMPORT_MODE="${IMPORT_MODE:-symlink}" # symlink|copy
IMPORT_OVERWRITE="${IMPORT_OVERWRITE:-1}" # 1=yes, 0=no
MANIFEST_MAX_FILES_PER_DATASET="${MANIFEST_MAX_FILES_PER_DATASET:-0}" # 0=all

TEXT_EXTENSIONS="${TEXT_EXTENSIONS:-.txt,.raw,.md,.tokens}"
TEXT_MIN_CHARS="${TEXT_MIN_CHARS:-160}"
TEXT_MAX_CHARS="${TEXT_MAX_CHARS:-600}"
TEXT_POOL_MAX_CHUNKS="${TEXT_POOL_MAX_CHUNKS:-2000000}"
AUDIO_SR="${AUDIO_SR:-16000}"

declare -a DATASET_ARGS=()
if [[ ",${MODALITIES}," == *",audio,"* ]]; then
  DATASET_ARGS+=(--dataset-path "audioset_1m=${GENERATED_ROOT}/audio_1m")
fi
if [[ ",${MODALITIES}," == *",image,"* ]]; then
  DATASET_ARGS+=(--dataset-path "imagenet_1m=${GENERATED_ROOT}/image_1m")
fi
if [[ ",${MODALITIES}," == *",text,"* ]]; then
  DATASET_ARGS+=(--dataset-path "wikitext103_1m=${GENERATED_ROOT}/text_1m")
fi

if [[ ${#DATASET_ARGS[@]} -eq 0 ]]; then
  echo "[ERROR] No datasets selected for import. Check MODALITIES=${MODALITIES}."
  exit 1
fi

IMPORT_ARGS=(
  "${DATASET_ARGS[@]}"
  --out-root "${CALIB_ROOT}"
  --mode "${IMPORT_MODE}"
)
if [[ "${IMPORT_OVERWRITE}" == "1" ]]; then
  IMPORT_ARGS+=(--overwrite)
fi
if [[ "${MANIFEST_MAX_FILES_PER_DATASET}" != "0" ]]; then
  IMPORT_ARGS+=(--manifest-max-files-per-dataset "${MANIFEST_MAX_FILES_PER_DATASET}")
fi

echo "[INFO] Importing generated 1M sets into ${CALIB_ROOT}"
pixi run python utils/calibration/import_calibration_datasets.py "${IMPORT_ARGS[@]}"

echo "[INFO] Done."
echo "[INFO] calibration_data root: ${CALIB_ROOT}"
echo "[INFO] manifests: ${CALIB_ROOT}/manifests"