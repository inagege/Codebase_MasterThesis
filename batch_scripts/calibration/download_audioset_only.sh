#!/bin/bash
#SBATCH --job-name=calib_audioset_dl
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=normal

set -euo pipefail

cd /hkfs/work/workspace_haic/scratch/unsvk-multimodal/Codebase_MasterThesis || exit 1
mkdir -p logs

# -------------------------
# Configuration (override via env vars before sbatch)
# -------------------------
CALIB_ROOT="${CALIB_ROOT:-data/calibration_data}"
SOURCE_ROOT="${SOURCE_ROOT:-${CALIB_ROOT}/sources}"
IMPORT_MODE="${IMPORT_MODE:-symlink}" # symlink|copy
IMPORT_OVERWRITE="${IMPORT_OVERWRITE:-1}" # 1=yes, 0=no
MANIFEST_MAX_FILES_PER_DATASET="${MANIFEST_MAX_FILES_PER_DATASET:-0}" # 0=all

AUDIOSET_REPO_ID="${AUDIOSET_REPO_ID:-confit/audioset-16khz-wds}"
AUDIOSET_SUBSET="${AUDIOSET_SUBSET:-500k}" # 20k|500k|2m
AUDIOSET_SPLITS="${AUDIOSET_SPLITS:-train,test}" # train|test
AUDIOSET_EXTRACT_WORKERS="${AUDIOSET_EXTRACT_WORKERS:-8}"
AUDIOSET_MAX_SHARDS_PER_SPLIT="${AUDIOSET_MAX_SHARDS_PER_SPLIT:-0}" # 0=all
AUDIOSET_OVERWRITE_EXTRACTED="${AUDIOSET_OVERWRITE_EXTRACTED:-0}"
AUDIOSET_SKIP_DOWNLOAD="${AUDIOSET_SKIP_DOWNLOAD:-0}"
AUDIOSET_SKIP_EXTRACT="${AUDIOSET_SKIP_EXTRACT:-0}"
AUDIOSET_HF_TOKEN="${AUDIOSET_HF_TOKEN:-${HF_TOKEN:-}}"

mkdir -p "${SOURCE_ROOT}"

PREP_ARGS=(
  --repo-id "${AUDIOSET_REPO_ID}"
  --subset "${AUDIOSET_SUBSET}"
  --out-root "${SOURCE_ROOT}/audioset"
  --splits "${AUDIOSET_SPLITS}"
  --extract-workers "${AUDIOSET_EXTRACT_WORKERS}"
  --max-shards-per-split "${AUDIOSET_MAX_SHARDS_PER_SPLIT}"
)
if [[ "${AUDIOSET_OVERWRITE_EXTRACTED}" == "1" ]]; then
  PREP_ARGS+=(--overwrite-extracted)
fi
if [[ "${AUDIOSET_SKIP_DOWNLOAD}" == "1" ]]; then
  PREP_ARGS+=(--skip-download)
fi
if [[ "${AUDIOSET_SKIP_EXTRACT}" == "1" ]]; then
  PREP_ARGS+=(--skip-extract)
fi
if [[ -n "${AUDIOSET_HF_TOKEN}" ]]; then
  PREP_ARGS+=(--hf-token "${AUDIOSET_HF_TOKEN}")
fi

echo "[INFO] Preparing AudioSet clips for calibration"
echo "[INFO]   out_root=${SOURCE_ROOT}/audioset"
echo "[INFO]   repo_id=${AUDIOSET_REPO_ID}"
echo "[INFO]   subset=${AUDIOSET_SUBSET}"
echo "[INFO]   splits=${AUDIOSET_SPLITS}"
echo "[INFO]   extract_workers=${AUDIOSET_EXTRACT_WORKERS}"
echo "[INFO]   max_shards_per_split=${AUDIOSET_MAX_SHARDS_PER_SPLIT}"
pixi run python utils/download_prepare_audioset_from_hf.py "${PREP_ARGS[@]}"

IMPORT_ARGS=(
  --dataset-path "audioset=${SOURCE_ROOT}/audioset/clips/${AUDIOSET_SUBSET}"
  --out-root "${CALIB_ROOT}"
  --mode "${IMPORT_MODE}"
)
if [[ "${IMPORT_OVERWRITE}" == "1" ]]; then
  IMPORT_ARGS+=(--overwrite)
fi
if [[ "${MANIFEST_MAX_FILES_PER_DATASET}" != "0" ]]; then
  IMPORT_ARGS+=(--manifest-max-files-per-dataset "${MANIFEST_MAX_FILES_PER_DATASET}")
fi

echo "[INFO] Importing AudioSet into calibration root"
pixi run python utils/import_calibration_datasets.py "${IMPORT_ARGS[@]}"

echo "[INFO] Done."
echo "[INFO] calibration_data root: ${CALIB_ROOT}"
echo "[INFO] manifests: ${CALIB_ROOT}/manifests"
