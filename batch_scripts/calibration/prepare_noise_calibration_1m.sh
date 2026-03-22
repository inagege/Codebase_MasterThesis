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
MODALITIES="${MODALITIES:-audio,image,text}" # audio,image,text

# Expected source datasets for this new pipeline:
# - AudioSet (HF mirror extracted clips, e.g. subset 500k)
# - ImageNet
# - WikiText
AUDIO_SOURCE_DIR="${AUDIO_SOURCE_DIR:-data/calibration_data/sources/audioset/clips/500k}"
TEXT_SOURCE_DIR="${TEXT_SOURCE_DIR:-data/calibration_data/sources/wikitext103}"

# IMAGE_SOURCE_DIR resolution priority:
# 1) explicit IMAGE_SOURCE_DIR
# 2) extracted ImageNet train JPEG directory under $DATASETS/imagenet-2012
# 3) local fallback under repository
if [[ -z "${IMAGE_SOURCE_DIR:-}" ]]; then
  if [[ -n "${DATASETS:-}" ]]; then
    if [[ -d "${DATASETS}/imagenet-2012/original/imagenet-raw/ILSVRC/Data/CLS-LOC/train" ]]; then
      IMAGE_SOURCE_DIR="${DATASETS}/imagenet-2012/original/imagenet-raw/ILSVRC/Data/CLS-LOC/train"
    else
      IMAGE_SOURCE_DIR="${DATASETS}/imagenet-2012"
    fi
  else
    IMAGE_SOURCE_DIR="data/calibration_data/sources/imagenet"
  fi
fi

CLEAN_COUNT="${CLEAN_COUNT:-500000}"
S1_COUNT="${S1_COUNT:-150000}"
S2_COUNT="${S2_COUNT:-100000}"
S3_COUNT="${S3_COUNT:-100000}"
S4_COUNT="${S4_COUNT:-100000}"
S5_COUNT="${S5_COUNT:-50000}"

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

mkdir -p "${CALIB_ROOT}" "${GENERATED_ROOT}"

BUILD_ARGS=(
  --out-root "${GENERATED_ROOT}"
  --modalities "${MODALITIES}"
  --audio-source-dir "${AUDIO_SOURCE_DIR}"
  --image-source-dir "${IMAGE_SOURCE_DIR}"
  --text-source-dir "${TEXT_SOURCE_DIR}"
  --clean-count "${CLEAN_COUNT}"
  --severity-1-count "${S1_COUNT}"
  --severity-2-count "${S2_COUNT}"
  --severity-3-count "${S3_COUNT}"
  --severity-4-count "${S4_COUNT}"
  --severity-5-count "${S5_COUNT}"
  --seed "${SEED}"
  --clean-link-mode "${CLEAN_LINK_MODE}"
  --audio-sr "${AUDIO_SR}"
  --text-extensions "${TEXT_EXTENSIONS}"
  --text-min-chars "${TEXT_MIN_CHARS}"
  --text-max-chars "${TEXT_MAX_CHARS}"
  --text-pool-max-chunks "${TEXT_POOL_MAX_CHUNKS}"
)
if [[ "${OVERWRITE_GENERATED}" == "1" ]]; then
  BUILD_ARGS+=(--overwrite)
fi

echo "[INFO] Generating 1M severity-balanced modality sets"
echo "[INFO]   calib_root=${CALIB_ROOT}"
echo "[INFO]   generated_root=${GENERATED_ROOT}"
echo "[INFO]   modalities=${MODALITIES}"
echo "[INFO]   clean=${CLEAN_COUNT} s1=${S1_COUNT} s2=${S2_COUNT} s3=${S3_COUNT} s4=${S4_COUNT} s5=${S5_COUNT}"
echo "[INFO]   audio_source_dir=${AUDIO_SOURCE_DIR}"
echo "[INFO]   image_source_dir=${IMAGE_SOURCE_DIR}"
echo "[INFO]   text_source_dir=${TEXT_SOURCE_DIR}"
pixi run python utils/calibration/build_noise_calibration_sets.py "${BUILD_ARGS[@]}"

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

