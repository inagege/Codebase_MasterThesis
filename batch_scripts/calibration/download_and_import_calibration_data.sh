#!/bin/bash
#SBATCH --job-name=calib_data_dl
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

DOWNLOAD_IMAGE="${DOWNLOAD_IMAGE:-1}"
DOWNLOAD_AUDIO="${DOWNLOAD_AUDIO:-1}"
DOWNLOAD_TEXT="${DOWNLOAD_TEXT:-1}"
DOWNLOAD_VIDEO="${DOWNLOAD_VIDEO:-0}"
DOWNLOAD_LARGE_AUDIO="${DOWNLOAD_LARGE_AUDIO:-0}" # enables MUSAN
DOWNLOAD_AUDIOSET="${DOWNLOAD_AUDIOSET:-0}"       # downloads AudioSet clips from YouTube via yt-dlp
GENERATE_TEXT_NOISE="${GENERATE_TEXT_NOISE:-1}"
TEXT_NOISE_MODE="${TEXT_NOISE_MODE:-chunked}" # chunked|full_per_severity
TEXT_NOISE_SEVERITIES="${TEXT_NOISE_SEVERITIES:-1,2,3,4,5}"
TEXT_NOISE_SEED="${TEXT_NOISE_SEED:-123}"
TEXT_NOISE_OVERWRITE="${TEXT_NOISE_OVERWRITE:-1}"

KONIQ_URL="${KONIQ_URL:-https://datasets.vqa.mmsp-kn.de/archives/koniq10k_512x384.zip}"
KADID_URL="${KADID_URL:-https://datasets.vqa.mmsp-kn.de/archives/kadid10k.zip}"
ODAQ_URL="${ODAQ_URL:-https://zenodo.org/records/10405774/files/ODAQ.zip?download=1}"
ESC50_URL="${ESC50_URL:-https://github.com/karoldvl/ESC-50/archive/master.zip}"
MUSAN_URL="${MUSAN_URL:-https://www.openslr.org/resources/17/musan.tar.gz}"
WIKITEXT_URL="${WIKITEXT_URL:-https://huggingface.co/datasets/mattdangerw/wikitext-103-raw/resolve/main/wikitext-103-raw-v1.zip?download=1}"

AUDIOSET_SPLITS="${AUDIOSET_SPLITS:-balanced,eval}" # balanced|eval|unbalanced (comma-separated)
AUDIOSET_MAX_CLIPS_PER_SPLIT="${AUDIOSET_MAX_CLIPS_PER_SPLIT:-2000}"
AUDIOSET_NUM_WORKERS="${AUDIOSET_NUM_WORKERS:-8}"
AUDIOSET_SAMPLE_SEED="${AUDIOSET_SAMPLE_SEED:-123}"
AUDIOSET_RETRIES="${AUDIOSET_RETRIES:-2}"
AUDIOSET_TIMEOUT_SECONDS="${AUDIOSET_TIMEOUT_SECONDS:-180}"
AUDIOSET_OVERWRITE="${AUDIOSET_OVERWRITE:-0}" # redownload clips even if they already exist
AUDIOSET_COOKIES_FILE="${AUDIOSET_COOKIES_FILE:-}" # optional cookies.txt for yt-dlp

# Optional video URL (example: UCF101 archive). If empty, video is skipped.
VIDEO_URL="${VIDEO_URL:-}"
VIDEO_DATASET_NAME="${VIDEO_DATASET_NAME:-ucf101}"
WGET_INSECURE="${WGET_INSECURE:-1}" # set to 0 to enforce cert validation

mkdir -p "${SOURCE_ROOT}"

download_if_missing() {
  local url="$1"
  local out_file="$2"
  if [[ -f "${out_file}" ]]; then
    echo "[INFO] Already downloaded: ${out_file}"
    return 0
  fi
  echo "[INFO] Downloading ${url}"
  wget_args=(-c --content-disposition -O "${out_file}")
  if [[ "${WGET_INSECURE}" == "1" ]]; then
    wget_args+=(--no-check-certificate)
  fi
  wget "${wget_args[@]}" "${url}"
}

ensure_valid_zip_archive() {
  local url="$1"
  local archive="$2"
  if [[ -f "${archive}" ]] && ! unzip -tq "${archive}" >/dev/null 2>&1; then
    echo "[WARN] Corrupt ZIP detected, removing: ${archive}"
    rm -f "${archive}"
  fi
  download_if_missing "${url}" "${archive}"
  if ! unzip -tq "${archive}" >/dev/null 2>&1; then
    echo "[ERROR] ZIP validation failed after download: ${archive}"
    exit 1
  fi
}

ensure_valid_targz_archive() {
  local url="$1"
  local archive="$2"
  if [[ -f "${archive}" ]] && ! tar -tzf "${archive}" >/dev/null 2>&1; then
    echo "[WARN] Corrupt TAR.GZ detected, removing: ${archive}"
    rm -f "${archive}"
  fi
  download_if_missing "${url}" "${archive}"
  if ! tar -tzf "${archive}" >/dev/null 2>&1; then
    echo "[ERROR] TAR.GZ validation failed after download: ${archive}"
    exit 1
  fi
}

extract_zip_if_needed() {
  local archive="$1"
  local dest="$2"
  if [[ -d "${dest}" ]] && find "${dest}" -type f -print -quit | grep -q .; then
    echo "[INFO] Already extracted: ${dest}"
    return 0
  fi
  if [[ -d "${dest}" ]]; then
    echo "[WARN] Existing extraction directory is empty, refreshing: ${dest}"
    rm -rf "${dest}"
  fi
  mkdir -p "${dest}"
  echo "[INFO] Extracting ZIP ${archive} -> ${dest}"
  unzip -q -n "${archive}" -d "${dest}"
}

extract_targz_if_needed() {
  local archive="$1"
  local dest="$2"
  if [[ -d "${dest}" ]] && find "${dest}" -type f -print -quit | grep -q .; then
    echo "[INFO] Already extracted: ${dest}"
    return 0
  fi
  if [[ -d "${dest}" ]]; then
    echo "[WARN] Existing extraction directory is empty, refreshing: ${dest}"
    rm -rf "${dest}"
  fi
  mkdir -p "${dest}"
  echo "[INFO] Extracting TAR.GZ ${archive} -> ${dest}"
  tar -xzf "${archive}" -C "${dest}"
}

declare -a DATASET_ARGS=()

if [[ "${DOWNLOAD_IMAGE}" == "1" ]]; then
  mkdir -p "${SOURCE_ROOT}/archives"
  ensure_valid_zip_archive "${KONIQ_URL}" "${SOURCE_ROOT}/archives/koniq10k_512x384.zip"
  extract_zip_if_needed "${SOURCE_ROOT}/archives/koniq10k_512x384.zip" "${SOURCE_ROOT}/koniq10k"
  DATASET_ARGS+=(--dataset-path "koniq10k=${SOURCE_ROOT}/koniq10k")

  ensure_valid_zip_archive "${KADID_URL}" "${SOURCE_ROOT}/archives/kadid10k.zip"
  extract_zip_if_needed "${SOURCE_ROOT}/archives/kadid10k.zip" "${SOURCE_ROOT}/kadid10k"
  DATASET_ARGS+=(--dataset-path "kadid10k=${SOURCE_ROOT}/kadid10k")
fi

if [[ "${DOWNLOAD_AUDIO}" == "1" ]]; then
  mkdir -p "${SOURCE_ROOT}/archives"
  ensure_valid_zip_archive "${ODAQ_URL}" "${SOURCE_ROOT}/archives/ODAQ.zip"
  extract_zip_if_needed "${SOURCE_ROOT}/archives/ODAQ.zip" "${SOURCE_ROOT}/odaq"
  DATASET_ARGS+=(--dataset-path "odaq=${SOURCE_ROOT}/odaq")

  ensure_valid_zip_archive "${ESC50_URL}" "${SOURCE_ROOT}/archives/esc50_master.zip"
  extract_zip_if_needed "${SOURCE_ROOT}/archives/esc50_master.zip" "${SOURCE_ROOT}/esc50"
  DATASET_ARGS+=(--dataset-path "esc50=${SOURCE_ROOT}/esc50")

  if [[ "${DOWNLOAD_AUDIOSET}" == "1" ]]; then
    AUDIOSET_ARGS=(
      --out-root "${SOURCE_ROOT}/audioset"
      --splits "${AUDIOSET_SPLITS}"
      --max-clips-per-split "${AUDIOSET_MAX_CLIPS_PER_SPLIT}"
      --sample-seed "${AUDIOSET_SAMPLE_SEED}"
      --num-workers "${AUDIOSET_NUM_WORKERS}"
      --retries "${AUDIOSET_RETRIES}"
      --download-timeout-seconds "${AUDIOSET_TIMEOUT_SECONDS}"
    )
    if [[ "${AUDIOSET_OVERWRITE}" == "1" ]]; then
      AUDIOSET_ARGS+=(--overwrite-existing)
    fi
    if [[ -n "${AUDIOSET_COOKIES_FILE}" ]]; then
      AUDIOSET_ARGS+=(--cookies "${AUDIOSET_COOKIES_FILE}")
    fi

    echo "[INFO] Preparing AudioSet clips for calibration"
    echo "[INFO]   splits=${AUDIOSET_SPLITS}"
    echo "[INFO]   max_clips_per_split=${AUDIOSET_MAX_CLIPS_PER_SPLIT}"
    echo "[INFO]   workers=${AUDIOSET_NUM_WORKERS}"
    pixi run python utils/download_prepare_audioset.py "${AUDIOSET_ARGS[@]}"
    DATASET_ARGS+=(--dataset-path "audioset=${SOURCE_ROOT}/audioset/clips")
  fi

  if [[ "${DOWNLOAD_LARGE_AUDIO}" == "1" ]]; then
    ensure_valid_targz_archive "${MUSAN_URL}" "${SOURCE_ROOT}/archives/musan.tar.gz"
    extract_targz_if_needed "${SOURCE_ROOT}/archives/musan.tar.gz" "${SOURCE_ROOT}/musan"
    DATASET_ARGS+=(--dataset-path "musan=${SOURCE_ROOT}/musan")
  fi
fi

if [[ "${DOWNLOAD_TEXT}" == "1" ]]; then
  mkdir -p "${SOURCE_ROOT}/archives"
  ensure_valid_zip_archive "${WIKITEXT_URL}" "${SOURCE_ROOT}/archives/wikitext-103-v1.zip"
  extract_zip_if_needed "${SOURCE_ROOT}/archives/wikitext-103-v1.zip" "${SOURCE_ROOT}/wikitext103"
  DATASET_ARGS+=(--dataset-path "wikitext103=${SOURCE_ROOT}/wikitext103")

  if [[ "${GENERATE_TEXT_NOISE}" == "1" ]]; then
    TEXT_NOISE_ROOT="${SOURCE_ROOT}/wikitext103_noise"
    if [[ "${TEXT_NOISE_OVERWRITE}" == "1" ]]; then
      rm -rf "${TEXT_NOISE_ROOT}"
    fi
    NOISE_ARGS=(
      --input-dir "${SOURCE_ROOT}/wikitext103"
      --out-dir "${TEXT_NOISE_ROOT}"
      --mode "${TEXT_NOISE_MODE}"
      --severities "${TEXT_NOISE_SEVERITIES}"
      --seed "${TEXT_NOISE_SEED}"
    )
    if [[ "${TEXT_NOISE_OVERWRITE}" == "1" ]]; then
      NOISE_ARGS+=(--overwrite)
    fi

    echo "[INFO] Generating text noise variants for calibration (severities=${TEXT_NOISE_SEVERITIES})"
    pixi run python utils/apply_text_noise_to_calib.py "${NOISE_ARGS[@]}"

    while IFS= read -r variant_dir; do
      variant_base="$(basename "${variant_dir}")"
      dataset_name="${variant_base//=/_}"
      dataset_name="${dataset_name//-/_}"
      dataset_name="${dataset_name,,}"
      DATASET_ARGS+=(--dataset-path "wikitext103_${dataset_name}=${variant_dir}")
    done < <(find "${TEXT_NOISE_ROOT}" -mindepth 1 -maxdepth 1 -type d | sort)
  fi
fi

if [[ "${DOWNLOAD_VIDEO}" == "1" ]]; then
  if [[ -z "${VIDEO_URL}" ]]; then
    echo "[ERROR] DOWNLOAD_VIDEO=1 but VIDEO_URL is empty."
    exit 1
  fi
  mkdir -p "${SOURCE_ROOT}/archives"
  video_archive="${SOURCE_ROOT}/archives/${VIDEO_DATASET_NAME}.zip"
  ensure_valid_zip_archive "${VIDEO_URL}" "${video_archive}"
  extract_zip_if_needed "${video_archive}" "${SOURCE_ROOT}/${VIDEO_DATASET_NAME}"
  DATASET_ARGS+=(--dataset-path "${VIDEO_DATASET_NAME}=${SOURCE_ROOT}/${VIDEO_DATASET_NAME}")
fi

if [[ ${#DATASET_ARGS[@]} -eq 0 ]]; then
  echo "[ERROR] No datasets selected for download/import."
  exit 1
fi

echo "[INFO] Importing downloaded datasets into ${CALIB_ROOT}"
pixi run python utils/import_calibration_datasets.py \
  "${DATASET_ARGS[@]}" \
  --out-root "${CALIB_ROOT}" \
  --mode "${IMPORT_MODE}" \
  --overwrite

echo "[INFO] Done."
echo "[INFO] calibration_data root: ${CALIB_ROOT}"
echo "[INFO] manifests: ${CALIB_ROOT}/manifests"
