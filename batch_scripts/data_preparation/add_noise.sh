#!/bin/bash
#SBATCH --job-name=add_noise                 # Job name
#SBATCH --output=logs/%x_%j.out           # Stdout log
#SBATCH --error=logs/%x_%j.err            # Stderr log
#SBATCH --time=4:00:00                   # Max runtime (hh:mm:ss)
#SBATCH --gres=gpu:full:1                 # Request 1 GPU
#SBATCH --cpus-per-task=8                 # CPU cores
#SBATCH --mem=100G                        # RAM
#SBATCH --partition=normal                # Partition name

# Go to your project directory
cd /hkfs/work/workspace_haic/scratch/ulrat-masters/MasterThesis/Codebase_MasterThesis || exit 1

# (Optional but recommended) create logs directory if it doesn't exist
mkdir -p logs

# Usage examples:
# sbatch batch_scripts/add_noise.sh meld test 3 text,audio,video
# sbatch batch_scripts/add_noise.sh homeprice all 3 text,image 5000
# sbatch batch_scripts/add_noise.sh imdb all 3 text,image 5000
# sbatch batch_scripts/add_noise.sh voxceleb all 3 audio,video 5000
# sbatch batch_scripts/add_noise.sh nejm all 3 text,image 5000
# sbatch batch_scripts/add_noise.sh marine all 3 audio,image 5000

DATASET="${1:-meld}"
SPLIT="${2:-test}"        # only used for MELD
SEVERITY="${3:-3}"
MODALITIES="${4:-}"       # optional comma list
STRATIFIED_SAMPLES="${5:-}"  # optional, non-MELD only

CMD=(
  pixi run python utils/apply_dataset_noise.py
  --dataset "${DATASET}"
  --split "${SPLIT}"
  --severity "${SEVERITY}"
  --overwrite
)

if [[ -n "${MODALITIES}" ]]; then
  CMD+=(--modalities "${MODALITIES}")
fi

if [[ -n "${STRATIFIED_SAMPLES}" ]]; then
  CMD+=(--stratified-samples "${STRATIFIED_SAMPLES}")
fi

"${CMD[@]}"
