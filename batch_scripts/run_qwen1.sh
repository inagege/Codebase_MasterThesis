#!/bin/bash
#SBATCH --job-name=voxceleb_av   # Job name
#SBATCH --output=logs/%x_%j.out           # Stdout log
#SBATCH --error=logs/%x_%j.err            # Stderr log
#SBATCH --time=72:00:00                   # Max runtime (hh:mm:ss)
#SBATCH --gres=gpu:full:1                 # Request 1 GPU
#SBATCH --cpus-per-task=8                 # CPU cores
#SBATCH --mem=10G                        # RAM
#SBATCH --partition=normal                # Partition name

module load devel/cuda/12.9

# Go to your project directory
cd /hkfs/work/workspace_haic/scratch/ulrat-masters/MasterThesis/Codebase_MasterThesis || exit 1

# (Optional but recommended) create logs directory if it doesn't exist
mkdir -p logs

# Run your Python script
pixi run python benchmark_modalities.py \
  --dataset voxceleb \
  --modalities audio,video \
  --batch-size 8 \
  --out-path out/voxceleb/prediction_av_noise_.csv \
  --out-error-path out/voxceleb/errors_av_noise_.csv \
  --start-at-sample 296284
