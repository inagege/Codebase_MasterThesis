#!/bin/bash
#SBATCH --job-name=extract_audio            # Job name
#SBATCH --output=logs/%x_%j.out           # Stdout log
#SBATCH --error=logs/%x_%j.err            # Stderr log
#SBATCH --time=05:00:00                   # Max runtime (hh:mm:ss)
#SBATCH --cpus-per-task=4                 # CPU cores
#SBATCH --mem=32G                        # RAM
#SBATCH --partition=normal                # Partition name

# Go to your project directory
cd /hkfs/work/workspace_haic/scratch/ulrat-masters/MasterThesis/Codebase_MasterThesis || exit 1

# (Optional but recommended) create logs directory if it doesn't exist
mkdir -p logs

# Run your Python script
pixi run python utils/extract_audio_only.py \
  --input-dir data/VoxCeleb2/dev/mp4 \
  --channels 1 \
  --workers "${SLURM_CPUS_PER_TASK:-4}"
