#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=wan-lora-infer
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=GPU-shared
#SBATCH --gpus=l40s-48:1
#SBATCH --mail-type=ALL

# You can request at most 4 GPUs from one node in the GPU-shared partition.
# type: v100-16, v100-32, l40s-48, and h100-80
# -n n Number of cores requested in total.
# --ntasks-per-node=n # Request n cores be allocated per node.

# set -x

module load anaconda3
module load cuda

source /jet/home/mshah10/miniconda3/etc/profile.d/conda.sh
conda activate diffsynth

pwd
echo "JOB_NAME: ${SLURM_JOB_NAME}"
echo "LOG_DIR: $LOG_DIR"


python examples/wanvideo/model_training/validate_lora/Wan2.1-VACE-1.3B.py

# Construct log file names using SLURM environment variables
OUTPUT_FILE="${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out"
ERROR_FILE="${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err"

# Move the log files to LOG_DIR after job execution
echo "Moving log files to ${LOG_DIR}"
mv ${OUTPUT_FILE} ${LOG_DIR}/
mv ${ERROR_FILE} ${LOG_DIR}/