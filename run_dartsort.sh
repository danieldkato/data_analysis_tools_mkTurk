#!/bin/bash
#SBATCH --job-name=dartsort
#SBATCH --account=issa
#SBATCH --chdir=/home/yy3658/NeuralWaveform
#SBATCH --cpus-per-task=32
#SBATCH --mem=128gb
#SBATCH --partition=issa
#SBATCH --gres=gpu:8
#SBATCH --nodelist=ax09
#SBATCH --output=/home/dk2643/dartsort_%A.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /share/issa/users/yy3658/shared_env/data_processing

echo "=== Resource Validation ==="
echo "Node: $(hostname)"
echo "Monkey: $MONKEY, Date: $DATE"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Validate GPUs
N_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())")
echo "PyTorch visible GPUs: $N_GPUS"

if [ "$N_GPUS" -eq 0 ]; then
    echo "ERROR: No GPUs detected. Exiting."
    exit 1
fi

nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader
echo "=========================="

LOCAL_STAGING="/tmp/dartsort_${SLURM_JOB_ID}"
cleanup() {
    echo "Cleaning up local staging: $LOCAL_STAGING"
    rm -rf "$LOCAL_STAGING"
}
trap cleanup EXIT

python "$SCRIPT_DIR/run_dartsort.py" --monkey "$MONKEY" --date "$DATE"
