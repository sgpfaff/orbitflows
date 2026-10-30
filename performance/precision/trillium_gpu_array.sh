#!/bin/bash
#SBATCH --job-name=orbitflows-gpu
#SBATCH --account=ctb-ripperda
#SBATCH -p compute

#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=08:00:00
#SBATCH --signal=B:TERM@600
#SBATCH --output=results/slurm_gpu_%A_%a.out
#
# Any job file on the Trillium GPU cluster, one H100 (a quarter node, 24 cores) per job, as a Slurm array.
# Submit from the GPU login node (trillium-gpu), inside $SCRATCH/.../performance/precision:
#   sbatch --array=0-1 trillium_gpu_array.sh jobs/fourier1_cpu.jsonl   # one task per line of the job file
# Resubmit to continue interrupted jobs from their checkpoints (finished jobs have a DONE marker and exit at once).
cd "$SLURM_SUBMIT_DIR"
module load gcc arrow/24.0.0 >/dev/null 2>&1
source "$HOME/.virtualenvs/ml_env/bin/activate"
export KMP_DUPLICATE_LIB_OK=TRUE NTHREADS=$SLURM_CPUS_PER_TASK
echo "$(date) $(hostname) task $SLURM_ARRAY_TASK_ID: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
exec python run.py "$1" --line "$SLURM_ARRAY_TASK_ID"
