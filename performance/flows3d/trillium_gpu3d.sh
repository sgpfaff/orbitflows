#!/bin/bash
#SBATCH --job-name=flows3d-gpu
#SBATCH --account=ctb-ripperda
#SBATCH -p compute
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=00:30:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=results/slurm_gpu3d_%j.out
#
# One H100 (a quarter node, 24 cores) on Trillium's GPU cluster. Submit from the GPU login node (trillium-gpu), inside
# $SCRATCH/orbitflows/performance/flows3d:
#   sbatch trillium_gpu3d.sh bench3d.py --device cuda          # time an LM step on the H100
#   sbatch --time=24:00:00 trillium_gpu3d.sh run3d.py configs/<name>.json
# Jobs resume from their checkpoints, so resubmitting continues an interrupted run.
cd "$SLURM_SUBMIT_DIR"
mkdir -p results
module load gcc arrow/24.0.0 >/dev/null 2>&1
source "$HOME/.virtualenvs/ml_env/bin/activate"
export KMP_DUPLICATE_LIB_OK=TRUE NTHREADS=$SLURM_CPUS_PER_TASK
echo "$(date) $(hostname): $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
exec python -u "$@"
