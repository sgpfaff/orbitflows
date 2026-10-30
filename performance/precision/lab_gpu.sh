#!/bin/bash
#SBATCH --job-name=orbitflows-lab
#SBATCH --account=ctb-ripperda
#SBATCH -p debug
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=02:00:00
#SBATCH --output=results/slurm_lab_%j.out
# One optim_lab.py experiment on one H100 (debug partition, up to 2 h). Usage, from the GPU login node inside
# $SCRATCH/.../performance/precision:  sbatch lab_gpu.sh damping results/mwteacher_v2/mwt_geo_s0 --iters 400 --out ...
cd "$SLURM_SUBMIT_DIR"
module load gcc arrow/24.0.0 >/dev/null 2>&1
source "$HOME/.virtualenvs/ml_env/bin/activate"
export KMP_DUPLICATE_LIB_OK=TRUE NTHREADS=$SLURM_CPUS_PER_TASK
echo "$(date) $(hostname): $(nvidia-smi --query-gpu=name --format=csv,noheader); optim_lab.py $*"
exec python optim_lab.py "$@" --device cuda --chunk 96
