#!/bin/bash
#SBATCH --job-name=orbitflows-bezier
#SBATCH --account=ctb-ripperda
#SBATCH -p compute
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=01:00:00
#SBATCH --array=0-1
#SBATCH --output=results/slurm_bezier_%A_%a.out
#
# Curved-path test on the 16x64 MW-teacher test bed (landscape_bezier.py): a quadratic Bezier curve from each of the
# two best stalled students (task 0: std_s1 = A, task 1: geo_s0 = B; aligned min+shift gauge) to the teacher, its
# control point fitted by LM on the H100. Needs results/landscape/aligned.npz (copied from the laptop) and
# results/mwteacher_v2/mwt_std_s1 (the base job). Submit from the GPU login node, inside .../performance/precision:
#   sbatch bezier_gpu.sh
ENV_ACTIVATE="$HOME/.virtualenvs/ml_env/bin/activate"
cd "$SLURM_SUBMIT_DIR"
module load gcc arrow/24.0.0 >/dev/null 2>&1
source "$ENV_ACTIVATE"
export KMP_DUPLICATE_LIB_OK=TRUE NTHREADS=$SLURM_CPUS_PER_TASK
export LANDSCAPE_RES="$SLURM_SUBMIT_DIR/results/landscape" LANDSCAPE_ORIG="$SLURM_SUBMIT_DIR/results/mwteacher_v2"
STUDENTS=(std_s1 geo_s0)
echo "$(date) $(hostname) task $SLURM_ARRAY_TASK_ID: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
exec python landscape_bezier.py --student "${STUDENTS[$SLURM_ARRAY_TASK_ID]}" --k 16 --iters 400 --wall 2700 --device cuda --chunk 96
