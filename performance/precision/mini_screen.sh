#!/bin/bash
#SBATCH --job-name=orbitflows-screen
#SBATCH --account=ctb-ripperda
#SBATCH -p compute
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=01:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --output=results/slurm_screen_%j.out
#
# Overparameterisation screen for the mini test bed: wider (4x16, 4x32), deeper (8x8, 16x8) and both (8x16) students
# of the 4x8 MW teacher (teachers/mini_4x8.pt, made by mini_node.sh), four seeds each, one thread per run, 45 min each.
# Its same-size 4x8 students all stalled at 1.1-1.6e-2. Submit from the GPU login node, inside .../performance/precision:
#   sbatch mini_screen.sh
ENV_ACTIVATE="$HOME/.virtualenvs/ml_env/bin/activate"
cd "$SLURM_SUBMIT_DIR"
module load gcc arrow/24.0.0 >/dev/null 2>&1
source "$ENV_ACTIVATE"
export MINI_RES="$SLURM_SUBMIT_DIR/results/mini"
echo "$(date) $(hostname): $(nproc) CPUs"
python mini_queue.py --teacher 4x8 --sizes 4x16,4x32,8x8,16x8,8x16 --seeds 4 \
    --student-cap 30000 --student-wall 2700 --workers 20 --threads 1 &
PID=$!
trap 'kill -TERM $PID; wait $PID' TERM
wait $PID
