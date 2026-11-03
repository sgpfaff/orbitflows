#!/bin/bash
#SBATCH --job-name=orbitflows-mini
#SBATCH --account=ctb-ripperda
#SBATCH -p compute
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=01:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --output=results/slurm_mini_%j.out
#
# The mini teacher-student pilot (mini_lab.py, mini_queue.py) on a quarter of a Trillium GPU node (its 24 cores;
# the networks are too small for the H100 to help): four sizes, one MW teacher each, then six same-size students per
# teacher, all in parallel at one thread per run. Kept short so it schedules quickly.
# Submit from the GPU login node (trillium-gpu), inside $SCRATCH/.../performance/precision:
#   sbatch mini_node.sh
ENV_ACTIVATE="$HOME/.virtualenvs/ml_env/bin/activate"
cd "$SLURM_SUBMIT_DIR"
module load gcc arrow/24.0.0 >/dev/null 2>&1
source "$ENV_ACTIVATE"
export MINI_RES="$SLURM_SUBMIT_DIR/results/mini"
echo "$(date) $(hostname): $(nproc) CPUs"
python mini_queue.py --sizes 4x8,4x16,8x16,8x32 --seeds 6 --teacher-cap 8000 --teacher-wall 900 \
    --student-cap 30000 --student-wall 1800 --workers 24 --threads 1 &
PID=$!
trap 'kill -TERM $PID; wait $PID' TERM
wait $PID
