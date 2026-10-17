#!/bin/bash
#SBATCH --job-name=orbitflows-node
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --time=24:00:00
#SBATCH --signal=B:TERM@600
#SBATCH --output=results/slurm_%j.out
#SBATCH --mail-type=FAIL
#
# One whole Trillium CPU node running the precision bundle: 45 jobs x 4 threads, pinned two jobs per 8-core L3
# cache (Trillium only schedules whole nodes, at most 24 h each). Jobs are listed in priority order.
#
# Submit from a CPU login node, inside $SCRATCH/.../performance/precision:
#   sbatch -p debug --time=00:30:00 trillium_node.sh     # optional 30-minute test on the debug partition
#   sbatch trillium_node.sh                              # first 24 h
#   sbatch --dependency=singleton trillium_node.sh       # twice more: each waits for the previous one to end
# Each segment resumes unfinished jobs from their checkpoints; the 68 h LM caps count across segments. Ten minutes
# before the time limit Slurm sends SIGTERM, the queue stops its workers, and they lose at most 25 LM steps.
#
# Python environment: by default this runs your `ml_env` alias from ~/.bashrc. If that fails in a batch job,
# submit with ENV_ACTIVATE=/path/to/venv/bin/activate sbatch trillium_node.sh

cd "$SLURM_SUBMIT_DIR"
module load gcc arrow/24.0.0
if [ -n "$ENV_ACTIVATE" ]; then
    source "$ENV_ACTIVATE"
else
    shopt -s expand_aliases
    source ~/.bashrc
    ml_env
fi
python -c "import torch, orbitflows; print('torch', torch.__version__)" \
    || { echo "Python environment not found: submit with ENV_ACTIVATE=/path/to/venv/bin/activate"; exit 1; }

# pin workers to cores only when logical CPUs are physical cores (no SMT siblings to collide on)
PIN=""
[ "$(lscpu | awk -F: '/Thread\(s\) per core/ {gsub(/ /, "", $2); print $2}')" = "1" ] && PIN="--pin"
echo "$(date) $(hostname): $(nproc) CPUs, pinning: ${PIN:-off}"

exec python queue_runner.py \
    jobs/anchor_cpu.jsonl jobs/anchor_geo_cpu.jsonl jobs/toygeo_cpu.jsonl jobs/mwgeo_cpu.jsonl \
    jobs/mwteacher_cpu.jsonl jobs/ladder2_cpu.jsonl jobs/capprobe_cpu.jsonl \
    --workers 45 --threads 4 $PIN
