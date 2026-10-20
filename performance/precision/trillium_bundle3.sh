#!/bin/bash
#SBATCH --job-name=orbitflows-b3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --time=12:00:00
#SBATCH --signal=B:TERM@600
#SBATCH --output=results/slurm_%j.out
#SBATCH --mail-type=FAIL
#
# One whole Trillium CPU node running the third precision bundle: 28 jobs x 6 threads, 12 h (2,304 core-hours).
# deep3: 48 and 64 layers with the learned frequency (amplitude series); ctrl3: angle and layout controls; resume3:
# the best second-bundle jobs, continued from their checkpoints with their saved damping and remaining LM time.
#
# Submit from a CPU login node, inside $SCRATCH/.../performance/precision:
#   sbatch -p debug --time=00:30:00 trillium_bundle3.sh     # optional 30-minute test on the debug partition
#   sbatch --dependency=singleton trillium_bundle3.sh  # one 12 h segment (add --time=24:00:00 for 24 h)
#   (resubmit the same line to continue; every job resumes from its checkpoint)
# Each segment resumes unfinished jobs from their checkpoints; the 68 h LM caps count across segments. Ten minutes
# before the time limit Slurm sends SIGTERM, the queue stops its workers, and they lose at most 25 LM steps.
#
# Python environment: the virtualenv behind the `ml_env` alias. Batch jobs on Trillium do not inherit the
# submitting shell's environment (and aliases are not available), so the path is set here; edit it if needed.
ENV_ACTIVATE="$HOME/.virtualenvs/ml_env/bin/activate"

cd "$SLURM_SUBMIT_DIR"
module load gcc arrow/24.0.0
source "$ENV_ACTIVATE"
python -c "import torch, orbitflows; print('torch', torch.__version__)" \
    || { echo "Python environment not found: edit ENV_ACTIVATE in trillium_node.sh"; exit 1; }

# pin workers to cores only when logical CPUs are physical cores (no SMT siblings to collide on)
PIN=""
[ "$(lscpu | awk -F: '/Thread\(s\) per core/ {gsub(/ /, "", $2); print $2}')" = "1" ] && PIN="--pin"
echo "$(date) $(hostname): $(nproc) CPUs, pinning: ${PIN:-off}"

exec python queue_runner.py \
    jobs/deep3_cpu.jsonl jobs/resume3_cpu.jsonl jobs/ctrl3_cpu.jsonl \
    --workers 28 --threads 6 $PIN
