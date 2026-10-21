#!/bin/bash
#SBATCH --job-name=orbitflows-g4
#SBATCH --account=def-bovy
#SBATCH -p compute
#SBATCH --array=0-12
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=24:00:00
#SBATCH --signal=B:TERM@600
#SBATCH --output=results/slurm_g4_%A_%a.out
#
# The 13 third-bundle continuations (cont4g) on the Trillium GPU cluster: one H100 (a quarter node, 24 cores) per
# job, as a Slurm array; LM in float64 on CUDA, the energy and the damped solve on the job's CPU cores.
# Submit from the GPU login node (trillium-gpu), inside $SCRATCH/.../performance/precision:
#   sbatch trillium_gpu_cont4.sh            # 24 h limit; each job stops on its own when LM can no longer descend
# Resubmit the same line to continue interrupted jobs from their checkpoints (finished jobs have a DONE marker
# and exit at once). Ten minutes before the limit Slurm sends SIGTERM; a job loses at most 25 LM steps.
cd "$SLURM_SUBMIT_DIR"
module load gcc arrow/24.0.0 >/dev/null 2>&1
source "$HOME/.virtualenvs/ml_env/bin/activate"
export KMP_DUPLICATE_LIB_OK=TRUE NTHREADS=$SLURM_CPUS_PER_TASK
echo "$(date) $(hostname) task $SLURM_ARRAY_TASK_ID: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
exec python run.py jobs/cont4g_cpu.jsonl --line "$SLURM_ARRAY_TASK_ID"
