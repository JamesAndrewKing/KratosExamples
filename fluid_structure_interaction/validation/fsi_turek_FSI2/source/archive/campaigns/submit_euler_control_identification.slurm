#!/bin/bash
# Submit from this directory on Euler with:
#   sbatch submit_euler_control_identification.slurm
#
# Check status:
#   squeue -u $USER
#
# After all array jobs finish, audit and collect summaries with:
#   python3 run_control_identification_campaign.py --audit run_outputs/fsi_control_identification_t40

#SBATCH --job-name=kratos_fsi_control
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --array=0-10
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err

set -euo pipefail

mkdir -p slurm_logs

# The tested Euler setup uses Kratos pip wheels in this virtual environment.
source "${SCRATCH}/kratos-venv/bin/activate"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# The pip wheels are found through the activated venv. Keep these empty by
# default so local source-build paths do not leak into the Euler job.
export KRATOS_FSI_PYTHONPATH="${KRATOS_FSI_PYTHONPATH:-}"
export KRATOS_FSI_LD_LIBRARY_PATH="${KRATOS_FSI_LD_LIBRARY_PATH:-}"

if [[ -n "${KRATOS_FSI_PYTHONPATH}" ]]; then
    export PYTHONPATH="${KRATOS_FSI_PYTHONPATH}:${PYTHONPATH:-}"
fi
if [[ -n "${KRATOS_FSI_LD_LIBRARY_PATH}" ]]; then
    export LD_LIBRARY_PATH="${KRATOS_FSI_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH:-}"
fi

export KRATOS_FSI_CONTROL_LABEL="${KRATOS_FSI_CONTROL_LABEL:-fsi_control_identification_t40}"
export KRATOS_FSI_CONTROL_MAX="${KRATOS_FSI_CONTROL_MAX:-0.12}"
export KRATOS_FSI_CONTROL_END_TIME="${KRATOS_FSI_CONTROL_END_TIME:-40.0}"
export KRATOS_FSI_CONTROL_OUTPUT_INTERVAL="${KRATOS_FSI_CONTROL_OUTPUT_INTERVAL:-0.01}"
export KRATOS_FSI_CONTROL_SIGNAL_DT="${KRATOS_FSI_CONTROL_SIGNAL_DT:-0.002}"
export KRATOS_FSI_CONTROL_WRITE_PARAVIEW="${KRATOS_FSI_CONTROL_WRITE_PARAVIEW:-0}"
export KRATOS_FSI_CONTROL_CASE_INDEX="${SLURM_ARRAY_TASK_ID}"

python3 run_control_identification_campaign.py
