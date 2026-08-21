#!/bin/bash
# Full campaign:
#   sbatch submit_euler_control_parametric_manifold.slurm
#
# Smoke tests before the full campaign:
#   sbatch --array=0-3 --time=04:00:00 \
#     --export=ALL,KRATOS_FSI_PM_SMOKE=1,KRATOS_FSI_PM_LABEL=fsi_control_parametric_manifold_t40_u04_smoke \
#     submit_euler_control_parametric_manifold.slurm
#
# List cases:
#   python3 run_control_parametric_manifold_campaign.py --list-cases
#
# Audit after completion:
#   python3 run_control_parametric_manifold_campaign.py --audit run_outputs/fsi_control_parametric_manifold_t40_u04

#SBATCH --job-name=kratos_fsi_pm
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --array=0-60
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err

set -euo pipefail

mkdir -p slurm_logs

source "${SCRATCH}/kratos-venv/bin/activate"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

export KRATOS_FSI_PYTHONPATH="${KRATOS_FSI_PYTHONPATH:-}"
export KRATOS_FSI_LD_LIBRARY_PATH="${KRATOS_FSI_LD_LIBRARY_PATH:-}"

if [[ -n "${KRATOS_FSI_PYTHONPATH}" ]]; then
    export PYTHONPATH="${KRATOS_FSI_PYTHONPATH}:${PYTHONPATH:-}"
fi
if [[ -n "${KRATOS_FSI_LD_LIBRARY_PATH}" ]]; then
    export LD_LIBRARY_PATH="${KRATOS_FSI_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH:-}"
fi

export KRATOS_FSI_PM_LABEL="${KRATOS_FSI_PM_LABEL:-fsi_control_parametric_manifold_t40_u04}"
export KRATOS_FSI_PM_U_MAX="${KRATOS_FSI_PM_U_MAX:-0.4}"
export KRATOS_FSI_PM_END_TIME="${KRATOS_FSI_PM_END_TIME:-40.0}"
export KRATOS_FSI_PM_OUTPUT_INTERVAL="${KRATOS_FSI_PM_OUTPUT_INTERVAL:-0.01}"
export KRATOS_FSI_PM_SIGNAL_DT="${KRATOS_FSI_PM_SIGNAL_DT:-0.002}"
export KRATOS_FSI_PM_WRITE_PARAVIEW="${KRATOS_FSI_PM_WRITE_PARAVIEW:-0}"
export KRATOS_FSI_PM_CASE_INDEX="${SLURM_ARRAY_TASK_ID}"

python3 run_control_parametric_manifold_campaign.py
