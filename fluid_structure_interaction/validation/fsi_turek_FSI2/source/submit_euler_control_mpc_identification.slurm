#!/bin/bash
# Scalar-control MPC-identification campaign.
#
# Submit from this directory on Euler with:
#   sbatch submit_euler_control_mpc_identification.slurm
#
# List cases:
#   python3 run_control_mpc_identification_campaign.py --list-cases
#
# Audit after completion:
#   python3 run_control_mpc_identification_campaign.py --audit run_outputs/fsi_control_mpc_identification_t40_u20

#SBATCH --job-name=kratos_fsi_mpc
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --array=0-47
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

export KRATOS_FSI_MPC_LABEL="${KRATOS_FSI_MPC_LABEL:-fsi_control_mpc_identification_t40_u20}"
export KRATOS_FSI_MPC_U_MAX="${KRATOS_FSI_MPC_U_MAX:-2.0}"
export KRATOS_FSI_MPC_U_KICK="${KRATOS_FSI_MPC_U_KICK:-0.4}"
export KRATOS_FSI_MPC_END_TIME="${KRATOS_FSI_MPC_END_TIME:-40.0}"
export KRATOS_FSI_MPC_T_KICK_END="${KRATOS_FSI_MPC_T_KICK_END:-2.0}"
export KRATOS_FSI_MPC_T_INTERVENTION="${KRATOS_FSI_MPC_T_INTERVENTION:-20.0}"
export KRATOS_FSI_MPC_RAMP_DURATION="${KRATOS_FSI_MPC_RAMP_DURATION:-2.0}"
export KRATOS_FSI_MPC_OUTPUT_INTERVAL="${KRATOS_FSI_MPC_OUTPUT_INTERVAL:-0.01}"
export KRATOS_FSI_MPC_SIGNAL_DT="${KRATOS_FSI_MPC_SIGNAL_DT:-0.002}"
export KRATOS_FSI_MPC_WRITE_PARAVIEW="${KRATOS_FSI_MPC_WRITE_PARAVIEW:-0}"
export KRATOS_FSI_MPC_INCLUDE_FAST_RAMPS="${KRATOS_FSI_MPC_INCLUDE_FAST_RAMPS:-0}"
export KRATOS_FSI_MPC_INCLUDE_ZOH="${KRATOS_FSI_MPC_INCLUDE_ZOH:-1}"
export KRATOS_FSI_MPC_MAX_REASONABLE_TIP_DISPLACEMENT="${KRATOS_FSI_MPC_MAX_REASONABLE_TIP_DISPLACEMENT:-0.5}"
export KRATOS_FSI_MPC_CASE_INDEX="${SLURM_ARRAY_TASK_ID}"

python3 run_control_mpc_identification_campaign.py
