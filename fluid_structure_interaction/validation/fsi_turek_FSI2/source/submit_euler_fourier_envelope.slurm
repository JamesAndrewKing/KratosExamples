#!/bin/bash
# Fourier-envelope controlled SSM-identification campaign for FSI2.
#
# Submit from this directory on Euler with:
#   sbatch submit_euler_fourier_envelope.slurm
#
# List cases:
#   python3 run_control_fourier_envelope_campaign.py --list-cases
#
# Audit after completion:
#   python3 run_control_fourier_envelope_campaign.py --audit run_outputs/fsi_control_fourier_envelope_t40_u20

#SBATCH --job-name=kratos_fsi_fe
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --array=0-42
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

export KRATOS_FSI_FE_LABEL="${KRATOS_FSI_FE_LABEL:-fsi_control_fourier_envelope_t40_u20}"
export KRATOS_FSI_FE_END_TIME="${KRATOS_FSI_FE_END_TIME:-40.0}"
export KRATOS_FSI_FE_OUTPUT_INTERVAL="${KRATOS_FSI_FE_OUTPUT_INTERVAL:-0.01}"
export KRATOS_FSI_FE_SIGNAL_DT="${KRATOS_FSI_FE_SIGNAL_DT:-0.002}"
export KRATOS_FSI_FE_WRITE_PARAVIEW="${KRATOS_FSI_FE_WRITE_PARAVIEW:-0}"
export KRATOS_FSI_FE_CARRIER_FREQUENCY_HZ="${KRATOS_FSI_FE_CARRIER_FREQUENCY_HZ:-1.90}"
export KRATOS_FSI_FE_ENVELOPE_BOUND="${KRATOS_FSI_FE_ENVELOPE_BOUND:-2.0}"
export KRATOS_FSI_FE_WARMUP_END_TIME="${KRATOS_FSI_FE_WARMUP_END_TIME:-20.0}"
export KRATOS_FSI_FE_INITIAL_KICK_AMPLITUDE="${KRATOS_FSI_FE_INITIAL_KICK_AMPLITUDE:-0.4}"
export KRATOS_FSI_FE_INITIAL_KICK_END_TIME="${KRATOS_FSI_FE_INITIAL_KICK_END_TIME:-2.0}"
export KRATOS_FSI_FE_INITIAL_KICK_RAMP_TIME="${KRATOS_FSI_FE_INITIAL_KICK_RAMP_TIME:-0.25}"
export KRATOS_FSI_FE_ACTIVATION_RAMP_DURATION="${KRATOS_FSI_FE_ACTIVATION_RAMP_DURATION:-2.0}"
export KRATOS_FSI_FE_RADIAL_RAMP_DURATION="${KRATOS_FSI_FE_RADIAL_RAMP_DURATION:-10.0}"
export KRATOS_FSI_FE_MAX_REASONABLE_TIP_DISPLACEMENT="${KRATOS_FSI_FE_MAX_REASONABLE_TIP_DISPLACEMENT:-0.75}"
export KRATOS_FSI_FE_CASE_INDEX="${SLURM_ARRAY_TASK_ID}"

python3 run_control_fourier_envelope_campaign.py
