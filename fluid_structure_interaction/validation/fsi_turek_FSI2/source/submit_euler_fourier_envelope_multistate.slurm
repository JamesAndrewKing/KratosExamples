#!/bin/bash
# Deterministic-startup multistate Fourier-envelope campaign for FSI2.
#
# Submit from this directory on Euler with:
#   sbatch submit_euler_fourier_envelope_multistate.slurm
#
# List cases:
#   python3 run_control_fourier_envelope_multistate_campaign.py --list-cases
#
# Audit after completion:
#   python3 run_control_fourier_envelope_multistate_campaign.py --audit run_outputs/fsi_control_fourier_envelope_multistate_t40_u20

#SBATCH --job-name=kratos_fsi_fe_ms
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --array=0-35
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

export KRATOS_FSI_FE_LABEL="${KRATOS_FSI_FE_LABEL:-fsi_control_fourier_envelope_multistate_t40_u20}"
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
export KRATOS_FSI_FE_MS_STARTUP_END_TIME="${KRATOS_FSI_FE_MS_STARTUP_END_TIME:-2.5}"
export KRATOS_FSI_FE_MS_NEAR_EQUILIBRIUM_PROBE_TIME="${KRATOS_FSI_FE_MS_NEAR_EQUILIBRIUM_PROBE_TIME:-3.0}"
export KRATOS_FSI_FE_MS_NEAR_EQUILIBRIUM_END_TIME="${KRATOS_FSI_FE_MS_NEAR_EQUILIBRIUM_END_TIME:-4.5}"
export KRATOS_FSI_FE_MS_EARLY_BURST_AMPLITUDE="${KRATOS_FSI_FE_MS_EARLY_BURST_AMPLITUDE:-0.1}"
export KRATOS_FSI_FE_MS_EARLY_BURST_DURATION="${KRATOS_FSI_FE_MS_EARLY_BURST_DURATION:-2.0}"
export KRATOS_FSI_FE_MS_EARLY_BURST_RAMP_DURATION="${KRATOS_FSI_FE_MS_EARLY_BURST_RAMP_DURATION:-0.5}"
export KRATOS_FSI_FE_MS_CASE_INDEX="${SLURM_ARRAY_TASK_ID}"

arguments=()
if [[ -n "${KRATOS_FSI_FE_MS_PILOT_CASE:-}" ]]; then
    arguments+=(--pilot-case "${KRATOS_FSI_FE_MS_PILOT_CASE}")
    arguments+=(--pilot-end-time "${KRATOS_FSI_FE_MS_PILOT_END_TIME:-6.0}")
    if [[ -n "${KRATOS_FSI_FE_MS_PILOT_DIRECTORY:-}" ]]; then
        arguments+=(--pilot-directory "${KRATOS_FSI_FE_MS_PILOT_DIRECTORY}")
    fi
    if [[ -n "${KRATOS_FSI_FE_MS_PASSIVE_REFERENCE:-}" ]]; then
        arguments+=(--passive-reference "${KRATOS_FSI_FE_MS_PASSIVE_REFERENCE}")
    fi
fi

python3 run_control_fourier_envelope_multistate_campaign.py "${arguments[@]}"
