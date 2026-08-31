#!/bin/bash
# Closed-loop Fourier-envelope ROM-MPC test for FSI2.
#
# Required artifact on Euler:
#   ${SCRATCH}/Vortex-Shedding/data/fsi2_fourier_envelope_controller.json
#
# Submit with:
#   sbatch submit_euler_fourier_envelope_mpc.slurm

#SBATCH --job-name=kratos_fsi_fe_mpc
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

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

export KRATOS_FSI_RUN_LABEL="${KRATOS_FSI_RUN_LABEL:-fourier_envelope_mpc_t40}"
export KRATOS_FSI_RUN_OUTPUT_DIRECTORY="${KRATOS_FSI_RUN_OUTPUT_DIRECTORY:-run_outputs/${KRATOS_FSI_RUN_LABEL}_${SLURM_JOB_ID:-manual}}"
export KRATOS_FSI_END_TIME="${KRATOS_FSI_END_TIME:-40.0}"
export KRATOS_FSI_OUTPUT_INTERVAL="${KRATOS_FSI_OUTPUT_INTERVAL:-0.01}"
export KRATOS_FSI_WRITE_PARAVIEW="${KRATOS_FSI_WRITE_PARAVIEW:-0}"
export KRATOS_FSI_CONTROLLER_TYPE="fourier_envelope_mpc"
export KRATOS_FSI_ROM_FILE="${KRATOS_FSI_ROM_FILE:-${SCRATCH}/Vortex-Shedding/data/fsi2_fourier_envelope_controller.json}"
export KRATOS_FSI_ROM_SHA256="${KRATOS_FSI_ROM_SHA256:-025cd090af008223b2e3e658056396746901dc304cb1d98384af2541f6269b96}"
export KRATOS_FSI_MPC_ACTIVATION_TIME="${KRATOS_FSI_MPC_ACTIVATION_TIME:-6.0}"
export KRATOS_FSI_MPC_INITIAL_KICK_VALUE="${KRATOS_FSI_MPC_INITIAL_KICK_VALUE:-0.4}"
export KRATOS_FSI_MPC_INITIAL_KICK_END_TIME="${KRATOS_FSI_MPC_INITIAL_KICK_END_TIME:-2.0}"

if [[ ! -f "${KRATOS_FSI_ROM_FILE}" ]]; then
    echo "Missing KRATOS_FSI_ROM_FILE=${KRATOS_FSI_ROM_FILE}" >&2
    exit 2
fi

actual_sha256="$(sha256sum "${KRATOS_FSI_ROM_FILE}" | awk '{print $1}')"
if [[ "${actual_sha256}" != "${KRATOS_FSI_ROM_SHA256}" ]]; then
    echo "Unexpected ROM artifact hash: ${actual_sha256}" >&2
    echo "Expected: ${KRATOS_FSI_ROM_SHA256}" >&2
    exit 3
fi

python3 fsi2_fourier_envelope_mpc_controller.py \
    "${KRATOS_FSI_ROM_FILE}" --benchmark-updates 1

python3 MainKratos.py
