#!/bin/bash
# Closed-loop test of the standalone phase-periodic envelope feedback law.

#SBATCH --job-name=kratos_fsi_periodic_fb
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

mkdir -p slurm_logs
source "${SCRATCH}/kratos-venv/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

export KRATOS_FSI_RUN_LABEL="${KRATOS_FSI_RUN_LABEL:-periodic_envelope_feedback_t40}"
export KRATOS_FSI_RUN_OUTPUT_DIRECTORY="${KRATOS_FSI_RUN_OUTPUT_DIRECTORY:-run_outputs/${KRATOS_FSI_RUN_LABEL}_${SLURM_JOB_ID:-manual}}"
export KRATOS_FSI_END_TIME="${KRATOS_FSI_END_TIME:-40.0}"
export KRATOS_FSI_OUTPUT_INTERVAL="${KRATOS_FSI_OUTPUT_INTERVAL:-0.01}"
export KRATOS_FSI_WRITE_PARAVIEW="${KRATOS_FSI_WRITE_PARAVIEW:-0}"
export KRATOS_FSI_CONTROLLER_TYPE="periodic_envelope_feedback"
export KRATOS_FSI_ROM_FILE="${KRATOS_FSI_ROM_FILE:-${SCRATCH}/Vortex-Shedding/data/fsi2_periodic_envelope_feedback_controller.json}"
export KRATOS_FSI_ROM_SHA256="${KRATOS_FSI_ROM_SHA256:-a531094f50bb89afecdd0a08cd5cd5efaed081f8c6930fe51ae3b7928ccd4aab}"
export KRATOS_FSI_MPC_ACTIVATION_TIME="${KRATOS_FSI_MPC_ACTIVATION_TIME:-6.0}"
export KRATOS_FSI_MPC_INITIAL_KICK_VALUE="${KRATOS_FSI_MPC_INITIAL_KICK_VALUE:-0.4}"
export KRATOS_FSI_MPC_INITIAL_KICK_END_TIME="${KRATOS_FSI_MPC_INITIAL_KICK_END_TIME:-2.0}"

if [[ ! -f "${KRATOS_FSI_ROM_FILE}" ]]; then
    echo "Missing controller artifact: ${KRATOS_FSI_ROM_FILE}" >&2
    exit 2
fi

actual_sha256="$(sha256sum "${KRATOS_FSI_ROM_FILE}" | awk '{print $1}')"
if [[ "${actual_sha256}" != "${KRATOS_FSI_ROM_SHA256}" ]]; then
    echo "Unexpected controller artifact hash: ${actual_sha256}" >&2
    echo "Expected: ${KRATOS_FSI_ROM_SHA256}" >&2
    exit 3
fi

python3 fsi2_periodic_envelope_feedback_controller.py "${KRATOS_FSI_ROM_FILE}"
python3 MainKratos.py
