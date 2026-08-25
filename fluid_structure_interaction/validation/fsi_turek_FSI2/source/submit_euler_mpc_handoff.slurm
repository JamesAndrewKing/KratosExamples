#!/bin/bash
# Single controlled MPC-to-local-LQR FSI2 experiment.

#SBATCH --job-name=kratos_fsi_mpc_lqr
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err

set -euo pipefail

mkdir -p slurm_logs
source "${SCRATCH}/kratos-venv/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

export KRATOS_FSI_MPC_HANDOFF_LABEL="${KRATOS_FSI_MPC_HANDOFF_LABEL:-fsi2_mpc_local_handoff_t40}"
export KRATOS_FSI_HANDOFF_CONTROLLER_FILE="${KRATOS_FSI_HANDOFF_CONTROLLER_FILE:-${SCRATCH}/Vortex-Shedding/data/fsi2_mpc_handoff_controller.json}"
export KRATOS_FSI_PASSIVE_REFERENCE_RUN="${KRATOS_FSI_PASSIVE_REFERENCE_RUN:-${PWD}/run_outputs/fsi_control_mpc_identification_t40_u20/runs/kick_then_passive_no_intervention}"
export KRATOS_FSI_MPC_HANDOFF_WRITE_PARAVIEW="${KRATOS_FSI_MPC_HANDOFF_WRITE_PARAVIEW:-0}"

if [[ ! -f "${KRATOS_FSI_HANDOFF_CONTROLLER_FILE}" ]]; then
    echo "Missing combined controller: ${KRATOS_FSI_HANDOFF_CONTROLLER_FILE}" >&2
    exit 2
fi
if [[ ! -d "${KRATOS_FSI_PASSIVE_REFERENCE_RUN}" ]]; then
    echo "Missing passive reference: ${KRATOS_FSI_PASSIVE_REFERENCE_RUN}" >&2
    exit 2
fi

python3 fsi2_mpc_handoff_controller.py \
    "${KRATOS_FSI_HANDOFF_CONTROLLER_FILE}"
python3 run_mpc_handoff_experiment.py \
    --validate-artifact "${KRATOS_FSI_HANDOFF_CONTROLLER_FILE}"
python3 run_mpc_handoff_experiment.py \
    --validate-reference "${KRATOS_FSI_PASSIVE_REFERENCE_RUN}"
python3 run_mpc_handoff_experiment.py
