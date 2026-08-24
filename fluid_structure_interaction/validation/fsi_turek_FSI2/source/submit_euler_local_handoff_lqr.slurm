#!/bin/bash
# Passive versus guarded local-LQR startup pilot. ParaView is disabled.

#SBATCH --job-name=kratos_fsi_local_lqr
#SBATCH --array=0-1
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err

set -euo pipefail

mkdir -p slurm_logs
source "${SCRATCH}/kratos-venv/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

export KRATOS_FSI_LOCAL_LQR_LABEL="${KRATOS_FSI_LOCAL_LQR_LABEL:-fsi2_local_handoff_lqr_pilot_t40}"
export KRATOS_FSI_LOCAL_CONTROLLER_FILE="${KRATOS_FSI_LOCAL_CONTROLLER_FILE:-${SCRATCH}/Vortex-Shedding/data/fsi2_local_handoff_controller.json}"
export KRATOS_FSI_LOCAL_LQR_WRITE_PARAVIEW=0

if [[ ! -f "${KRATOS_FSI_LOCAL_CONTROLLER_FILE}" ]]; then
    echo "Missing local-controller artifact: ${KRATOS_FSI_LOCAL_CONTROLLER_FILE}" >&2
    exit 2
fi

python3 fsi2_local_handoff_lqr_controller.py \
    "${KRATOS_FSI_LOCAL_CONTROLLER_FILE}"
python3 run_local_handoff_lqr_pilot.py \
    --validate-artifact "${KRATOS_FSI_LOCAL_CONTROLLER_FILE}"
python3 run_local_handoff_lqr_pilot.py \
    --case-index "${SLURM_ARRAY_TASK_ID}"
