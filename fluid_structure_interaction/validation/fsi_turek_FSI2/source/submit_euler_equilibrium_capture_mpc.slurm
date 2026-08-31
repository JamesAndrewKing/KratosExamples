#!/bin/bash
# One fixed-deadline amplitude-frequency MPC capture followed by zero-input coast.

#SBATCH --job-name=kratos_fsi_capture
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

mkdir -p slurm_logs
source "${SCRATCH}/kratos-venv/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

export KRATOS_FSI_CAPTURE_LABEL="${KRATOS_FSI_CAPTURE_LABEL:-fsi2_equilibrium_capture_mpc_t40}"
export KRATOS_FSI_HANDOFF_CONTROLLER_FILE="${KRATOS_FSI_HANDOFF_CONTROLLER_FILE:-${SCRATCH}/Vortex-Shedding/data/fsi2_mpc_handoff_controller.json}"
export KRATOS_FSI_CAPTURE_WRITE_PARAVIEW="${KRATOS_FSI_CAPTURE_WRITE_PARAVIEW:-0}"

python3 run_equilibrium_capture_mpc_experiment.py \
    --validate-artifact "${KRATOS_FSI_HANDOFF_CONTROLLER_FILE}"
python3 run_equilibrium_capture_mpc_experiment.py
