#!/bin/bash
# One guarded 10 s local-LQR pilot with a scalar feedback gain of three.

#SBATCH --job-name=kratos_fsi_local_gain3
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

mkdir -p slurm_logs
source "${SCRATCH}/kratos-venv/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

export KRATOS_FSI_LOCAL_GAIN_LABEL="${KRATOS_FSI_LOCAL_GAIN_LABEL:-fsi2_local_handoff_lqr_gain3_t10}"
export KRATOS_FSI_LOCAL_CONTROLLER_FILE="${KRATOS_FSI_LOCAL_CONTROLLER_FILE:-${SCRATCH}/Vortex-Shedding/data/fsi2_local_handoff_controller.json}"
export KRATOS_FSI_LOCAL_GAIN_PASSIVE_REFERENCE_RUN="${KRATOS_FSI_LOCAL_GAIN_PASSIVE_REFERENCE_RUN:-${PWD}/run_outputs/fsi2_local_handoff_lqr_pilot_t40/runs/passive_startup}"

python3 run_local_handoff_gain_pilot.py \
    --validate-artifact "${KRATOS_FSI_LOCAL_CONTROLLER_FILE}"
python3 run_local_handoff_gain_pilot.py \
    --validate-reference "${KRATOS_FSI_LOCAL_GAIN_PASSIVE_REFERENCE_RUN}"
python3 run_local_handoff_gain_pilot.py
