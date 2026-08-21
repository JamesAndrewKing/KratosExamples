#!/bin/bash
# Closed-loop ROM-MPC test for the Turek-Hron FSI2 example.
#
# Before submitting, make sure the controller JSON exists on Euler. Default:
#   ${SCRATCH}/Vortex-Shedding/data/fsi2_mpc_controller.json
#
# Submit from this directory on Euler with:
#   sbatch submit_euler_rom_mpc_closed_loop.slurm

#SBATCH --job-name=kratos_fsi_rom_mpc
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

export KRATOS_FSI_RUN_LABEL="${KRATOS_FSI_RUN_LABEL:-rom_mpc_kick_lc_closed_loop_t40_u20_du005}"
DEFAULT_RUN_OUTPUT_DIRECTORY="run_outputs/${KRATOS_FSI_RUN_LABEL}_${SLURM_JOB_ID:-manual}"
export KRATOS_FSI_RUN_OUTPUT_DIRECTORY="${KRATOS_FSI_RUN_OUTPUT_DIRECTORY:-${DEFAULT_RUN_OUTPUT_DIRECTORY}}"
export KRATOS_FSI_END_TIME="${KRATOS_FSI_END_TIME:-40.0}"
export KRATOS_FSI_OUTPUT_INTERVAL="${KRATOS_FSI_OUTPUT_INTERVAL:-0.01}"
export KRATOS_FSI_WRITE_PARAVIEW="${KRATOS_FSI_WRITE_PARAVIEW:-0}"
export KRATOS_FSI_CONTROLLER_TYPE="${KRATOS_FSI_CONTROLLER_TYPE:-rom_mpc}"
export KRATOS_FSI_ROM_FILE="${KRATOS_FSI_ROM_FILE:-${SCRATCH}/Vortex-Shedding/data/fsi2_mpc_controller.json}"
export KRATOS_FSI_MPC_ACTIVATION_TIME="${KRATOS_FSI_MPC_ACTIVATION_TIME:-20.0}"
export KRATOS_FSI_MPC_INITIAL_KICK_VALUE="${KRATOS_FSI_MPC_INITIAL_KICK_VALUE:-0.4}"
export KRATOS_FSI_MPC_INITIAL_KICK_END_TIME="${KRATOS_FSI_MPC_INITIAL_KICK_END_TIME:-2.0}"
export KRATOS_FSI_MPC_CONTROL_INTERVAL="${KRATOS_FSI_MPC_CONTROL_INTERVAL:-0.05}"
export KRATOS_FSI_MPC_PREDICTION_HORIZON="${KRATOS_FSI_MPC_PREDICTION_HORIZON:-1.0}"
export KRATOS_FSI_MPC_CONTROL_BOUND="${KRATOS_FSI_MPC_CONTROL_BOUND:-2.0}"
export KRATOS_FSI_MPC_MAX_CONTROL_INCREMENT="${KRATOS_FSI_MPC_MAX_CONTROL_INCREMENT:-0.05}"
export KRATOS_FSI_MPC_MOVE_BLOCKS="${KRATOS_FSI_MPC_MOVE_BLOCKS:-20}"
export KRATOS_FSI_MPC_OPTIMIZER_ITERATIONS="${KRATOS_FSI_MPC_OPTIMIZER_ITERATIONS:-8}"

if [[ ! -f "${KRATOS_FSI_ROM_FILE}" ]]; then
    echo "Missing KRATOS_FSI_ROM_FILE=${KRATOS_FSI_ROM_FILE}" >&2
    exit 2
fi

python3 MainKratos.py
