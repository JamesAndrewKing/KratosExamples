#!/bin/bash
# Two-case 60 s FSI2 amplitude-frequency MPC experiment.
# Array index 0 is passive; index 1 is closed loop.

#SBATCH --job-name=kratos_fsi_af_mpc
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

export KRATOS_FSI_PYTHONPATH="${KRATOS_FSI_PYTHONPATH:-}"
export KRATOS_FSI_LD_LIBRARY_PATH="${KRATOS_FSI_LD_LIBRARY_PATH:-}"
if [[ -n "${KRATOS_FSI_PYTHONPATH}" ]]; then
    export PYTHONPATH="${KRATOS_FSI_PYTHONPATH}:${PYTHONPATH:-}"
fi
if [[ -n "${KRATOS_FSI_LD_LIBRARY_PATH}" ]]; then
    export LD_LIBRARY_PATH="${KRATOS_FSI_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH:-}"
fi

export KRATOS_FSI_AF_MPC_LABEL="${KRATOS_FSI_AF_MPC_LABEL:-fsi2_amplitude_frequency_mpc_t60}"
export KRATOS_FSI_ROM_FILE="${KRATOS_FSI_ROM_FILE:-${SCRATCH}/Vortex-Shedding/data/fsi2_fourier_envelope_controller.json}"

if [[ ! -f "${KRATOS_FSI_ROM_FILE}" ]]; then
    echo "Missing KRATOS_FSI_ROM_FILE=${KRATOS_FSI_ROM_FILE}" >&2
    exit 2
fi

# Each array task checks the exact artifact it will use. No MPC tuning is
# duplicated here; all controller settings come from the JSON.
python3 fsi2_fourier_envelope_mpc_controller.py \
    "${KRATOS_FSI_ROM_FILE}" --benchmark-updates 2
python3 run_amplitude_frequency_mpc_experiment.py \
    --validate-artifact "${KRATOS_FSI_ROM_FILE}"
python3 run_amplitude_frequency_mpc_experiment.py \
    --case-index "${SLURM_ARRAY_TASK_ID}"
