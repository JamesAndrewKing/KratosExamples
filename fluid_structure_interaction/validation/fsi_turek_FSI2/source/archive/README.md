# Archived FSI2 Control Workflows

These files preserve completed calibration, identification, and controller
experiments. They are not part of the active amplitude-frequency MPC runtime.
Run them only from a matching historical checkout because their imports and
environment variables reflect the software state used for those campaigns.

## Campaigns

| Files | Retained purpose |
| --- | --- |
| `run_parametric_actuation_library.py`, `submit_euler_parametric.slurm`, `collect_parametric_campaign_summary.py`, `repair_parametric_campaign_results.py` | Original harmonic amplitude/frequency campaign and Euler result repair. |
| `run_control_identification_campaign.py`, `submit_euler_control_identification.slurm` | Five-level ZOH scalar-control campaign `fsi_control_identification_t40`. |
| `run_control_parametric_manifold_campaign.py`, `submit_euler_control_parametric_manifold.slurm` | Smooth scalar-control parametric-manifold campaign `fsi_control_parametric_manifold_t40_u04`. |
| `run_control_authority_discovery_campaign.py`, `submit_euler_control_authority_discovery.slurm` | Minimal scalar authority study through `|u|=2`. |
| `run_control_mpc_identification_campaign.py`, `submit_euler_control_mpc_identification.slurm` | Scalar-control MPC identification campaign `fsi_control_mpc_identification_t40_u20`. |
| `run_control_fourier_envelope_campaign.py`, `submit_euler_fourier_envelope.slurm` | First quadrature Fourier-envelope campaign. |
| `run_control_fourier_envelope_multistate_campaign.py`, `submit_euler_fourier_envelope_multistate.slurm` | Provenance generator for `fsi_control_fourier_envelope_multistate_t40_u20`, covering early growth and mature limit-cycle states. |

## Controllers

| Files | Retained purpose |
| --- | --- |
| `fsi2_rom_mpc_controller.py`, `submit_euler_rom_mpc_closed_loop.slurm` | Deprecated scalar-input ROM MPC used in the earlier 20 s and 40 s closed-loop experiments. |

The active runtime remains in the parent directory:

- `MainKratos.py`
- `localized_cylinder_actuator_process.py`
- `fsi2_fourier_envelope_mpc_controller.py`
- `run_amplitude_frequency_mpc_experiment.py`
- `submit_euler_fourier_envelope_mpc.slurm`

