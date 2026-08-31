# Archived FSI2 Controller Runtime Experiments

This branch preserves the Kratos runtime used for the amplitude-frequency,
local-LQR, handoff, and fixed-deadline capture experiments. These files are
archived for reproducibility and are not the active FSI2 control workflow.

## Preserved experiments

- `run_amplitude_frequency_mpc_experiment.py`: receding-horizon control in
  amplitude and frequency. It achieved persistent 28--29% late-time RMS
  attenuation, but not equilibrium stabilization.
- `run_local_handoff_lqr_pilot.py`: guarded local stroboscopic LQR. It crossed
  its radius-0.01 guard at 6.33 s and subsequently returned to the passive
  limit cycle.
- `run_local_handoff_gain_pilot.py`: short higher-gain follow-up retained as
  experimental infrastructure, not as a validated controller.
- `run_mpc_handoff_experiment.py`: combined nonlinear MPC and local-controller
  runtime.
- `run_equilibrium_capture_mpc_experiment.py`: one 8 s fixed-deadline capture
  episode followed by zero-input coast. The minimum local radius was about
  0.823, so it did not approach the intended local neighborhood.

The matching Slurm scripts and controller implementations are retained with
the runners.

## Git provenance

The principal tracked development commits are:

- `833bed0`: amplitude-frequency MPC experiment;
- `c5f15f9`: guarded local-LQR pilot;
- `b6e839c`: MPC handoff experiment;
- `df439e4`: final pre-rollback tracked state.

The capture and gain-pilot files were uncommitted at rollback time and are
captured by this archive commit.

## Active reference

The active branch has been restored to the carrier-quadrature controller

\[
u(t)=a_c(t)\cos\theta(t)+a_s(t)\sin\theta(t),
\]

matching runs `fourier_envelope_mpc_t40_nopv_11182646` and
`fourier_envelope_mpc_t40_pv_11182648`.
