# PointMaze A0 contract side audit — 2026-11-03

## Finding

The existing privileged physical CEM is a real, development-only
simulation-based receding-horizon execution, but it is not the same contract
as the older `perfect_dynamics_mpc` beam control and is not certified perfect
dynamics or an equal-total-resource RGB baseline.

- The older `perfect_dynamics_mpc_action` in
  `le-wm-main/experiments/run_live_pointmaze_planning.py` searches a bounded
  beam, scores distance to an active waypoint, executes the first archive
  action and replans. Its strict evaluation uses a root-fixed bank and
  global-goal terminal evaluation; see
  `docs/POINTMAZE_STRICT_ORACLE_PROTOCOL_AUDIT_20261021.md`. Its local search
  target and terminal evaluator are not necessarily identical.
- The newer physical CEM starts from recorded frame 0, targets the privileged
  position at archive frame 10, searches two constant-action packets, executes
  the first five archive actions, and searches a remaining one-packet horizon
  from cp5. The cost is endpoint position distance, not RGB feature MSE and
  not native task success. The visual comparator uses its own latent visual
  goal metric. See `docs/PC_POINTMAZE_PHYSICAL_CEM_MPC_PROTOCOL_20261103.md`
  and `docs/PC_POINTMAZE_PHYSICAL_CEM_MPC_RESULT_20261103.md`.
- One packet is five repeated archive-level actions; each action clips
  velocity and advances five MuJoCo ticks. Thus cp5 and cp10 mean 25 and
  50 simulator ticks, respectively. One older fixed-bank replan instead
  optimized to cp15 while evaluating at cp10; it is not the same clock
  contract (`docs/PC_POINTMAZE_PACKET_REPLAN_ORACLE_RESULT_20261102.md`).
- Per root, physical CEM scores 3 × 16 ten-action candidates and 3 × 16
  five-action candidates: 96 candidate rollouts and 720 high-level simulated
  steps, excluding execution and verification. Matching proposal counts or
  Gaussian innovations does not equate physical simulation to model inference,
  wall time, information access, or total compute.

## New CPU diagnostic

The source result contains four selected CEM action chains, roots 1884–1887.
The standalone side runner reconstructs each chain from its saved first
packet and replanned tail, and replays identical actions from identical saved
four-component initial states in (1) the audited MuJoCo 2.2 position backend
and (2) a fresh, non-rendering `mujoco-py` 2.1.2 simulation using the native
RGB bridge's XML and stepping loop. At cp5 and cp10, all four position and
velocity coordinates matched exactly at float64 output precision for all
four chains; the re-run 2.2 endpoints also exactly matched the source's saved
states.

This is a cross-backend simulator replay, **not a measurement of hidden state
from the earlier RGB bridge execution**: the side worker reproduces its
stepping but does not render or modify the visual target site. It provides no
candidate-wide guarantee, native task success, held-out generalization,
optimality, or fairness claim. The saved outcomes were already exposed.

Source SHA-256:
`7fed2f87bfce3119a28a84ee895a408cdebe3f052a69850919e43a800332cd51`.
Side runner SHA-256:
`18d55aac78e0d0da3126162e5c0343914c16f615e5028f4d7393a49bfb951bf3`.
Side result SHA-256:
`8233985d36513c88b67536739fb49e65fab3e98bb6fc2e5ba4eeef949b143d1c`.

Command, run from `le-wm-main`:

```sh
env CUDA_VISIBLE_DEVICES='' PYTHONPATH=. .venv/bin/python experiments/a0_pointmaze_cpu_crossbackend_contract_20261103_side.py
```

The runner refuses to overwrite its result. A more discriminating next check
would compare all 96 candidate trajectories per root across backends,
including candidate ranking and tolerance-sensitive near ties, with native
success and full resource accounting under a separately frozen contract.
