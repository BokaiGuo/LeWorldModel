# PointMaze action replay and RGB render timing

This development diagnostic replays the stored actions on roots 1880–1883
from each stored initial state. It uses the released visual XML, local
`mujoco-py` backend, the development-selected camera `(3,3)`, action and
velocity clipping, and five simulator ticks per stored action. At steps
0/5/10 it compares two renders from the same replay state against the
stored RGB:

1. `raw`: render immediately after the action's last `sim.step`, following
   the source environment's render path without an explicit forward.
2. `synced`: call `sim.forward` at that state and then render.

| Output | Pixel-identical frames | Non-identical frame locations |
| --- | ---: | --- |
| Raw post-step RGB | **12/12** | None |
| Forward-synchronized RGB | **4/12** | 98–129 pixels at each of eight nonzero steps |

The maximum position L2 difference between replayed and stored state at the
sampled steps was `1.215e-7`; the maximum velocity L2 difference was
`1.982e-7`. Each of the three previously observed root 1881–1883 step-5/10
differences matches the corresponding synchronized-render differing-pixel
count in the previous fixed-state probe. The root 1880 synchronized renders
also differ after the initial frame.

This directly demonstrates an **observation timing contract** for these
fixed trajectories: reading a stored post-action state, forwarding it and
rendering is not equivalent to the source post-step image. It is consistent
with derived rendering geometry lagging the integrated state until forward,
but this test does not independently inspect MuJoCo internal geometry or
establish the cause for every possible state. The raw result is exact for
these frames on this machine/backend only; it is not a proof of general
state, backend, or archive equivalence.

The camera was selected after seeing development images, all four roots
are previously exposed, and no world model, candidate selector, physical
success, or independent outcome was evaluated. This does not establish
RGB-only planning, finite-horizon spectral advantage, P1, or training
independence. For subsequent online experiments, preserve the environment's
post-step RGB before any `forward`/reset and keep simulator state outside
the planner. Recheck the contract with candidate-action replay and the
actual model preprocessing before reporting any feedback comparison.

- Runner: `le-wm-main/experiments/a0_pointmaze_replay_render_timing_20261102.py`
- Result: `le-wm-main/results/a0_pointmaze_replay_render_timing_20261102_DEV01.json`
- Result SHA-256:
  `24f1088dc0b12ab448757c8d3dc1ba6e039ba047ae8651b67143599c919f0892`
- Four targeted tests passed across camera and replay result contracts; they
  do not certify planning performance or held-out archive provenance.
