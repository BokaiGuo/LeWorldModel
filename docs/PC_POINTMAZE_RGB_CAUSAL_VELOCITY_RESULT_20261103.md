# PointMaze: causal RGB velocity feedback, development result

The fixed protocol in `PC_POINTMAZE_RGB_CAUSAL_VELOCITY_PROTOCOL_20261103.md`
completed on host CUDA. Four already exposed roots and two previously selected
first packets per root were each executed under three feedback inputs: raw-zero
proprio, RGB position with zero velocity, and RGB position with causal
cp4-to-cp5 finite-difference velocity. This produced 24 physical ten-step
trajectories, not 24 independent roots. A single common first packet and common
CEM noise were used within each three-arm comparison.

The physical endpoint numbers below are **position-only backend replay
distances to the evaluation goal**, lower is better; they are not native
success, and the backend's full velocity state is not independently certified
against the RGB environment. The privileged evaluation state was read after
all three plans were chosen; it was not used in action selection.

| Root | First-packet source | Zero state | RGB position / zero velocity | RGB position / causal velocity |
| --- | --- | ---: | ---: | ---: |
| 1884 | random CEM | 1.0266 | 0.8711 | 0.8823 |
| 1884 | spectral CEM | 1.6790 | 1.3450 | 1.8845 |
| 1885 | random CEM | 0.9938 | 0.0954 | 0.0922 |
| 1885 | spectral CEM | 2.4003 | 1.8319 | 2.4790 |
| 1886 | random CEM | 0.8204 | 0.3911 | 0.1308 |
| 1886 | spectral CEM | 0.9543 | 0.4209 | 0.0885 |
| 1887 | random CEM | 2.0130 | 1.6678 | 1.6678 |
| 1887 | spectral CEM | 1.6068 | 1.6885 | 2.2023 |

RGB position with zero velocity is better than the zero-state arm in seven
of eight dependent comparisons and worse in one. Adding causal velocity
instead of zero velocity improves three, worsens four, and ties one. Compared
with zero-state feedback, causal velocity improves five and worsens three.
These are exploratory counts, not independent-sample inference. They do not
establish that velocity helps reliably or that spectral search beats random
search. In particular, the 1884, 1885 and 1887 spectral groups worsen when
causal velocity is added; this failure is retained, not attributed to theory
or a single code defect without further evidence.

Every group shares all six saved RGB frames across arms. The zero-state arm
reproduced the sealed source tail, score and final RGB hash. Each feedback
plan used three batches and 48 model predictions; the run requested five
intermediate observations per arm and did one feedback encoding. The saved
plan traces were replayed against the original CEM innovations and scores.
These records do not certify equal wall time or total full-pipeline cost.

The velocity is an average across 0.05 seconds, not an instantaneous
measurement. Legacy rendering may lag physical integration, and pixel
quantization and collisions can perturb it. The reported velocity errors
are versus a replay backend whose velocity equivalence is not certified;
they cannot establish the cause of the mixed results. Camera/scene geometry
is fixed environment metadata; its prior development choice and unknown
checkpoint training provenance limit generalization. Initial packets were
chosen by the earlier planner, so this is not an end-to-end RGB-state
planner, a reconfiguration mechanism test, or an independent P1 result.

Artifacts, SHA-256:

- JSON `le-wm-main/results/a1_pointmaze_rgb_causal_velocity_20261103_DEV01.json`:
  `4a041a1e3e1419a000f97407226817858529f874aa99ab4fdeba6ab5730c45a6`
- NPZ `le-wm-main/results/a1_pointmaze_rgb_causal_velocity_20261103_DEV01.npz`:
  `5652549233d3fb67e7148df146b1baac24143fc8e993e3012e10694198f5953f`
- Verification: `.venv/bin/pytest -q tests/test_a1_pointmaze_rgb_causal_velocity_20261103.py`
  from `le-wm-main`: **2 passed**. This verifies saved bindings and arithmetic,
  not an independent rerun of model predictions or physical outcomes.
