# v22 PointMaze actual finite-difference probes: physical diagnostic

The response arm's sealed third decision supplied two nominal batches, each
with 21 actual five-step probes. All 42 branches were replayed from the
original setup and action history. Each branch's RGB at the decision boundary
matched the sealed frame; native prefix states matched the prior native
physical replay. The independent data audit recomputed action bindings,
causal RGB estimates, stored physical states and reported metrics.

| nominal | model/native absolute error | model/native relative error | response cosine | score/native-distance pair agreement |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.163875 | 0.300427 | 0.958236 | 190/210 (90.48%) |
| 1 | 0.301005 | 0.634986 | 0.776532 | 69/210 (32.86%) |

For the same batches, RGB-estimated versus native response relative errors
were 0.359801 and 0.308164. The model score and physical distance are
different quantities; low ordering agreement identifies a task-scoring
risk in this local bank, not a proven cause of the overall outcome. The
causal estimator, projection, model and physical replay may all contribute.
The data audit recomputes metrics from saved inputs but does not independently
rerender or replay the physics backend.

## Post-hoc score decomposition

A separately recorded privileged diagnostic anchors the model's nominal
predicted position to the physical nominal before evaluating the saved
perturbations. This is not an admissible planner input: it uses the physical
successor. Score/native-distance ranking agreement changes from 190/210 to
196/210 for nominal 0, and from 69/210 to 136/210 for nominal 1. The remaining
disagreement prevents explaining nominal 1 solely by a constant offset.
The predicted and goal coordinates share an affine projection gauge, so
their large absolute offset from simulator coordinates cannot itself be
called a model error; their difference is what the implemented score uses.
On these batches the saved scores match that projected goal-distance
calculation to at most 1.03e-6.

A second post-hoc fixed-bank hypothesis uses only the current RGB-estimated
position, RGB goal and the saved zero-action model prediction to anchor
relative predicted displacement. Its ranking agreements are 24/210 and
65/210, versus the original score's 190/210 and 69/210. It is not promoted:
the zero-action future can move, and this construction ignores passive
drift. This is not an end-to-end planning comparison.

The result covers one exposed root, one feedback boundary and ±0.05 probes;
it does not certify the full candidate scale, PC1, or independent
generalization. No physical evaluation informed the planner or changed a
previously selected action.

Artifacts: `le-wm-main/results/pc_v22_pointmaze_actual_fd_physical_dev_20261104_codex_{manifest.json,arrays.npz,result.json,audit.json}`.
The post-hoc decomposition is stored in the corresponding
`_score_attribution.json`; the RGB-anchor hypothesis is in
`_rgb_anchor_score_diagnostic.json`.
