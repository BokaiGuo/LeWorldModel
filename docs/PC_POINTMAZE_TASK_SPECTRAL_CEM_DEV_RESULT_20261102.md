# Common CEM with task-spectral initialization

On the four exposed development roots, both arms ran the same local bounded
CEM: three generations of 16 two-packet proposals, executed the selected
first packet, then used actual step-5 RGB and a matching three-generation
one-packet CEM to select and execute the tail. The spectral arm replaced
four initial proposals with bounded prefixes derived from the RGB task-region
Jacobian; the random arm used the same seeded Gaussian innovations without
those replacements. Both used the same nonlinear task-region score and
frozen model.

| Development root | Random CEM distance | Spectral-initialized CEM distance |
| --- | ---: | ---: |
| 1884 | 1.02661 | 1.67904 |
| 1885 | 0.99383 | 2.40032 |
| 1886 | 0.82040 | 0.95427 |
| 1887 | 2.01301 | 1.60678 |

The spectral arm was closer on one root and farther on three, measured by
offline position-only replay relative to archive frame 10. The two roots
1884 and 1885 had lower initial model objective under spectral initialization
but worse physical endpoint distance; this is an exploratory observation,
not evidence that the model objective alone caused the difference.

Each arm scored 48 two-packet plans and 48 one-packet tails, totaling 144
predicted block transitions and one real-feedback encoding. The spectral
arm additionally required two JVPs and an SVD. Its measured structure
calculation took approximately 0.039–0.173 seconds per root on this run;
first-use effects and different operations make these timings descriptive.
Equal prediction counts do **not** mean equal total resources. The random
arm was not allowed to spend an equivalent extra amount on search. This
small local CEM is not the official CEM implementation or a certified
strong-baseline comparison.

The task mask is a PointMaze-specific RGB heuristic. The checkpoint receives
fixed raw-zero proprio and only a single RGB context; normalization uses the
full archive and checkpoint training membership is unknown. These roots
were already exposed during development. Physical distance is not native
success; the newly selected chains' offline replay states have not been
cross-checked against directly captured live hidden states. These results do
not establish reconfiguration, independent generalization, an equal-budget
advantage, or P1 passage.

The report saves all CEM proposals, scores, elite updates, selected actions,
RGB hashes and cost counts; the NPZ saves the Jacobians, Gaussian innovations,
masks and images. Four targeted tests passed, including deterministic CEM
replay against every saved generation, bounds and image/distance arithmetic.
The tests do not independently rerun model scores or certify physical backend
equivalence for the new chains.

- Protocol: `docs/PC_POINTMAZE_TASK_SPECTRAL_CEM_DEV_PROTOCOL_20261102.md`
- Runner: `le-wm-main/experiments/a1_pointmaze_task_spectral_cem_dev_20261102.py`
- Result: `le-wm-main/results/a1_pointmaze_task_spectral_cem_dev_20261102_DEV01.json`
- SHA-256:
  `ff9a54ae9728d7ec4dbcda2e9bb37a6cfdcfde57829a18376316c4c8e6581ade`
- Tests: `le-wm-main/tests/test_a1_pointmaze_task_spectral_cem_dev_20261102.py`
