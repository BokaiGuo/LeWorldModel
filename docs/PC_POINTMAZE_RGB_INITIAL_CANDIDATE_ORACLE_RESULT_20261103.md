# RGB-position initial CEM candidate decomposition

The prewritten `PC_POINTMAZE_RGB_INITIAL_CANDIDATE_ORACLE_PROTOCOL_20261103.md`
replayed all 384 sealed initial proposals (four exposed roots, two methods,
48 proposals each) with the position backend. Each proposal was executed as
two complete five-action packets **without cp5 replanning**. CEM proposal
traces and selected action identity were reconstructed from sealed noise and
scores before using physical outcomes. These are post-outcome physical
oracles; no physical state or outcome entered the original selector.

| Root | Candidate bank | Model-selected distance | Physical oracle distance | Candidates below 0.5 | Reversed / comparable score pairs |
| --- | --- | ---: | ---: | ---: | ---: |
| 1884 | random | 0.3215 | 0.0241 | 30/48 | 247/1128 |
| 1884 | spectral | 0.3029 | 0.0419 | 17/48 | 216/1128 |
| 1885 | random | 0.2819 | 0.1991 | 9/48 | 784/1128 |
| 1885 | spectral | 0.2042 | 0.1127 | 8/48 | 815/1128 |
| 1886 | random | 0.2522 | 0.1533 | 37/48 | 201/1128 |
| 1886 | spectral | 0.0463 | 0.0463 | 39/48 | 196/1128 |
| 1887 | random | 0.7038 | 0.2124 | 5/48 | 549/1128 |
| 1887 | spectral | 0.6755 | 0.1717 | 10/48 | 609/1128 |

The threshold matches the checked-in wrapper's strict position criterion,
but these are position-backend replay labels, not a new native RGB execution.
The pairwise counts use exact score/outcome ties and are dependent candidate
pairs, not independent samples. The adaptive banks can overlap and should
not be interpreted as equally sampled distributions.

On root 1887, both banks contain candidates below the threshold even though
each model-selected two-packet plan remains above it. Of the four spectral
proposals injected in the first generation, three have physical distances
below 0.5 (0.4938, 0.3416 and 0.2906); none was the final model-selected
plan. This identifies **candidate availability plus selection mismatch**
for these fixed banks; it does not prove that selecting one of these
after observing outcomes would be deployable, that it is non-greedy, or
that spectral initialization causes an advantage. The oracle compares
complete open-loop plans, whereas earlier feedback results replanned at cp5.
The spectral bank also paid extra JVP/SVD cost, so this is not an equal-total-
budget method comparison or P1 evidence.

Artifacts:

- JSON `le-wm-main/results/a0_pointmaze_rgb_initial_candidate_oracle_20261103_DEV01.json`:
  SHA-256 `44f3b133ca6332b9fcb831c6accd50052c239e38a9ef1c5e7ce4657debda2e76`.
- NPZ `le-wm-main/results/a0_pointmaze_rgb_initial_candidate_oracle_20261103_DEV01.npz`:
  SHA-256 `4e21e4bf62d82ce86544c6cb94b0532ae9c23876a06feadc3a0cced5c6a073d3`.
- `.venv/bin/pytest -q tests/test_a0_pointmaze_rgb_initial_candidate_oracle_20261103.py`:
  **2 passed**; tests verify saved trace identity and arithmetic, not an
  independent physics rerun.
