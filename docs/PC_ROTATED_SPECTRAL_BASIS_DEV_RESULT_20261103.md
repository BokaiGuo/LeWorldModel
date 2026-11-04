# Rotated finite-horizon basis development result

The fixed protocol ran all eight previously exposed exact-unicycle goals and
four sampling seeds. Each arm evaluated 16 two-action mixed proposals plus
shared zero and greedy prefixes, then optimized one suffix action per prefix.
The spectral basis was genuinely non-axis-aligned: for example, its first
row was approximately `[0.710, -0.105, 0.697, 0]`.

| Goal | Raw isotropic | Full spectral isotropic | Spectral anisotropic |
| ---: | ---: | ---: | ---: |
| 0 | 0.03100 | 0.03100 | 0.03100 |
| 1 | 0.05652 | 0.05570 | 0.05439 |
| 2 | 0.02867 | 0.02607 | 0.02462 |
| 3 | 0.01914 | 0.03190 | 0.04277 |
| 4 | 0.02467 | 0.01259 | 0.00995 |
| 5 | 0.02207 | 0.00919 | 0.01370 |
| 6 | 0.02458 | 0.02318 | 0.01332 |
| 7 | 0.02547 | 0.02488 | 0.04314 |
| Mean | 0.02901 | 0.02681 | 0.02911 |

Entries are mean terminal position errors over four seeds; lower is better.
The full spectral arm had a lower root mean on six goals, tied on one and
was worse on one. Its isotropic proposal distribution is exactly the same
as the raw distribution under orthogonal rotation, although paired finite
samples differ. Consequently this difference cannot establish an advantage
of spectral information. The anisotropic proposal changes the distribution
and is worse on some goals; its panel mean is slightly worse than raw.

Each arm made 112 returned rollout calls and 20 returned Jacobian calls;
even the raw arm computed the same SVD for this controlled ablation. Counts
do not certify identical wall time or general compute. No learned model,
physical simulator, feedback, independent root, or formal gate was used.
All split/direct contexts matched. This is a candidate-distribution
diagnostic, not the full Atlas algorithm or evidence of non-greedy mechanism.

Protocol: `docs/PC_ROTATED_SPECTRAL_BASIS_DEV_PROTOCOL_20261103.md`.
Full actions, per-candidate costs, counters, timings and source hashes:
`le-wm-main/results/pc_rotated_spectral_basis_dev_20261103.json`.
Unit checks: 2 passed.
