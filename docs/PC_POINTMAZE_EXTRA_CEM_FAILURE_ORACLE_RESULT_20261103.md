# Physical candidate oracle for three development failures

The prewritten `PC_POINTMAZE_EXTRA_CEM_FAILURE_ORACLE_PROTOCOL_20261103.md`
replayed all 624 sealed candidates on the three outcome-selected development
roots. For each arm, the first row type below evaluates full two-packet
open-loop proposals; the second evaluates each tail proposal after that
arm's **actual selected first packet**. The selected tail's physical
position exactly matches the preceding sealed execution on all six arms.

| Root | Arm | Initial open-loop: chosen / best distance; number below 0.5 | Fixed-prefix tail: chosen / best distance; number below 0.5 |
| --- | --- | --- | --- |
| 1889 | random-extra | 1.247 / 0.139; 6 of 64 | 1.169 / 0.867; 0 of 48 |
| 1889 | spectral | 1.262 / 0.139; 4 of 48 | 1.165 / 0.863; 0 of 48 |
| 1893 | random-extra | 1.281 / 0.065; 7 of 64 | 1.320 / 0.470; 1 of 48 |
| 1893 | spectral | 1.236 / 0.237; 3 of 48 | 1.176 / 0.783; 0 of 48 |
| 1895 | random-extra | 0.469 / 0.060; 41 of 64 | 0.208 / 0.099; 44 of 48 |
| 1895 | spectral | 0.611 / 0.064; 16 of 48 | 1.218 / 1.106; 0 of 48 |

All six initial banks contain some successful open-loop plans, but the
model-selected open-loop plan is above 0.5 in five. Thus the observed
failure is not simply absence of a successful plan in those initial banks.
For 1889, neither selected prefix's saved 48-tail bank contains a successful
tail; for the spectral arms on 1893 and 1895 the same is true. On 1893
random-extra, one saved tail is below 0.5 but the model selects another.
These are properties of the saved banks and selected prefixes: **they do not
prove that no other tail or feedback planner can succeed from those states**.

Initial model-score versus physical-distance pairwise reversal counts are
1224/2016 and 705/1128 on 1889, 1211/2016 and 709/1128 on 1893, and
936/2016 and 663/1128 on 1895 (random-extra, spectral). The roots were
chosen after outcomes and candidates overlap within adaptive banks; the
pairs are not independent experiments. These physical-score mismatches
combine model prediction, latent metric, and state-estimation effects; they
cannot identify one as the unique cause without actual-RGB and interface
controls. The two stages have different policies: the initial open-loop
oracle cannot be substituted for a feedback controller's outcome.

The physically best candidates are **privileged post-outcome oracles**,
not actions available to the planner. This work does not establish the
non-greedy mechanism, fair strong-baseline performance, or P1 admission.
No neural inference was run for this replay.

Artifacts:

- JSON `le-wm-main/results/a0_pointmaze_extra_cem_failure_oracle_20261103_DEV01.json`,
  SHA-256 `dc78522fb837507e37fcf255c275e885210fb3e2fbb8dec1e2f804700f2ae8ef`.
- NPZ `le-wm-main/results/a0_pointmaze_extra_cem_failure_oracle_20261103_DEV01.npz`,
  SHA-256 `a40c87bf1369a74e1432d5679a3c19597a445c744526247cf53a5ee46fbdaca8`.
- Saved candidate binding, distance, ranking and selected-endpoint test:
  **1 passed**; it does not rerun the physical backend.
