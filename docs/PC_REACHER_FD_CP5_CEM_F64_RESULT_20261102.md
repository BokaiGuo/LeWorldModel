# FD cp5 CEM on two exposed Reacher roots

The float64 bridge was frozen in plan
`c4b33e3b67242173617d73bc0f577d8f2f9b2ac3f95a170970180d51dca4444e`.
Host CUDA, the cooperative GPU lock, fixed warmup and exact original prefix
action hashes were used. The saved consumer verified four of four branches:
for each, five selected/parameter-zero prefix steps, five extra parameter-zero
control steps from the restored original root, and ten independently optimized
suffix steps from the actual successor context were returned without stop.
Every suffix used the same CEM configuration, 32 candidate rollouts, and the
same per-root random seed; full upstream and model costs are not metered.

All distances below are wrapped joint distances; smaller is better. The
parentheses show selected minus parameter-zero:

| Development root | Initial | Selected cp5 | Parameter-zero cp5 | Selected final | Parameter-zero final | Final difference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 20263160 | 1.003500 | 0.987020 | 1.010031 | 0.541186 | 0.546408 | -0.005221 |
| 20263161 | 1.182600 | 1.209842 | 1.245444 | 0.588218 | 0.598310 | -0.010092 |

Selected starts the suffix closer than parameter-zero on both roots. Root
20263160 also improves distance during its selected prefix relative to initial;
root 20263161 worsens relative to initial but less than parameter-zero.
The difference between the two suffix-phase changes,
`(selected final - selected cp5) - (zero final - zero cp5)`, is **+0.017790**
and **+0.025509**, respectively: selected's suffix reduces distance less
from its own cp5 baseline, despite its lower absolute final distance. This is
a descriptive contrast between separately optimized actions, not a causal
estimate of controllability. Both arms used the same CEM seed on each root;
the selected and parameter-zero suffix witnesses are different on root
20263160 but exactly the same on root 20263161. A shared seed does not imply
identical search candidates after adaptive distribution updates. The journal
does not contain the complete random stream or candidate bank.

This experiment shows that the dtype repair recovered an executable feedback
comparison and that selected had a slightly lower absolute final distance on
these exposed roots. It does **not** demonstrate a non-immediate-progress
mechanism, better downstream response, independent generalization, a win over
greedy/CEM strong baselines, equal total compute, or P1 admission. FD score
gradients are not the original Jacobian/SVD spectral program. The journal
records completion and stop flags; do not infer a native success rate from
distance alone.

After opening physical outcomes, a separately defined same-bank immediate-Q
greedy rule identified index 0 on both roots. Its sealed parameter and raw
action are exactly the existing parameter-zero arm, so this is an alias of
the same two physical outcomes, not new predeclared or independent greedy
trials. It is not a full-horizon greedy optimizer or a strong baseline.
Details are in `PC_REACHER_P1_FD_SAMEBANK_GREEDY_ALIAS_20261102_CODEXEXCLUSIVE.md`.

Run journal SHA-256:
`587c88657831babbe270eeb250179688cff57d6cbe17801228a6cab6e66be3bf`.
Saved consumer:
`le-wm-main/results/reacher_fd_cp5_host_f64_SAVED_20261102_codexexclusive_01.json`.
Its SHA-256 is
`b8198c0067f350f028ff64cc79a53c3fec6eb4db8bb2f351ef71bdc25fb9add9`.
The paired-description JSON generated after that consumer revalidates the
sealed input is `results/reacher_fd_cp5_host_f64_PAIRED_20261102_v1.json`,
SHA-256 `37064876346e7bfbc1a5b9c0262a9a3b17b81d8b524e574c61e231a13ac20edb`.
It validates the saved event chain and witnesses, not independent physical
replay; source closure remains explicitly incomplete. The earlier partial
run and its selected pre-dispatch failures remain preserved.
