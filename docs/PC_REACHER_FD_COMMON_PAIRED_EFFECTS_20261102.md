# Reacher FD fixed-tail physical development: paired effects

The host execution of the previously frozen plan recorded 20/20 branches,
all 15 steps, with no termination or truncation and reward 0 on every
returned step. These are two exposed development roots, not 20 independent
roots. The model's strict selector returned null on both, and the selected
FD prefix is not a spectral Jacobian/SVD program.

An independent analysis recomputes wrapped joint distance for every state,
checks all branches and tails are present in order, checks equal common tail
actions and identical saved same-prefix trajectories through step 5. A
separate saved-output consumer also checks the sealed action/plan/model
bindings, decoded RGB hashes, per-step actions, stops, and source hashes:
`le-wm-main/results/reacher_pc_fd_common_physical_v2_audit_20261102_codexexclusive_01.json`.
That consumer does not independently replay physical execution.
Selected minus parameter-zero: negative means the selected result has lower
wrapped joint distance. Parameter-zero is not necessarily zero raw action.

| root | selected cp5 − initial | parameter-zero cp5 − initial | selected − parameter-zero cp5 |
| --- | ---: | ---: | ---: |
| 20263160 | -0.016481 | +0.006530 | -0.023011 |
| 20263161 | +0.027241 | +0.062843 | -0.035602 |

The table below contains all fixed tails, without selecting favorable ones.
The adjusted contrast is `(selected tail − selected tail 0) −
(parameter-zero tail − parameter-zero tail 0)` in terminal distance.
Tail 0 is a parameter-zero tail, not a claim of zero physical action or
zero passive motion. This adjustment describes a comparison to that control;
it does not fully isolate controllability or establish causal mechanism.

| root | tail | selected final | parameter-zero final | absolute difference | adjusted difference |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20263160 | 0 | 0.987554 | 1.008969 | -0.021414 | 0.000000 |
| 20263160 | 1 | 0.800758 | 0.837686 | -0.036927 | -0.015513 |
| 20263160 | 2 | 1.353606 | 1.362116 | -0.008511 | +0.012904 |
| 20263160 | 3 | 1.473673 | 1.503425 | -0.029753 | -0.008338 |
| 20263160 | 4 | 0.655822 | 0.650048 | +0.005773 | +0.027188 |
| 20263161 | 0 | 1.206365 | 1.246073 | -0.039709 | 0.000000 |
| 20263161 | 1 | 1.192106 | 1.232042 | -0.039936 | -0.000228 |
| 20263161 | 2 | 1.423978 | 1.457395 | -0.033417 | +0.006292 |
| 20263161 | 3 | 1.752340 | 1.793014 | -0.040674 | -0.000965 |
| 20263161 | 4 | 0.687400 | 0.723443 | -0.036043 | +0.003666 |

The first selected prefix improves the distance immediately from the initial
state; the second worsens it but remains closer than parameter-zero at cp5.
Adjusted effects vary in sign, and all terminal rewards are zero. The results
do not demonstrate a consistent advantageous reconfiguration mechanism,
same-budget independently optimized suffix benefit, or P1 admission.

The sandbox attempt failed RGB identity before dispatch; the subsequent host
reconstruction matched the frozen identity and ran this unchanged plan. This
does not establish which host/sandbox environmental difference caused the
original mismatch.

Input report SHA-256:
`4e100e02f8c3feda959f3b25883663e66b063126d4d599fd83a77a40607caa3e`.
Analysis JSON SHA-256:
`9784a9fd941ee2d06afd7c8e58d8f54b782e6c2dfbd71194b92a9eada443699d`.
The separate analysis script has six passing synthetic validation tests;
these tests are not physical evidence.
