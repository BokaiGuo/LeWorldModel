# Same-action cp5 velocity ablation: development result

The prewritten `PC_POINTMAZE_CP5_VELOCITY_SAME_ACTION_PROTOCOL_20261103.md`
replayed all 384 fixed plans from four exposed roots. Every cp5 and cp10
image and full cp10 state exactly matched prior saved replay. The prior
zero-velocity scores and masked MSE were reproduced within the fixed tolerance.
Frozen-model predictions ran on host CUDA; model parameter hashes agree.

| Root / bank | Masked MSE: zero / causal / privileged velocity | RGB-ranking reversals: zero / causal / privileged | Terminal distance at causal-score oracle choice |
| --- | --- | --- | ---: |
| 1884 random | 0.812 / 0.156 / 0.082 | 300 / 146 / 79 | 0.145 |
| 1884 spectral | 0.734 / 0.111 / 0.056 | 366 / 77 / 43 | 0.042 |
| 1885 random | 0.407 / 0.064 / 0.034 | 357 / 91 / 59 | 0.199 |
| 1885 spectral | 0.447 / 0.085 / 0.027 | 384 / 143 / 48 | 0.113 |
| 1886 random | 0.830 / 0.098 / 0.046 | 478 / 115 / 64 | 0.219 |
| 1886 spectral | 0.763 / 0.075 / 0.039 | 495 / 124 / 65 | 0.219 |
| 1887 random | 0.818 / 0.060 / 0.042 | 588 / 176 / 130 | 0.326 |
| 1887 spectral | 0.812 / 0.056 / 0.037 | 466 / 154 / 79 | 0.172 |

Each ranking count has 1,128 comparable pairs. On these dependent development
banks, causal RGB velocity reduces both mean masked MSE and ranking reversals
versus zero velocity in all eight; the privileged velocity diagnostic reduces
them further in all eight. The causal estimate has mean velocity L2 error
between 0.36 and 0.93 versus raw cp5 simulator velocity across banks, compared
with 2.11 to 4.89 for zero velocity. This establishes an important *measured
input confound*: the earlier zero-velocity refresh comparison is not evidence
that RGB feedback itself fails.

The causal-score oracle choice has lower terminal distance than the prior
zero-velocity oracle choice in all eight banks; it is below 0.5 in both root
1887 banks. **This is not a deployable choice:** all candidates' true cp5
images were obtained by executing their different prefixes. The raw velocity
arm additionally uses privileged simulator state and is strictly diagnostic.
The causal velocity is a pixel-quantized 0.05-second average, while raw qvel
is instantaneous; renderer/state temporal alignment is not independently
certified. Model success, mechanism, and independent generalization remain
unproved. In particular, an earlier actual closed-loop experiment on selected
prefixes had mixed causal-velocity outcomes, so the present oracle result
must not be substituted for real replanning.

Next experiment should use one causally observed prefix per arm, perform
same-budget tail optimization with fixed initial or prespecified feedback
mask, and execute the selected tails physically. Compare causal velocity and
zero velocity with identical feedback access and cost accounting; preserve
the previously negative closed-loop outcomes. Independent roots and
reconfiguration/strong-baseline admission remain separate requirements.

Artifacts:

- JSON `le-wm-main/results/a0_pointmaze_cp5_velocity_same_action_20261103_DEV01.json`,
  SHA-256 `1a31884a8ab092a04f9d6437a043c54815d1cc0821f212d47bb96a14e8a0569a`.
- NPZ `le-wm-main/results/a0_pointmaze_cp5_velocity_same_action_20261103_DEV01.npz`,
  SHA-256 `5f72892e58ce6837dfc03088d73641d5e8f0a70d3021033b46daeb8d276b14e8`.
- Combined saved-binding, causal-RGB input and metric tests: **3 passed**.
  They do not rerun neural inference or physics.
- Receipt reports 3,840 high-level physical steps, 1,152 cp5 encodes,
  384 cp10 encodes, and 1,152 one-packet prediction samples. The extra oracle
  executions cannot be presented as equal-budget deployable planning.
