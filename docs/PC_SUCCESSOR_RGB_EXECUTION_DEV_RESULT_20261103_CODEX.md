# Successor composition in RGB feedback: development result

The protocol is `PC_SUCCESSOR_RGB_EXECUTION_DEV_PROTOCOL_20261103_CODEX.md`.
Four previously exposed roots were evaluated with a successor-composition
policy and a direct CEM control, each using three real-RGB decisions at
15/10/5 remaining actions. These are development observations, not independent
confirmation.

| Root | Successor final distance | CEM final distance |
|---|---:|---:|
| 0 | 0.23184 | 0.10122 |
| 1 | 0.04605 | 0.09916 |
| 2 | 0.02525 | 0.06898 |
| 3 | 0.04995 | 0.05452 |
| Mean | 0.08827 | 0.08097 |

Successor won three paired roots but had a worse four-root mean due to root 0.
This tiny exposed panel does not establish an advantage, a failure rate, or
generalization. The zero-prefix fallback was selected in 0/12 successor
decisions, so its physical effect remains unobserved.

At the first decision, each run generated four two-segment prefixes from
first-segment and successor-conditioned model responses; the four suffix
searches received 24, 23, 23 and 23 complete-chain samples, each more than one
CEM batch. Joint refinement produced the selected pre-comparison chain in
all four runs. Later decisions used the existing multi-nominal policy and
rescored the carried tail. This is one executable development policy, not a
claim that the same composition search runs at every decision.

Successor used 1,439 prediction tokens and seven SVDs per three-decision
trajectory; CEM used 1,440 prediction tokens and no SVDs. Prediction tokens
do not capture all computation or imply total-cost equivalence. The records
remain `full_cost_certified=false`.

The audit checked 24 prediction traces, counterfactual selection, exact
model-input casts, budget counts and policy hash. Native MuJoCo 2.1.2 replay
checked the eight executed trajectories, RGB receipts and carry binding.
Model scores were not independently rerun in this audit. The new policy
records its own source hash, but the inherited runner's generator hash binds
its legacy generator, not all actual successor-policy dependencies; this is
a provenance limitation to repair before any confirmation.

No same-chain physical mechanism intervention was conducted. PC2 and PC3
remain unpassed; this result also does not establish the original spectral
increment, PC4 long-horizon advantage, or PC5 independence and generalization.
