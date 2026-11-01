# P1 feedback / diversity / fixed-prefix experiments

## Completed evidence

Three predefined variants × two known roots × four arms = 24 completed host-CUDA
planning plus physical trajectories. No neural CPU fallback, extra model training,
outcome-based root deletion or gate relaxation.

| Variant | Structured root0 | Structured root1 | Immediate-progress successes |
|---|---|---|---|
| First-feedback carry | success15 | success14 | 2/2 |
| Response-diverse + carry | no success20 | success14 | 2/2 |
| Fixed initial prefixes + diverse carry | success11 | success12 | 1/2 |

These are descriptive, correlated development results. Four internal arm names
are not certified GRASP/CEM/other external strong baselines. Actual heterogeneous
compute costs are recorded, not converted into an unverified fairness claim.

Source reports and initial trace hashes, component-wise immediate changes, bank
singular values and exact prefix-preservation checks are saved in:
`le-wm-main/results/p1_feedback_variants_summary_20261101_main_MAIN01.json`.
The three protocols and runners have matching `FIRST_FEEDBACK_CARRY`,
`RESPONSE_DIVERSE`, and `PREFIX_PROTECTED` names.

Saved independent selection/ledger checks consumed 14+13, 15+13, and 13+15 nodes
respectively: 83 actual decision nodes total. Each check verifies trace chains,
original/final candidate identity, old plan tail, actual first-packet dispatch,
checkpoint RGB binding, finite deterministic score selection and exact extra/total
node charges with module-hook agreement. They do not independently reproduce the
entire physical event chain.

Prefix-protected report hashes:
- root0 `17ffd129c42e7f56f9b7dafd0eafc57b034e92b21774b6640af8072b83826322`
- root1 `8dcd11721b5f2e0256d8613e4386310d8265fd26a126c82548ee781517670845`

## What changed and what was learned

1. Original carry ignored cp5. First-feedback carry compares original tails from
   cp5 onward on actual RGB, with matched effective horizons and metered scoring.
   Root1 keeps the original tail at cp5 and cp10 and succeeds at14. This closes
   a real deployment retention gap, not proof that all old failures were code bugs.
2. Original DLS initialization is essentially one direction with eight scalar
   multipliers. Adding three signed spectral directions gives distinct coverage,
   but structured root0 worsens. More directions alone is not a validated fix.
3. The original `raw_suffix.optimize` updates ALL three packets. The name
   "suffix" must not be read as a fixed-prefix guarantee. The new control protects
   the initial packet through base optimization AND winner refinement for all arms,
   with exact saved-array verification. It succeeds on both development roots,
   but this is an algorithm ablation, not a retroactive declaration of a bug in
   legitimate joint optimization.
4. Serialization failure (MAIN01 first-feedback) and nonblocking GPU-lock refusal
   (MAIN01 diverse) happened before experiments, were retained, and are not
   scientific negative results.

## Why P1 is still closed

Every selected structured prefix in these six root-variant instances improves
all three reported immediate physical errors. For the strongest fixed-prefix
variant, changes are:

| Root | Pusher XY error | Object XY error | Wrapped object angle |
|---|---:|---:|---:|
| 0 | -61.23521 | -37.47189 | -0.202633 |
| 1 | -6.06334 | -0.62720 | -0.006235 |

Thus the newly successful method is not an established non-immediate-progress
mechanism. Do not reinterpret a 2/2 success result as P1 passage, reinterpret
small improvements as zero after seeing outcomes, or choose a different immediate
metric to rescue the claim.

The remaining work is not simply more ledger checks: candidate/task design must
produce a model-selected prefix satisfying the original non-greedy condition,
followed by shared-action and own-planning capability interventions. Then freeze
the full method and run independent, appropriately powered, resource-constrained
strong-baseline confirmation. Historical reset and training-source uncertainty
remain explicit. Current two cp10-goal development tasks do not establish whether
the mechanism exists or is necessary on the broader target distribution.
