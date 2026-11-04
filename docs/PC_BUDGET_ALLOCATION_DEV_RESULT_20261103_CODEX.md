# Budget allocation development result

Protocol: `PC_BUDGET_ALLOCATION_DEV_PROTOCOL_20261103_CODEX.md`. These are
reused development roots, not independent confirmation. No admission gate was
modified.

## Resource scheduling

The new deterministic allocator gives each prefix an equal share of complete
trajectory samples, distributing any remainder in stable bank order. CEM
accepts a final batch of 1–11 samples. For singleton batches it evaluates but
does not update the sampling distribution. For other batches this new optimizer
uses population standard deviation; therefore this run is not a pure
one-variable replay of the previous optimizer, which used sample standard
deviation. All four new arms use the new optimizer.

All 32 root plans spent exactly their 900 or 1800 prediction-token caps.
Spectral candidate generation spent 318 tokens and six SVDs per root plan;
random generation spent 18; nominal and full-horizon CEM generation spent
zero. cp10 costs an additional 96 prediction tokens per arm, with three RGB
encodings per planning decision. Mean root planning time, including root
encoding but excluding shared model loading and goal encoding, was 8.56,
8.03, 8.04, 8.04 seconds at 900, and 16.55, 16.06, 16.07, 16.08 seconds
at 1800, for spectral, random, nominal and CEM respectively.

Token equality does not prove equal FLOPs or end-to-end resource fairness.
SVD, encoding, call counts and wall time are recorded separately. The inherited
execution output correctly retains `full_cost_certified=false`; the scheduler
source hash is recorded below but was not embedded in the sealed root JSON.

## Feedback execution and physical replay

Distances are to the fixed physical goal; lower is better. Every arm executed
10 root-plan actions, replanned five actions from its own real cp10 RGB and
action history, then executed those actions. All 32 plans and 232 physical
feedback/common-suffix traces were saved.

| Root | Budget | Spectral | Random | Nominal | Feedback CEM |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 900 | 0.1849 | 0.2936 | 0.2936 | 0.2145 |
| 0 | 1800 | 0.1849 | 0.2936 | 0.3117 | 0.2148 |
| 1 | 900 | 0.0591 | 0.0591 | 0.0154 | 0.0343 |
| 1 | 1800 | 0.0591 | 0.0154 | 0.0154 | 0.0340 |
| 2 | 900 | 0.2900 | 0.4095 | 0.2862 | 0.3686 |
| 2 | 1800 | 0.2900 | 0.4464 | 0.5345 | 0.3782 |
| 3 | 900 | 0.0318 | 0.0339 | 0.0339 | 0.0949 |
| 3 | 1800 | 0.0318 | 0.0339 | 0.0728 | 0.0949 |

The spectral method beats feedback CEM on six of eight paired development
settings; it does not beat nominal on three. These eight settings are not eight
independent roots, and their terminal distances are not a success-rate or
statistical superiority claim.

At the selected 5/10-step boundary, physical distance changes for roots 0–3
were respectively -0.91449, -0.34436, -1.20403 and -0.47409 at each budget.
Thus all eight selected prefixes improved position immediately. The
predeclared non-improvement condition fails for every decision. The separate
zero/greedy real-RGB replanning mechanism arms were not run, and the existing
common-suffix output compares the four method-selected prefixes and zero, not
a same-bank immediate-greedy prefix. They cannot support the full mechanism
test. Nevertheless, because the necessary first condition fails for all eight,
the declared stopping rule is triggered: pause this candidate construction for
the non-immediate-progress mechanism and reassess task/mechanism fit. Retain
the planning-performance development signal separately.

## Fixed-candidate quality diagnostic

The two sealed 900-budget banks were deduplicated by horizon and exact prefix
bytes, giving 22 candidates per root. Each received 48 root-state model
trajectory samples, after which its best full 15-action chain was physically
replayed open loop. The table reports the physical terminal distance of the
model-selected candidate within each category. Nominal is the six shared
nominal candidates; spectral and random are their corresponding perturbations.
The separate category selections and open-loop replay are post-seal
diagnostics, not deployable method scores or the feedback comparison above.

| Root | Nominal | Spectral perturbation | Random perturbation |
| --- | ---: | ---: | ---: |
| 0 | 0.6194 | 0.3955 | 0.6034 |
| 1 | 0.1294 | 0.1777 | 0.0661 |
| 2 | 0.5398 | 0.3903 | 0.5138 |
| 3 | 0.1064 | 0.0535 | 0.1338 |

This suggests candidate-quality differences worth examining, but does not
identify a robust spectral advantage. In root 1 the best physical spectral
chain had distance 0.0316 while the model-selected spectral chain had 0.1777;
the former is a post-hoc diagnostic, not an authorized selection. This also
does not establish that the previous physical gap was caused by allocation.

## Reproduction and boundaries

The sealed artifacts use the prefix
`le-wm-main/results/pc_budget_allocation_compare_20261103_codex` for root,
cp10, execution, RGB and physical replay; the separate quality artifacts use
`pc_budget_candidate_quality_20261103_codex`. The full scheduler SHA-256 at
execution was `bfc31378f15c2e7ba912d3a9ec1f4127ff9baf060f2e0071a7125f6ba75656fb`.
The protocol SHA-256 was
`8a281a02439433eee98987e802866c99c4db32b84f349cf0dd760f7c0da8d34f`.
Unit tests: 9 passed. Training-data membership, independent root provenance,
complete cost certification, statistical confirmation, and the required
independently replanned zero/greedy mechanism comparisons remain unresolved.
