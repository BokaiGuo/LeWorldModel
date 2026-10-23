# PointMaze V10: Fixed-Policy Confirmation Findings

## Question and protocol status

V10 tested exactly one recurrent policy that V9 selected before opening its
outcomes: frozen latent history, width 16, contrast weight 1.0, temperature
0.75, and recurrent candidate scoring.  V10 did not search a feature family,
width, temperature, gate, or policy.  Its 1,088 fresh roots were disjoint from
all earlier PointMaze contexts: 256 source-training roots, 512 authorization
roots, 256 sealed outcome roots, and 64 separate robust-bank design roots.

The V9 selection file is hash-bound in V10 and declares that V9 outcome labels
had neither been opened nor generated at selection time.  V10's outcome labels
were materialized only after its V10 fixed policy and authorization decision
were frozen.

Relevant immutable artifacts:

- `results/pointmaze_counterfactual_contrast_v10_fixed_confirmation_20261022_cuda/`
- `results/audits/pointmaze_counterfactual_contrast_v10_fixed_forensic_20261022.json`

## Candidate-bank and authorization checks

The two disjoint candidate-design folds passed the pre-fit robust-bank gate:

| Metric | Fold 1 | Fold 2 |
| --- | ---: | ---: |
| normalized winner entropy | 0.866 | 0.870 |
| distinct winner fraction | 0.875 | 0.875 |
| best-second gap at least 0.003 | 0.594 | 0.625 |
| candidate-cost spread, 10th percentile | 0.151 | 0.108 |

Thus a collapsed candidate bank does not explain the result.  On all 512
fresh authorization roots, the fixed recurrent policy had mean paired regret
gain of `+0.000347` over action-only, but its 95% bootstrap lower bound was
`-0.001396`.  The authorization rule therefore correctly produced
`action_only_fallback` before any outcome label was generated.

## Held-out outcome

The independently opened 256-root outcome did not replicate V9's positive
diagnostic direction:

| Arm | Mean root regret |
| --- | ---: |
| action-only | 0.012649 |
| fixed recurrent | 0.012870 |

The paired recurrent-over-action gain was `-0.000221` with 95% bootstrap
interval `[-0.003227, +0.002928]`.  This is not evidence for deployment, and
it does not license changing V10's frozen policy after outcomes were opened.

## Mechanism diagnosis

The recurrent arm changed the winning candidate on only 65 of 256 roots
(25.4%).  Among those interventions, it improved 29 roots and harmed 36;
their mean gain was `-0.000871`, with a 5th-to-95th percentile range of
`[-0.037995, +0.030928]`.  The action-only winner distribution also retained
10 of 16 candidate IDs and a normalized entropy of 0.628, so this is not a
fixed-action collapse.

The supported conclusion is narrower and more useful than a generic failure:

> A learned history-action interaction can change candidate rankings, but the
> currently available pre-action signals do not reliably identify which
> overrides improve true task regret.  Enlarging the authorization panel alone
> does not resolve that ambiguity.

This is offline, renderer-backed PointMaze evidence only.  It is neither a
visual closed-loop success nor a general claim about learned world models.

## Consequent intervention

The justified next test is not another outcome-tuned static selector.  The
candidate-anchored affine active-probe protocol first obtains an observed RGB
action-effect chart using safe actions from the exact planner bank, then allows
an override only under frozen rank, leave-one-out, convex-hull, and successor
verification gates.  Its result must be evaluated on new held-out roots under
the same physical pulse budget.
