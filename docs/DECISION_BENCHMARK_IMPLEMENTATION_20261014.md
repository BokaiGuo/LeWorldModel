# Decision benchmark implementation status — 2026-10-14

This artifact records the first code-level execution of the next-stage strategy.
It does not upgrade the scientific claim beyond the evidence below.

## Implemented

- The PointMaze task readout is now explicitly affine:
  \[
  \widehat{\Delta s}=(z-\mu_z)W+\mu_{\Delta s}.
  \]
  Both centers are fitted on training rows only and are carried through the
  planning, calibration, external-intervention, and diagnostic paths.
- `experiments/decision_benchmark/audit_action_effects.py` adds an independent
  root-disjoint `+axis/-axis/zero` audit with finite-difference action Jacobian,
  rank, condition number, identifiable input directions, root bootstrap CIs,
  and support coverage.
- `experiments/decision_benchmark/generate_candidate_bank.py` creates a
  goal-conditioned PointMaze bank with the same initial state, explicit goal,
  16 complete candidates, and 20 simulator steps per candidate.
- `experiments/decision_benchmark/evaluate_counterfactual_regret.py` compares
  fixed random, action-alignment, and train-only affine task-decoder selection
  on the same bank and scores only after selection.
- The `paired_order_v2` bank includes same-net-action, different-order pairs so
  a net-action heuristic cannot be treated as a learned transfer result.

## Executed artifacts

Primary smoke/full run for one existing learned checkpoint:

- action audit: `le-wm-main/results/decision_benchmark_action_audit_20261014/jepa_task_effect_mlp_seed8411_stage005_full/action_identifiability.json`
- hard candidate bank: `le-wm-main/results/decision_benchmark_candidate_bank_20261014/jepa_task_effect_mlp_seed8411_stage005_hard_v2/`
- D0 validation: `.../d0_validation/d0_report.json`
- regret report: `.../counterfactual_regret.json`
- readout repair panel: `le-wm-main/results/pointmaze_readout_intercept_repair_20261014/`

The bank contains 40 episode roots, 16 candidates per root, 20 steps per
candidate, and root-disjoint train/calibration/outcome partitions (24/8/8).
D0 schema/provenance validation passes. D1 is intentionally not ready because
new simulated candidates have no RGB rollout and therefore no realized visual
latent. The validator reports this as a warning and keeps scientific readiness
blocked.

The action audit gives normalized inverse-risk estimates of approximately
0.512, 0.798, and 0.896 at adapter horizons 1, 2, and 4 respectively. The
predicted Jacobian is rank 1 while the simulator effect Jacobian is rank 2 in
this checkpoint; this is direct evidence of an action-identifiability failure,
not a positive planning result.

On the eight held-out outcome roots in `paired_order_v2`, decision regret was
approximately 0.111 for random, 0.064 for action-alignment, and 0.137 for the
train-only latent task decoder. The decoder therefore does not beat the strong
simple baseline in this run. The v1 bank is retained as a separate artifact;
the v2 change is a benchmark repair, not outcome-driven tuning.

The compatibility repair was also rerun over the 50-record v2 external panel
with 0 failures. The old records have mean planning-risk approximately 0.675,
whereas the train-only affine-readout recomputation is approximately 0.311.
This is an evaluation-protocol correction, not a learned-model improvement;
all old records remain unchanged.

## Current boundary and next gate

These artifacts establish a reproducible P0/P1 benchmark and expose the next
missing component: RGB execution of each candidate sequence, frozen encoder
re-encoding of the realized terminal observation, and an independent
predicted/realized transition-consistency gate. No GDD-WM distillation or
closed-loop visual-planning claim is permitted before that component exists.

## Follow-up repair and causal-history branch

The initial hard bank was not a valid success benchmark: its outcome roots
had zero positive task labels. `generate_candidate_bank.py` now supports the
pre-outcome `nearest_goal` source rule and `distance_balanced_v1` root split.
The D0 validator reports outcome positive-label support and warns when
success-based estimands are unsupported.

The intervention adapter now reports paired regret/success gains and supports
four explicitly separated input modes:

- `frozen_latent_action`;
- `history_latent_action` (causal frozen latent history only);
- `privileged_state_action` (diagnostic upper bound);
- `latent_plus_state_action` (diagnostic upper bound).

The history branch uses source window 15 and only the preceding windows
`0,5,10,15`; it never uses future windows relative to the decision point. On
the repaired bank, the history branch reduced held-out regret from `0.073`
for single-frame frozen latent to `0.064`, with paired gain `+0.127` versus
random and success gain `+0.15`. This is a positive intervention-level
result, not yet RGB successor verification or closed-loop planning evidence.

## 2026-10-14 data and PushT closure audit

The previously present `pusht_expert_train.h5.zst` had the expected byte count
but failed Zstandard decoding with `Data corruption detected`. It was replaced
from the declared `hf-mirror.com` endpoint and successfully decompressed to a
46,300,921,856-byte HDF5. Any experiment that depended on the corrupt artifact
is retained as historical evidence but is not used as a clean model failure.

The contact-stratified paired intervention then ran with 64 train and 32 held-
out roots per stratum (`free`, `approach`, `normal_contact`, and
`tangential_contact`). Geometry specialists improved free/approach prediction,
but did not yield a stable advantage in normal/tangential contact. This
separates mode observability from within-mode action identifiability.

The signed-torque audit was also repaired. Its old center-radius root protocol
and center/vertex action frame could produce high direction accuracy while
losing the actual T-boundary contact. The corrected audit uses actual polygon
boundary roots and a boundary inward normal. It reports classifier regret below
the geometric baseline, but direction accuracy `0.609`, angle improvement
`+0.124 rad` with a CI crossing zero, and strict success `0/32`; it therefore
does not establish stable pose closure.

Finally, RGB-only repaired predictor MPC completed on eight held-out episodes.
Repair reduced block-XY error (`172.74 -> 159.47`, and `145.38` under a
one-step action block) and endpoint pose error (`262.31 -> 126.67`), while
strict success remained `0/8`. The result is a valid local transfer gain, not
an online-planning success claim. The full six-goal evidence matrix is in
`docs/SIX_GOALS_DECISION_TRANSFER_EVIDENCE_MATRIX_20261014.md` and the
root-cause record is in `docs/PUSHT_NEGATIVE_RESULT_REPAIR_20261014.md`.
