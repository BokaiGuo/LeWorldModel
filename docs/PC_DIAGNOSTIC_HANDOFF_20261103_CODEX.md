# Atlas and PointMaze diagnostic handoff

These are development diagnostics on exposed data, not independent confirmation
or a PC gate result. See the linked protocols and receipts for source bindings.

## Atlas: injected strong optimizer chain

The existing `PC_ATLAS_ORACLE_INJECTION_RESULT_20261103.md` records eight
exact-unicycle goals. The strong gradient optimizer's chain was accepted,
retained and selected in all eight; the split and direct replay context hashes
matched. This identifies a search-coverage concern in this particular setting,
not a general proof that scoring and pruning are correct. The injected
solutions and unequal compute do not count as Atlas performance.

## PointMaze: fixed-action ranking reversals

The existing `PC_POINTMAZE_VELOCITY_SCORE_CROSSTAB_RESULT_20261103.md`
compares 384 fixed plans, eight dependent groups and 9,024 within-group pairs:

| cp5 velocity | predicted score | vs actual RGB cost | vs physical distance |
| --- | --- | ---: | ---: |
| zero | visual | 3434 | 3383 |
| causal RGB | visual | 1026 | 2229 |
| simulator qvel, diagnostic only | visual | 567 | 2318 |
| zero | proprio | 3797 | 2994 |
| causal RGB | proprio | 3677 | 2894 |
| simulator qvel, diagnostic only | proprio | 3636 | 3045 |
| actual RGB cost | — | — | 2167 |

Candidate-specific cp5 frames and qvel are oracle diagnostic inputs; qvel
must not be used in the RGB-only planner. These are dependent pair counts,
not independent trials. Lower RGB prediction error does not establish
better physical action selection.

## Optimizer and saved-tail checks

The AP-compatible CEM scores clipped proposals but fits its sampling
distribution to unclipped elite proposals. The same behavior is present in
the pinned official-compatible kernel. It is an observable algorithmic
choice/mismatch, not evidence that historical results are invalid. A
separate `projected_elite_update=True` variant now fits the scored packets;
the default remains unchanged for baseline parity. No neural or physical
comparison of the variant has yet been run.

`pc_lookahead_saved_tail_binding_audit_20261103.py` checked the saved
nonprogress lookahead archive: all 36 tails across four roots exactly match
the winner in their final saved CEM batch, and reported costs exactly match
the saved final batch. This is a saved-data binding check, not an independent
model rerun or proof of physical effectiveness. Regression tests also cover
tampered scores and tails.

The prior spectral-injection comparison
`PC_POINTMAZE_AFFINE_SPECTRAL_CEM_RESULT_20261103.md` was 4 wins / 4 losses
against random CEM on exposed roots, but spectral structure had extra compute;
it is not an equal-total-budget spectral ablation. A matched-resource
spectral/random/lookahead comparison remains outstanding.
