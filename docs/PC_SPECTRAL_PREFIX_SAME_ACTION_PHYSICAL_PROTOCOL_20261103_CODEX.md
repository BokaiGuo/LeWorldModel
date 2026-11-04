# Same-action physical follow-up of sealed prefix bank

Execute every one of the eight prefix/suffix chains for each of the two
exposed cp12 contexts in the sealed
`pc_spectral_prefix_common_suffix_dev_20261103_CODEX.json`. Do not select,
resample, optimize or alter any action after seeing outcomes. For each chain
independently reset to the saved source state, replay its fixed 12-action
history, verify all 13 history RGB hashes, then execute all ten saved raw
actions. Use the durable single-bridge dispatcher with exclusive snapshots,
per-command attempt/return and no automatic retry on failure. All 16 slots
are required; failures and incomplete slots must remain visible.

At cp12, cp17 and cp22 encode actual returned RGB using the fixed causal
two-frame state estimator (initial cp12 uses frames 11/12). Compare actual
RGB-proxy scores with the sealed open model prefix and terminal scores at the
same action times. Evaluate physical position at cp12, cp17 and cp22 using
the existing MuJoCo 2.2 position-label replay on the exact saved actions.
These physical positions are evaluator-only and do not select actions. The
2.2 position backend is not the original 2.1.2 shadow environment; report
this backend and its cross-run limitation, not native success.

For each root report all eight outcomes, selected-chain identity, the
pairwise ranking reversals among the eight chains for model vs actual RGB
and actual RGB vs physical position at cp17 and cp22. Exclude ties in either
quantity rather than treating them as correct; action pairs share roots and
are not independent. Compare model selection with the physical fixed-bank
oracle only as hindsight diagnosis, not as actual method performance. The
two cp12 histories were already exposed during development. This does not
establish same-prefix ability interventions, independent confirmation,
fair total compute or a P1 admission.
