# v13 global proposals: E0 development result

## Scope and result

The predeclared four-arm, eight-task comparison in
`PC_E0_V13_GLOBAL_PROPOSALS_DEV_PROTOCOL_20261104_CODEX.md` completed all
32 plans. All four arms had **0/8** final distances below 0.05.

| Arm | Mean final distance | Successful / attempted | Mean wall seconds |
| --- | ---: | ---: | ---: |
| v13 global proposals + response | 0.502023 | 0/8 | 0.0303 |
| v12 response | 0.501554 | 0/8 | 0.0250 |
| Matched global proposals + full-action CEM | 0.501588 | 0/8 | 0.0295 |
| Three-start direct CEM | 0.501124 | 0/8 | 0.0319 |

These timings are small CPU-side E0 measurements, not RGB/GPU cost curves.
All plans used 1920 prediction tokens except one root in each response arm,
which used 1918. A shared token limit does not equalize derivative,
decomposition, search, or other resource costs.

v13 did not outperform v12 on any of the eight paired distances: four were
ties and four were worse. Relative to matched global CEM it was better on
four roots and worse on four. This small exposed development panel does not
establish a population comparison or a strong-search upper bound. The
global bank itself was unprivileged and contained zero, constant actions,
and two-phase actions; it did not include the privileged PushT witness.

## Audit

The sealed manifest, raw action/position/score traces, results and audit are
in `le-wm-main/results/pc_e0_v13_global_proposals_dev_20261104_codex/`.
The audit checked hashes, all scored trajectories against the original
exact implementation and a separately coded same-environment geometry
implementation, action bounds, token accounting, complete-chain incumbent
preservation, and identical initial candidate banks for v13 and global CEM.
This is an implementation consistency check, not independent replication.
All 32 plans and eight bank-pair audits completed.

The v12/v13 regression tests passed (24 tests). Tests and replay checks do
not establish the planning mechanism or a scientific gate.

## Interpretation and stop

Under this configuration, spending part of the same token budget on this
global proposal family did not produce a successful chain or improve upon
v12. It does not isolate whether coverage, retention, or optimization is
the limiting stage, and it cannot decide whether response reconfiguration
is useful in general. The separate PushT post-hoc witness motivates the
hypothesis but does not prove transfer to this E0 panel.

The predeclared 32-attempt stop has been reached. Do not extend or retune
this panel. PC2 remains unestablished: no final selected chain here has
passed its predeclared mechanism intervention. PC3 remains unestablished:
this is not an independent confirmation against the full required baseline
set. PC0, PC1, and PC4–PC5 are not certified by this experiment.
