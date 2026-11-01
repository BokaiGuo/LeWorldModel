# Independent saved GRASP-H1 physical outcome audit

Only new audit source/test/doc/output. Reads the completed actual physical
report, pinned to SHA
`1fa5bf810109ced7c36813d46915a842ed62b7dfe798d397eaf5fa93e60a5c62`,
and physical prepare SHA
`112cabb0684b7a16f3213a6b64759313c3711330571b2d4e6e38befa053399e8`.
No import of consumer/runner runtime, NumPy, torch, gym or simulator. PNG
decoding and exact RoyalBlue pixel counting are saved-data inspection only.

Checks independent arithmetic and full journal ordering, not just successful
report flags: eight unique root-arm selections, exact command/selection hashes,
whole-plan rejected classification and zero dispatch artifacts, durable branch
JSON/endpoint/budget files, initial context and goal0/frame SHA joins, all
selected attempts/returns/observations, fresh visible RGB before dispatch,
PNG hashes and pixel counts, termination/guard/full/partial/fresh classification,
trace equality, independent official and physical pose labels. Setup journal
is independently recounted by phase and root/arm, including resets, zero-action
warmup and exact selected commands. Upstream sources/seals are rechecked.
Inherited planning journal is streamed under 128-MiB bound and hash checked;
this audit does not reexecute planning or infer new derivative costs.

Denominators are **8 branches, 4 root pairs, one repetition, zero independent
new roots**. Completed single arms are not complete pairs. All rejected/partial
rows stay in output. Only complete 15-step same-context pairs contribute a
GRASP-minus-action-Adam physical pose difference. No partial/unmatched pooling.
This is real replay of H1-planned actions, not H3-initialized equal-compute
planning. Inherited costs remain 21624 H1 transitions, 24 separately acquired
H3 diagnostic transitions and 7200 backward calls, charged once; inherited
RGB/goal encoding cost remains unknown. No neural/physical work is added.

Important evidence limits: original pre-warmup state is not saved in branch
output, so its state/goal fingerprint cannot independently be recomputed here.
Root SHA/source provenance is bound; post-warmup state, goal and context are
joined. Physical output does not contain the whole original three-frame stack
or independent goal image: their recorded hashes are checked against sealed
input receipts, not called a new visual reconstruction. No fresh simulator or
encoder is run to fill these gaps. Source seals verify bytes, not historical
execution truth. No held-out efficacy, P0/P2 admission or full official GRASP
reproduction is inferred from these four calibration roots.

Tests and saved-only launch from `le-wm-main`:

```bash
.venv/bin/python -m pytest -q tests/test_p2_grasp_h1_saved_outcome_audit_20261027_v1.py
.venv/bin/python -m experiments.spectral_reachability.p2_grasp_h1_saved_outcome_audit_20261027_v1 --output results/p2_grasp_h1_saved_outcome_audit_20261027_v1
```

Output is exclusive and contains a source seal and machine-readable report with
all eight original rows, independently computed pairs, phase budgets/reset
counts, branch/pair denominators, full-pair-only comparison and explicit gaps.
Incomplete/unhashable/changed inputs are rejected, never repaired in place.
