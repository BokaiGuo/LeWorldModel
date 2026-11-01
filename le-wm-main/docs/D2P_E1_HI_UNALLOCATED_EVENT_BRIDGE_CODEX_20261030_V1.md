# E1 Hi-FIRST source-row event bridge

The Hi-FIRST ancestor has events embedded in its saved report, not a separate
raw attempt journal. The existing pinned audit checks each row's event indices,
parent references, returned flags and input layouts, reconciles row aggregates
with the report and consume receipt, and checks the delivered target and blind
identity for the two `off` rows. The bridge recomputes that audit and checks it
against the saved trace before joining the source rows to the two existing
learned-prior slots by fixed row, root, task and target identities. It also
recomputes and checks the previously integrated compiled-ancestor ledger.

The fixed report SHA-256 is
`083d0d9aa190b561e6ccf865b09d7224b97d404b16ba38c792bd8af386f5622a`;
the consume receipt SHA-256 is
`40f28414682c9540e9e195b361a36dfc514f816358adbeacb379fedc460e4ef0`;
the prior event trace SHA-256 is
`b4382de707531a59e36bc287f8932656f2cb3186cf552e13e96d5f0f68d1d349`.
Source paths and digest construction are in the existing Hi trace tool.

Each source row records 2 returned action-encoder events, 2 image-encoder
events, 3 high-cost events, 14 high-predictor events, 2 low-predictor events,
and 98 high-predictor input rows. These are row execution observations, not
proof of the minimal or complete causal cost of the inherited target. Failed
attempt coverage, shared allocation, earlier ancestors, overlap with other
stages, and FLOPs are unknown. Axis hooks may be nested and must not be summed
as independent work. The trace's attempted field is derived from returned
events, so this bridge explicitly reports observed returns only.

The bridge attaches these observations to source references on slots 1 and 4,
without populating proposal measurements or assigning a proposal receipt.
Their proposal phases therefore remain unknown, as do all other unresolved
costs. The result is `blocked_incomplete`, with no parity or ranking authority.
The in-memory `join` accepts previously verified inputs; only the executable
entry point performs source hash checks and recomputation.

Exclusive output:
`results/codex_e1_hi_unallocated_event_bridge_20261030_v1_LOCAL01.json`,
SHA-256 `7673ac35fcf7103e8231b5b6e0ea1578da05528ebb18fce587119a5a8879261c`.
The exclusive writer refuses overwrites. The bridge, Hi trace, and prior
integrated-ledger tests passed together: 37 tests. No production, GPU,
physics, or P1 access was used.
