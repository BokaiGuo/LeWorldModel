# E1 compiled-witness ancestor event audit

This is a local, read-only audit of a pinned historical beam-run journal, not
production, replay, inference, or an equal-budget certification. The saved
result remains `incomplete`; it does not authorize ranking or scientific
admission.

## Fixed sources and attribution

All paths below are relative to `le-wm-main/results/`. The tool checks the
entire file against the listed SHA-256 before consuming it.

| Source | SHA-256 |
| --- | --- |
| `agent_experiment1_common_controller_20261028_v1_prepared_MAIN01.json` | `8c472485d44c40ce3d95f8043d2c58f9ec2694fef74786a840d1699d5a1a86bd` |
| `agent_e1_six_slot_common_prepare_20261029_v1_prepare_MAIN01.json` | `cd96f6e28b25d3b7e3b6649c760b2f111747aec5c0ce9c8a41741318209df8a0` |
| `agent_kinodynamic_matched_domain_20261028_v1_cuda_MAIN01/report.json` | `d4976aeca357b5324c74be8b2b7e3bfc254e82b834b3acfe1b0ab90a1ef4614c` |
| `agent_kinodynamic_matched_domain_20261028_v1_cuda_MAIN01/events.jsonl.gz` | `ef8c43ec73ba9cc66f9f1708bd7c348651817f211c0a43302f24637c7469b623` |

The old prepare binds the report and journal and records the compiled target,
selection and result identities. The six-slot prepare identifies the inherited
compiled target. The audit checks matching root, task, blind identity,
selection, declared result digest, and float32 target bytes; it checks the
journal's root boundaries and attempt/return ticket identities and reconciles
its counts with the report. This establishes attribution through the saved
source records, not an independent full re-execution or validation of the
result gzip: that gzip is not opened here.

For each of the two roots, the journal records 144 attempted and 144 returned
calls on each of `predict_API`, `predictor`, `action_encoder`, and `pred_proj`,
and 144 attempted and returned predictor samples. These are observed hooks of
the ancestor beam run, including search and replay, not total proposal or
historical shared costs. Module axes can overlap or nest; predictor samples and
predictor calls reference the same events. Do not sum these axes as independent
work, or add this stage to another ledger without checking overlap and
allocation. Earlier/shared ancestor costs, other proposal/resource axes,
coverage and FLOPs remain unverified; missing costs are not zero.

Each axis digest is SHA-256 of a sorted-key, compact JSON object followed by a
newline, containing the fixed compressed journal SHA, zero-based root, axis,
and ordered pairs of zero-based original JSONL line index and SHA-256 of the
original line bytes including newline. The journal itself is not a hash chain.
Different axis labels produce different digests even when they reference the
same events.

The exclusive output is
`codex_e1_compiled_ancestor_ledger_20261030_v1_LOCAL01.json`, SHA-256
`6c0149485e6f5238ea932b6eef7fd53d6e3763fd38863300a81a0d66102317fd`.
It is a separate stage reference and does not alter existing ledgers.

## Verification

The dedicated tests exercise the real pinned files, counts and deterministic
digests, broken attribution, altered or missing dispatch events, truncated
gzip, digest mismatch, and exclusive output creation. Together with the
existing event-ledger consumer tests, 24 tests passed using CPU-only pytest.
No P1 source was opened and no GPU or physical production was started.
