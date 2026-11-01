# E1 partial compiled-ancestor integration

The exclusive consumer rebuilds the existing six-slot ledger from its pinned
saved inputs and rebuilds the compiled-beam ancestor audit from its pinned
sources. It verifies that the latter equals the saved ancestor audit with
SHA-256 `6c0149485e6f5238ea932b6eef7fd53d6e3763fd38863300a81a0d66102317fd`,
then joins only by the fixed root, compiled arm, task and target identities.
The public executable verifies provenance; the in-memory `join` function
requires already verified inputs and is not an authenticity boundary.

In compiled slots 2 and 5, the inherited phase contains the observed beam
`predict_API`, predictor sample, action encoder, and projection counts mapped
to the corresponding ledger axes. Each is 144 attempted and 144 completed.
Predictor module calls are retained separately because the ledger has no
matching axis. All other inherited axes remain null; noncompiled inherited
phases are unchanged. Controller and replay measurements remain separate.

The beam journal covers search and replay hooks, not the complete inherited
phase. The ledger's inherited scope remains `unknown`, and its receipt remains
null: the pinned journal is evidence of partial events, not a receipt proving
full phase coverage. The preflight therefore correctly flags these partial
measurements as unbound under the existing receipt contract. Their source and
event digests are separately recorded in `partial_inherited_source`; this does
not certify allocation of unknown shared costs. Proposal, earlier and shared
costs, cross-stage event overlap, coverage, and FLOPs remain unresolved. No
cross-stage totals or equal-budget claim are made.

Saved result: `results/codex_e1_ancestor_integrated_ledger_20261030_v1_LOCAL01.json`,
SHA-256 `0bd6601848d43ed8c5e0763f98e7858a7d6909347046b4a277497eadc0a5988b`.
It remains `blocked_incomplete`, with ranking and scientific admission false.
The exclusive writer refuses to overwrite an existing result.

CPU-only verification: 54 tests passed across the integrated, ancestor,
event-consumer, and ledger-interface test files. No P1, GPU, physics, or
production execution was used. The original ancestor audit's attribution
limits remain applicable; this integration is not an independent full replay.
