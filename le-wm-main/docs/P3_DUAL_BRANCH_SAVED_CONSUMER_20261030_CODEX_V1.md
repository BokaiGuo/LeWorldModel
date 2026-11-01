# P3 dual-branch saved consumer

The independent consumer is
`p3_dual_branch_saved_consumer_20261030_codex_v1.py`. It is read-only with
respect to the run directory and writes only a caller-selected new receipt.

It refuses to read a run until a caller-specified `systemd --user` unit is
loaded, inactive or failed, has `MainPID=0`, and its complete control group
contains no processes. It then requires an external SHA-256 for `end.json`,
the dual-branch terminal schema, and a complete decision journal terminal.
A missing terminal, torn journal, changed journal hash, active service or
changed terminal after the check is a refusal—not an incomplete-run
conclusion.

For a terminal run it verifies:

- the decision journal SHA, record index, predecessor hash and canonical
  record hash;
- the fixed six-row denominator, slot order, cleanup state and terminal row
  equality;
- each saved original H1/H3 signal map, original decision and recomputed
  `core.choose` decision without changing the saved decision;
- actual `progress/4` versus `reconfiguration/8` branch, branch-specific
  budget, attempted/completed ledger, both horizon panels, eligible census,
  winner rank and selected five-action prefix;
- decision-to-feedback ordering, packet boundaries, returned-action equality,
  encoder input SHA and next-context digest;
- native journal inventory and chain, native slot anchors to decision-journal
  indices, native dispatch/return float32 action bits, stop status, PNG/array
  references and saved latent hashes.

Failed decisions, setup failures and native stops are retained as statuses.
Partial ledgers are audited without padding or retry. A run without
`end.json`, including an external hard-kill partial, is deliberately not
interpreted by this consumer.

The output status is always `saved_records_checked_not_certified`.
`source_authenticated`, `physical_certified`, `full_saved_strict`,
`unconditional_equal_budget` and `scientific_pass` remain false. The
consumer does not establish full source closure, device attestation, hidden
work absence, equal budget across the two branches, or scientific validity.
It does not modify the running source or any historical result.
