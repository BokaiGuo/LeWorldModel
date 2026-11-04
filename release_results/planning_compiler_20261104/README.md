# Planning Compiler development snapshot — 2026-11-04

This snapshot publishes development code, tests, protocols, reports and a
bounded selection of saved JSON records. It is **not a scientific gate pass,
an independent confirmation, or a complete reproduction archive**.

## Evidence boundaries

- The v12/v13 recursive implementation supports carry rescoring, full-chain
  incumbent preservation/refinement, packet-aware accounting, and paired
  zero-prefix comparison. v14 additionally conditions a recursive nominal
  on the chosen global carry.
- The v13 exact-Wall prospective panel completed 12 sealed plans and their
  interventions: three final successes, one base mechanism intersection, and
  zero full intersections. This is a limited development witness.
- On the separate v14 eight-root comparison, v14 and v13 each succeeded once,
  while matched global-bank CEM succeeded twice. It does not establish a
  response-search advantage.
- v12 completed three real JEPA contact-task feedback decisions per arm;
  both arms failed the physical success rule despite a separately replayed
  evaluator-only feasibility witness.
- The v13 new-contact run and v13 Reacher run stopped at infrastructure
  preflight. They are not scientific negative outcomes.
- The v13 synthetic-observation isolation smoke completed six real-model
  cases. This verifies an engineering interface, not physical control.
- Independent planning advantage, qualified generalization and the complete
  PC0–PC5 claim remain unestablished. Historical stopping rules are retained.

See the corresponding reports:

- [v13 prospective mechanism](../../docs/PC_V13_E0_PROSPECTIVE_MECHANISM_DEV_RESULT_20261104_CODEX.md)
- [v14 carry nominal](../../docs/PC_E0_V14_CARRY_NOMINAL_DEV_RESULT_20261104_CODEX.md)
- [v12 contact feedback](../../docs/PC_JEPA_NATIVE_V12_CONTACT_FEEDBACK_DEV_RESULT_20261104_CODEX.md)
- [v13 contact infrastructure failure](../../docs/PC_JEPA_V13_NEW_CONTACT_FEEDBACK_DEV_RESULT_20261104_CODEX.md)
- [v13 Reacher infrastructure failure](../../docs/PC_V13_REACHER_RGB_FEEDBACK_DEV_RESULT_20261104_CODEX.md)
- [v13 synthetic isolation smoke](../../docs/PC_JEPA_V13_SYNTHETIC_ISOLATION_SMOKE_RESULT_20261104_CODEX.md)
- [runtime preflight contract](../../docs/PC_RUNTIME_PREFLIGHT_V1_20261104_CODEX.md)

## What is included and omitted

`snapshot_manifest.json` binds every copied file to its source path, byte
count and SHA-256. Compact saved records are copied under `saved_records/`
with their original result-relative hierarchy. These copies preserve the
original bytes; embedded paths and dependency hashes refer to the original
run and are not silently rewritten.

Datasets, model weights, third-party source trees, local environments,
conversation transcripts, raw arrays/images/videos, and large decision
records are not uploaded. Consequently, historical reports may refer to
local artifacts absent from this repository. The omitted JSON record list
is in the manifest; it is not an inventory of every omitted binary.
Some scripts retain machine-specific paths and require separately acquired
dependencies. Saved-data audits are not independent model or physics replay.

The publication was prepared in a separate temporary clone so that the
active research worktree and its staging area were not changed.

## Publication checks

All selected Python files were parsed and selected JSON files decoded before
copying. Common private-key and credential token patterns were checked, and
copied bytes were checked against their source SHA-256.

The isolated snapshot passed 41 tests across
`test_pc_recursive_response_v10.py` through
`test_pc_recursive_response_v14.py` and
`test_pc_v13_reacher_feedback.py`, using the existing local Python environment
with bytecode and pytest cache writes disabled. This is targeted regression
coverage, not the full repository suite, an experiment rerun, or a gate pass.
