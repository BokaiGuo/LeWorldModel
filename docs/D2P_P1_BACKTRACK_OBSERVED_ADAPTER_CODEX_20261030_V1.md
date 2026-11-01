# P1 backtrack to observed-H2 source adapter

Development-only saved-data preparation. No model, GPU, simulator, training,
or physical execution is invoked; this is not a three-arm experiment or a
physical authorization. The formal roster stays at eight logical arms times
selected/zero/greedy; greedy is unknown for all eight.

## Semantic join

The compact backtrack task has six top-level fields and cannot be handed
directly to the observed compiler. Its `task_sha256` points to the original
compressed overlay task export, which already contains a full compiler task.
The adapter checks the pinned export bytes, the original overlay interface,
the original task and contract digests, and exact equality of context, goal,
projection and normalizer against the compact task. It uses the original
complete task unchanged rather than manufacturing missing metadata.

The compact `task_contract_sha256` is an ABI binding created by the nonlinear
suffix library, not the original export's contract hash. Both identities are
retained separately; they are not asserted equal. Similarly the backtrack
`model_sha256` is the saved model state-dict identity, whereas the original
task model hash is the weights-file SHA. The adapter checks the former against
the historical model receipt and the latter and config hash against the
current checkpoint files. It does not load the model or independently prove
the file-to-state-dict relation in this run.

The original overlay's on condition intentionally has `with_target=true`
as renderer-only metadata, with no edited goal pose. The existing overlay
interface verifies this distinction; the off condition has `with_target=false`.
Warmup raw10 must agree with the physical-source join and normalize exactly
to the task's action history; observed compiler normalization, identity Q and
packet cap are checked. Each cp5 PNG is checked for bytes, RGB shape and pixel
digest. The 16 existing rows retain their original order, including duplicates.

The generated artifact contains the full model task, prefix, original ten
suffix actions and cp5 image reference; it deliberately excludes physical
states, snapshots, terminal outcomes and future images. This is a useful
development input, **not an isolation boundary**: image paths still point
into an old physical-run directory. Never hand this artifact directly to a
production planner process until access is isolated and reverified.

## Outstanding gates

1. Prove the eight greedy choices from the full common library and immediate
   goal, including tie rules, and obtain their own authentic cp5 RGB; do not
   substitute incumbent or silently reduce the 24 conditions.
2. Construct an isolated, allowlisted planner staging area and verify its full
   source/runtime closure, cp5 encoding on coordinated host CUDA, frozen model
   and component census. Require no physical-state/future-outcome access.
3. The two-H1 kernel is only exploratory composition. Enforce and reconcile
   actual per-packet receipts and end-to-end budgets, prove original contract
   compatibility, and separately account for prefix and encoder work.
4. Build and test an independent physical executor: new exclusive attempt,
   source-bound root/warmup/prefix replay, physical-only hidden-state and
   rendered checkpoint guards, literal frozen new 10-action suffix, explicit
   native-stop/partial nonzero behavior and independent saved consumer.
5. External GPU coordination, main-thread review and the unchanged scientific
   admission gates remain prerequisites. No host production command is
   authorized or provided here.

`MAIN01` was created by a preparation process while its new adapter source was
edited. Retain it but do not use it for acceptance; a distinct `MAIN02` is
prepared after the edit and must be verified against an external digest.
