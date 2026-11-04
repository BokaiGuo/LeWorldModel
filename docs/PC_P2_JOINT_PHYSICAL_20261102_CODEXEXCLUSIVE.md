# Fixed-winner joint versus conditional: physical development contract

This is a precommitted, model-selected two-arm execution on the already
exposed B0 PushT root. The saved RUN_02 model consumer is rechecked before the
plan is constructed. It fixes both candidate/iteration identities and all
float32 action bytes before physics; physical outcomes do not select actions.
The `no_composition_sra` label is retained only for ABI compatibility. Its
actual method is joint Gauss–Newton, not historical SRA.

Both arms get separately created environments, the same source root, goal,
warmup 10, common original prefix 15, and selected action suffix of 10.
Exact root, RGB history, raw action history and prefix guards run before
suffix dispatch. The 25-step plan horizon is an upper bound, not an
instruction to step through termination. On native termination or truncation
the branch is censored with its actual step count and missing checkpoints;
the next arm still runs. An engineering failure stops later arms, is retained,
and cannot be imputed or reported as a complete pair.

The CPU-only physical child uses a read-only verified historical-source overlay,
an exclusive session, CUDA hidden, and the existing cooperative lock. The old
preparation is validated as-is before deriving this separately bound plan.
The overlay does not alter the live worktree or certify historical SRA.
Existing public/physics/render caps, 900-second dispatch boundary, RGB224,
resource ledger, native metrics, source checks and RGB byte/pixel hashes apply.
The deadline is not a preemptive kernel timeout. Partial files remain on errors
or abrupt interruption; there is no retry or replacement.

The saved consumer verifies the plan and report identities, event chain,
operation nesting/counts, two branch closures, source guards, literal action
prefixes, real stop flags, metrics, PNG bytes/pixels and artifact inventory.
It preserves shorter trajectories and never fabricates a 25-step endpoint.
An external report digest is required. The consumer validates saved-record
consistency, not independent proof that physics was executed.

This is one known development root, not independent confirmation, equal
FLOPs, equal total cost, mechanism identification or a P1/P2 gate result.
The historical-source overlay and explicit source manifest are not a complete
transitive environment provenance certificate.
