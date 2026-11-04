# Two-root fixed-bank joint versus conditional development

Freeze case order `[0, 2]` from the four-case saved cp15 panel, using its
`two_program_transfer` case for each of its two root/task identities. Cases
2 and 3 have the same context and prefix and are not distinct roots. Both
methods and both roots receive the same pre-existing B0 bank of 16 float32
10×2 actions in the same order. The bank was exposed before this protocol and
is not adapted to root 1; root 1 use is out of distribution. Context and goal
are inherited model data; no physical outcome is used for selection.

Each root independently executes the same native model terminal-MSE objective,
same projected-gradient optimizer (eight updates, nine scores), same bank,
conditional 192×10 and joint 192×20 Gauss–Newton initializer. The joint arm
is not historical SRA despite its legacy internal label. Attempted calls are
subject to the existing per-root caps and synchronized timing/transition token
ledger. The root deadline is 900 seconds and session deadline 2100 seconds,
checked before dispatch, not preemptive. Files are exclusive, a 128 MiB
session ceiling is checked before each ledger event with a final reserve.
An engineering failure stops the session, preserves partial journals and
reports the remaining roots as not started; no retry or imputation.

Each fully audited root produces a fixed-winner handoff with exact common
prefix and selected suffix bytes, root/task/context hashes and model score.
The output consumer verifies both journals, candidates, scores, winners,
actions, sources and handoff reconstruction. Both handoffs explicitly have
`physical_ready=false` until a separately sealed physical plan is built from
the consumed model run. The separate physical bridge revalidates each old
preparation inside the read-only historical source overlay, and applies its
root and observed RGB/prefix guards to the fixed model-selected actions.
Physical horizon per branch is warm10, guarded prefix15, suffix10; native
stops are censored and not padded, engineering failures remain visible.
The physical runner and consumer require review and separate authorization;
neither is invoked by the model launcher or by preparation.

The two cases and fixed horizon were chosen without inspecting new outcomes,
but these are exposed development roots and old panel/model provenance is not
a full transitive environment closure. The new source manifest and raw asset
hashes are not proof of historical equivalence. No historical SRA claim, equal
FLOPs, equal total costs, independent confirmation, mechanism result, or P2
gate admission is made.
