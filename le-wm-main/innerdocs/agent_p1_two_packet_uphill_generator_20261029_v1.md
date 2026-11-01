# P1 paired two-packet uphill candidate generator

New model-only candidate method, not a physical finding. Start both arms from
the same authenticated original winner raw10 and logits in the same root/task.
At each of eight simultaneous updates, calculate stage Q's native raw gradient
for prefix from root and terminal Q's suffix gradient from that prefix's own
predicted context. Map through the original float32 tanh chart derivative and
the original eta=1 projected-logit proposal, independently for both packets.
The control descends stage Q; the candidate ascends stage Q. Both descend
suffix terminal Q. This differs from fixed-prefix suffix optimization and the
existing selector: it actually regenerates prefix and suffix, not just re-ranks
the old 222 candidates. It does not differentiate terminal Q through prefix.

Each arm scores all nine complete raw10 iterates, then independently replays
the original winner's two packets. A model witness requires both strict
`stage_Q > root_initial_Q` and `terminal_Q < original_winner_terminal_Q`.
If none exists the selected iteration is null. No margin, fallback, physical
score or outcome-based root selection is introduced. Four fixed slots, two
arms and 72 iterate slots remain denominators even on failure; dispatch stops at
first failure, later entries remain not_started and no result is fabricated.

Each arm pays 36 forward calls and 16 scalar reverse calls: 2 initial,
8*(2 AD forwards + 2 fresh score forwards), and 2 baseline replay forwards.
Total across all four slots: 288 forward and 128 reverse. Identical API counts
do not prove equal FLOPs, time, or equal total cost to the old library. Native
model forward and AD must be metered separately in any future host; the core
does not issue actual resource receipts. Candidate contexts and full callback
events are retained; each AD context must equal its preceding scored context.

This executable callback kernel has RAM fixtures, not a production adapter:
source/root/winner provenance, original solver-domain evidence, native ledger,
saved reconstruction, source/model/precision closure and atomic engineering
caps must be implemented and reviewed before MAIN can run a model. Explicit
winner_authority here is only a digest, not an authenticated certificate. No
CUDA, physics, training, P1 gate change or scientific claim is made.
