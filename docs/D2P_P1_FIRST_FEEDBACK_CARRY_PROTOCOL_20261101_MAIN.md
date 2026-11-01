# First-feedback carry exploration

Fixed roster: both existing roots, all four original arms, no physical-outcome
filtering. Frozen checkpoint, host CUDA only. This is NOT independent confirmation.

Identified deployment gap: previous carry implementation starts at cp10, not cp5.
This variant compares the full original plan's remaining H2 tail at cp5 against
all eight newly optimized candidates truncated to H2, scored on actual RGB and
actual normalized action history. At cp10 compare H1 tails; at cp15 the selected
plan is exhausted and all eight fresh candidates receive H1 scoring. No zero
padding, copied physical winning suffix or extra unobserved actions.

Tie retains carry. All arms use identical rules. Added calls: cp5 18 action encodes,
18 predictions/samples, 9 replay and selection calls; cp10 9,9,9,9,9; cp15 8 each.
Original H3 optimization remains paid, so this tests deployment/horizon selection,
NOT optimized equal-compute superiority. Extra scoring is in node reservation and
metered dispatch; mismatched accounting aborts.

Preserve full selected plans, issued first-packet identity, current RGB hash and
raw-history normalization. Record all failures, original success and real states.
This change cannot itself cure selected==greedy initial prefixes or establish
non-greedy capability. Interpret it as a targeted test of feedback replacement.
