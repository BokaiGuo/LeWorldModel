# Goal co-metric selector over fixed feasible-prefix bank

This is a new, outcome-informed development hypothesis, not independent
confirmation. Take the entire fixed candidate bank and every candidate's
already model-selected CEM suffix from
`reacher_pc_feasible_prefix_development_20261102_v1/report.json`.
Reconstruct and verify root/goal/history and the frozen checkpoint. For each
root derive the previously specified goal-only encoder co-metric Q with
full 192-dimensional CUDA Jacobian and its unchanged finite-difference checks.

For each fixed prefix, run the frozen predictor from real root context to its
imagined successor and then through the saved two-token suffix. Score both
prefix and final latent residual using the same Q. Relative to bank index 0,
require prefix cost at least 1e-6 larger and final cost at least 1e-6 smaller.
Select the eligible candidate with minimum final Q cost, ties by original
index; return no selection if none qualifies. This changes neither saved
prefixes nor suffixes and does not optimize suffixes for Q. No physical state
or outcomes enter this selection. Record all scores and receipts before any
new physical execution.

Record goal Jacobian and encoder cost, one prefix token and two suffix token
predictor samples per bank candidate, plus prior candidate-generation and
suffix costs. It is not an equal-budget comparison. If a genuinely new prefix
is selected, a separate development physical execution can use the same
real-RGB feedback and suffix optimizer; otherwise avoid treating replay as
new evidence. Common-tail mechanism and independent confirmation remain
mandatory for P1.
