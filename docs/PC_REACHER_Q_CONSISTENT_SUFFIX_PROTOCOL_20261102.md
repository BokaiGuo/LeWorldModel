# Fixed-prefix, Q-consistent suffix optimization: development protocol

Before this run, use all prefix candidates in the sealed
`reacher_pc_goal_q_selection_20261102_v1/report.json` on both known roots,
including the zero candidate and both refined candidates. The roots and
historical physical outcomes are exposed; this is an outcome-informed
development experiment, not confirmation. Do not filter, add, reorder or
physically select prefixes based on outcome.

Retain the saved encoder-derived goal Q from that report (including its
original 192 VJPs and FD receipt). Verify the source hash, root/goal RGB,
history identity, action normalizer and frozen model parameter hash. At each
predicted prefix successor, run the same bounded two-packet nonlinear model
CEM for every candidate: 16 samples, two iterations, four elites; use the
same root-derived random seed for every prefix, initial mean zero/std one,
include the mean and select the best final-iteration sample. First-stage and
terminal costs both use the same goal Q. Every sample is evaluated by model
rollout, not simulator state. Before optimizing each prefix, require its
recomputed immediate Q cost to match the saved score within relative 1e-4
and absolute 1e-6, otherwise fail. Reuse the previous fixed zero-reference
selector and its existing margin; no eligible candidate means null.

Save every candidate's optimized suffix, predicted first/final Q cost,
selector result, operation counts and source hashes; include Q construction
and upstream candidate costs as inherited rather than treating suffix
evaluations as total compute. This tests a previously untested consistency
of suffix objective, not a new confirmation gate. No physical execution,
real reconfiguration, equal-total-budget advantage or P1 success is implied.
