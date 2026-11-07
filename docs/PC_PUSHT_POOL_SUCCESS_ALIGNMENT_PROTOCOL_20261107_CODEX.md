# PushT successful pool chains: prediction and physical alignment

On the four exposed tasks in the sealed scored-incumbent first-plan pool,
retain every byte-distinct candidate whose prior physical replay ever succeeded,
plus each task's actually selected chain. This post-hoc diagnostic uses the
prior truth labels only to choose chains to examine, never to select planner
actions. Reuse the recorded action, goal and model prediction from the exact
query; for duplicates use the first recorded occurrence.

Independently reset and replay each chain for 25 environment steps. Check the
initial simulator snapshot and the prior replay's five packet states, and
record termination at every step. At steps 5, 10, 15, 20 and 25 render RGB and
apply the same frozen LeWM preprocessing and encoder. Save full predicted and
actual encoded vectors, target encoding and per-step physical success. Check
that the freshly encoded target matches the goal representation saved with the
model query, and that frozen model weights are unchanged during this run.

For each task compare each successful chain against its selected chain using
last-packet predicted and real-encoded goal costs and physical outcomes;
report per-task numbers, not 96 comparisons as independent episodes. Also
report actual success at each packet and at any step; do not reinterpret an
intermediate physical hit as a terminal hit. When selected chain itself
successfully terminates, report it but do not include self-comparisons.
This is an oracle-only, exposed, post-hoc diagnosis; neither encoding actual
future images nor choosing successful actions is available to the planner.
No gate status can be inferred from these comparisons.
