# Saved-prefix coverage diagnostic: fixed post-hoc protocol

Question: how close in raw action space were the recorded v13/v14 response
candidate and retained prefixes to the *selected* same-bank CEM complete
chains from the already stopped v14 E0 panel? The reference chains and all
response layers are already saved. This is a post-hoc descriptive analysis,
not a new method evaluation or a reopening of the panel.

Use all eight paired roots and both response arms, at each 5-, 10-, and
15-step layer. For each root and layer compute the minimum coordinatewise
RMS action difference to the CEM selected prefix among (a) all generated
candidates and (b) the indices retained by that layer. Record candidate
count, retained count, nearest candidate and its retention status, and
the CEM final score. Report the two already observed CEM-success roots
separately (distance below the existing 0.05 threshold), as well as all
eight; do not infer a significance test from this development panel.

The source audit and result hash must match, all task and root pairings,
action bounds, layer lengths, and retained indices must be checked. Fix the
metric and output schema before computing. The single output is written
exclusively to a new path, without modifying source artifacts. No planner,
predictor, simulator, or physical rollout is called, and no search budget is
added. Any missing or malformed source stops the analysis.

These distances use a post-hoc reference and do not establish whether a
different prefix has a successful continuation; raw-action distance is not
a dynamics-aware metric. A close prefix is not proof of retained value,
and a distant prefix is not proof that pruning or candidate generation
caused failure. CEM results are a comparator and must not be injected into
the method or treated as PC2 evidence.
