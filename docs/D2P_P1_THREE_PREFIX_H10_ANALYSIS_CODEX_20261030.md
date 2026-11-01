# H10 three-prefix saved analysis

The analyzer first invokes the frozen H10 saved consumer. It accepts only a
complete 24-by-42 run with source, actions, trajectory, resource, terminal
and censor checks passed. It then rechecks the pinned preparation and shard
report hashes, retaining each row's accepted probe-step count, native stop
flags or original censor evidence. An incomplete or invalid run has no
accepted analysis; no missing endpoint is imputed, including a native stop
at step 10.

For every fixed logical group and probe the output contains the selected,
zero and greedy status and endpoint object XY displacement from that role's
prefix checkpoint, wrapped angle displacement, Euclidean object-position
error to the common goal and absolute wrapped-angle error. Pairwise fields
are first role minus second. A separate same-role response compares each
suffix against that role's zero-suffix endpoint: XY and wrapped-angle response
and reduction in position/angle error. Pairwise response fields compare these
within-role contrasts. If either needed endpoint is missing, the corresponding
comparison is null. The zero baseline may itself be stopped or censored.

The output preserves all 336 group/probe records and 1008 statuses, including
duplicate-prefix identifiers and identical full-group identifiers. It performs
no winner selection, retry, replacement, significance test or P1 promotion.
The selected original-tail probe and known roots are post-outcome exploratory;
the result is not independent confirmation or equal-budget evidence.

The analyzer is new and intentionally not bound into the running execution's
frozen sources. Its output records its own code hash and the pinned prepare and
run-report hashes. Saved verification is not an independent physical replay
and cannot authenticate unrecorded host or solver-cache state.
