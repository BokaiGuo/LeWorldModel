# First-planning-pool physical decomposition

This is a retrospective diagnostic on the 40 already exposed TwoRoom roots
of the completed 16-score RGB feedback experiment. It does not change that
experiment's prespecified stopping decision. For each task reconstruct the
three first-round prefixes, the common zero and two queried probes, and the
two additional tail proposals from the sealed seeds. Verify common first-round
inputs and scores across arms before doing physical evaluation.

Replay all 15 prefix/tail combinations per task from the dataset reset state
for 25 steps, including after an environment success solely to obtain matched
endpoint distances. Check whether zero-reference motion changes physical
position after each prefix. Compare predicted hold-minus-probe effects with
true endpoint-distance differences, treating `1e-6` as a tie threshold in each
respective scale; report ties separately, per task and in aggregate. Physical
distance and state are evaluation-only, not RGB planner inputs.

For each prefix, label the independent two-tail bank useful only if the
initial state and prefix have not already succeeded and at least one bank tail first achieves
success after the prefix. Report per-task useful-prefix recall for the sealed
retention arms and their paired comparisons; report the full denominator and
prefix-already-success counts. This tail bank was generated independently of
the probes but was scored after retention in the original run; replaying all
prefixes is a privileged diagnostic and does not constitute deployable search,
success-rate evidence, or an unbiased independent confirmation.
