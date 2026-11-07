# Shared-prefix full-pool rescoring on exposed PushT tasks

The previous diagnostic found that some successful first-plan chains share the
actual selected first ten environment steps, while the prior two-chain rescore
test did not examine the entire already-scored pool. Test whether retaining and
rescoring *all* byte-distinct tails with this same prefix changes the selected
physical outcome. Use the four sealed tasks and original first-plan candidate
pool; do not filter candidates using physical success, model ranking, or later
observations. Include the original selected chain. All hypotheses, timings,
and results are development-only and post-hoc.

Use the selected chain's actual RGB encoding at ten steps as the shared
observation and the original target encoding, together with each candidate's
remaining three normalized action packets. Reuse the existing one-frame
conditional rollout; no later observations or physical labels enter scores.
The real feedback arm would first execute this same ten-step prefix and incur
the cost of retaining and reevaluating the entire shared-prefix set. Report
all candidate count, model packet evaluations and wall time, physical success
of the argmin action determined before opening physical labels, and ranks of
successes afterward. Preserve all tasks including the task without a successful
shared-prefix tail. Baseline is the original scored-incumbent choice, not an
oracle or free access to success labels.

The selected ten-step observation is from a replay of the actual first chain;
the recorded run actually dispatches a longer prefix. Thus this is a valid
conditional counterfactual on an available observation, not an actual
feedback-policy rollout. A 10-step observed context can be shared only when
both full action prefixes match exactly; explicitly verify this in code and
record byte identities and query shapes. No effect on gates is claimed.
