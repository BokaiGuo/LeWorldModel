# PushT action-conditioned mechanics V4

The prior mechanics V2 report has a direction-label inconsistency: its
numeric paired delta equals mechanics minus geometry history, but its recorded
direction names the raw candidate-pair representation.  The old report is not
rewritten.  The forensic audit records this as a report-level rejection.

V4 uses fresh train/calibration/test seed bands and explicitly confirms the
RGB contact/action mechanics representation.  It emits a consistency audit
that binds the reported paired delta to the selected representation.  This is
an offline two-action identifiability replication only; it does not test a
planner, pose closure, or a world-model transfer certificate.
