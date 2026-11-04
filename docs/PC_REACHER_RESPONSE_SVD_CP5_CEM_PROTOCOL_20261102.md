# Response-SVD all-prefix cp5 CEM development panel

The already-exposed two-root SVD bank has no strict selected prefix. Evaluate
all 13 prefixes per root, not a post-outcome subset: the neutral prefix and
the twelve SVD proposals. Each arm restores the same root, executes its fixed
five-step prefix, acquires the actual successor RGB and action history, then
independently optimizes and executes a ten-step suffix using the same CEM
configuration (16 samples, two iterations, four elites and root seed +5000).
The existing two-stage executor also executes a five-step parameter-zero
control from the restored root. Planning receives RGB, goal, action history,
and frozen model only; simulator state is for restore and evaluation.

The preexisting common-tail outcomes were already opened; this is exploratory
development, not blinded confirmation or a new strict selector. Bind and
authenticate all 26 prefixes, the action chart and warmup before dispatch;
retain each branch's actual actions, planner contexts, stops, results and
errors in a durable journal. A root or arm failure cannot be imputed as a
success or dropped from the denominator. Use host CUDA and the cooperative
lock, fixed model hash and postchecks. Each suffix is allotted 32 model
candidates, but upstream SVD, observation encoding and complete compute are
not yet equalized or fully metered. The panel does not show an equal-total-
budget comparison, independent generalization, or P1 admission.
