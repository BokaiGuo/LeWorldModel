# TwoRoom success-contract audit

This is a retrospective integrity audit of the previously exposed roots 10–49,
not a new performance experiment. Check the recorded reference states against
the current dataset, check each saved feedback step against the installed
environment's strict `distance < 16` rule, and report initial-success
stratification without removing cases. Audit both deadline CEM arms and all
four timed-prefix arms. Reject missing roots, mismatched rows, invalid actions,
inconsistent termination, truncation, or discrepancy between recorded and
derived ever-success. Record input and environment-source hashes. State
coordinates are used only for this evaluation audit, not by the RGB planner.

The feedback experiments call `env.unwrapped.step`, impose their own 25-step
limit, and use a local reset/render protocol. These checks do not establish
official benchmark equivalence, dataset independence, or generalization.
