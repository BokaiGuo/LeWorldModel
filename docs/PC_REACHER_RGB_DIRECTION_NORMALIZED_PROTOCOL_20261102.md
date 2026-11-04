# Dimension-normalized RGB direction CEM: development protocol

On the same two exposed roots and all 13 fixed prefixes per root, use the
existing real cp5 RGB/history feedback and the same 16-sample, 2-iteration,
4-elite, two-packet CEM as the archived L2 and direction experiments. Define
`u = (z_observed - z_goal) / ||z_observed - z_goal||`, fail closed for zero or
nonfinite direction, and evaluate a predicted terminal `x` by

`(sum((x-z_goal)^2) + 0.1 * dot(x-z_goal, u)^2) / latent_dimension`.

This makes the direction coefficient 1.1 times, rather than 20.2 times, the
perpendicular coefficient. This is a development-only intervention chosen
after seeing the prior negative results, not a preregistered confirmation,
and does not change any archived result or gate.

Use frozen model, real RGB and actual actions only for planning. Instrument
the actual suffix model `encode` and `predict` calls, predicted sample
counts, and synchronized suffix planner wall time. Instrumentation excludes
upstream candidate construction, simulator restoration and execution and is
not full-resource certification. Compare all 26 physical outcomes with the
same-prefix L2 and strong-direction runs only after verifying matching cp5
RGB, goal and action history. Treat all 13 prefixes within a root as
dependent. Outcome truth is not fed into the planner.
