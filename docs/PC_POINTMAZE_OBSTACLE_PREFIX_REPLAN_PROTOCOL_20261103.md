# Obstacle prefix followed by common RGB feedback optimizer

On all four exposed roots, execute the fixed first five actions from the
sealed spectral plan, random plan, or a zero prefix. Then, for each prefix,
independently run the same random-initialized AP-compatible CEM at steps
5, 10, 15 and 20, using its own actual RGB and causal RGB velocity. Each
decision uses the same seed for all prefixes at a given root/time, and
50 scored candidates. Execute the resulting five actions and do not stop
early. The target, frozen model, action bounds and normalizer are the same
as the source. The first prefix was generated previously and its different
generation costs are not matched; this is a conditional successor-planning
diagnostic, not a full-method equal-total-compute experiment.

All 12 trajectories and RGB histories are saved; compare original random
prefix frames and the complete random trajectory with the source receipt,
and replay all final RGB frames in fresh bridge instances. Use the audited
position backend for endpoint and prefix evaluation; record separate
planning time and candidate counts. Freeze parameters and use CUDA only.
No true simulator state enters the planner. The target state is used for
goal rendering and evaluator only.

The roots, source plans and prior physical outcomes are already exposed.
Their outcome cannot serve as independent confirmation. None of the saved
spectral prefixes satisfies the no-immediate-distance-improvement condition
on the previous measured panel; no P1 or mechanism claim follows even if a
conditional endpoint comparison is favorable.
