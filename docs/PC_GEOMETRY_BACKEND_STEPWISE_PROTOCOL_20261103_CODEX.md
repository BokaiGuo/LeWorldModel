# Geometry panel backend replay protocol

Before further prefix outcomes, replay all eight already recorded physical
trajectories exactly, from their original four resets, with no action or
stopping changes. Compare the official shadow executor's MuJoCo 2.1.2
physics (official XML and persistent state, instrumented evaluator) with
the previous diagnostic's MuJoCo 3.1.6 legacy-XML physics. This is a
post-outcome reproducibility check, not selection or an independent test.

Record position, velocity, simulation time, contact geom-name pairs and
distance at every action boundary. Report discrepancies both over all
boundaries and at and after first recorded particle-wall contact in either
engine. At most `1e-8` absolute L2 error at every boundary for both position
and velocity, exact clock agreement within `1e-12`, and identical
particle-wall contact-presence indicators are required for this fixed
action panel to be deemed numerically consistent. Report contact pair
disagreements separately. This finite check does not establish general
equivalence or independently verify RGB images.

Measure the already-fixed `<= 0.25` arrival time in high-level actions,
with 24 actions as the fixed cap and no success at the cap classified as
failure/censored at 24. This is a descriptive development measure only:
the trajectories were already seen, the waypoint and direct methods use
privileged information, and the threshold is not a prespecified independent
performance claim. Do not choose a new between-method horizon from outcomes.
Different XML and binding/runtime versions must be recorded. If the check
fails, treat the 3.1.6 results as a separate environment protocol.
