# End-to-end RGB causal-feedback planner: development protocol

Before execution fix four exposed roots 1884–1887 and the prior frozen
checkpoint, noise, frame-0 reset, frame-10 RGB goal and source hashes.
Recompute the initial RGB-position/zero-velocity context and initial task
mask. Independently recompile each random-CEM and spectral-CEM arm using
three generations of 16 two-packet candidates. Spectral CEM adds two JVPs
and one SVD for its four bounded injected candidates; do not call this an
equal-total-budget comparison. Require each recomputed initial plan and cp5
image to match the earlier sealed RGB-position execution.

Execute the chosen first packet, obtain real cp4 and cp5 RGB, derive
causal velocity from their position difference over 0.05 seconds, then
recompile the tail with 3×16 one-packet CEM predictions and the same
initial task mask. Execute the selected tail. All decision inputs are RGB,
known executed actions, fixed scene metadata, and frozen model; simulator
state is used only to reset and evaluate, never to select. Save noise,
actions, initial and tail traces, RGB observations, model/structure
calls, measured time and hashes. Both policies see the same RGB opportunity.

Only after all selections execute, separately replay all action sequences;
require the terminal RGB and physical position to agree with the execution
and position backend. Report terminal distance and strict `<0.5` endpoint
check, not online native success. No early-stop imputation. The former
fixed-prefix causal-tail experiment must be exactly reproduced for actions
and RGB if initial binding, mask, noise and context match.

This is a four-root development check, not independent confirmation.
It cannot establish that spectral structure helps, reconfiguration exists,
or P1 is passed. Training membership and historical root exclusion are
uncertified. A stronger fair comparison must give a competing planner access
to the resources consumed by spectral structure and include actual total
cost, independent roots and prespecified statistics.
