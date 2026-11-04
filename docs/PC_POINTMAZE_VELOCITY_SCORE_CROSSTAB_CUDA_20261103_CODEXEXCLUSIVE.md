# Fixed-action cp5 velocity × scoring CUDA diagnostic

Use the complete fixed bank of 384 previously exposed candidates in order:
roots 1884–1887, random then spectral, 48 per bank. Preserve each candidate's
actual cp5 RGB, two original action packets, target RGB, initial task mask and
three previously saved cp5 state inputs. Predict only its fixed second packet.
No optimizer, policy, new physical execution or training is called.

Cross cp5 input velocity zero, causal RGB velocity and legacy simulator qvel
with predicted visual and predicted proprio goal scores. qvel is a strictly
offline privileged diagnostic and must never be passed to a planner. Goal
position is estimated solely from observed goal RGB and fixed scene metadata;
goal velocity is explicitly zero, not simulator ground truth.

Re-encode each saved actual cp10 RGB, store the full actual visual encoding,
and calculate same-mask actual RGB goal cost. Require all 1,152 recalculated
visual scores and visual errors to match their respective sealed receipt at
rtol 1e-4, atol 1e-6; require actual RGB goal cost to match its receipt and
the frozen model parameter hash to match before/after. Only after these checks
are complete, load physical distances and calculate pairwise ranking reversal
and tie counts for each velocity/score versus actual RGB cost and versus
physical distance; also report actual RGB versus physical. No values are
imputed and no observed physical outcome selects an action or input.

Plan binds sources, numerical code and environment inputs by SHA-256, checks
source receipt lineage, fixed groups and shapes, stores per-group numeric
outputs, event journal and final model hash. Capture failures without
overwriting a prior output. Use host CUDA only with the existing cooperative
lock and no CPU neural fallback. Report actual counts, checkpoint/source hashes,
and numerical error. This is an oracle replay of already exposed dependent
candidates; full actual cp5 observations and simulator qvel are not available
to a deployable planner. It is neither independent evidence nor a P1 gate or
equal-total-compute comparison. Image-derived velocity is a quantized 0.05 s
average; legacy qvel is instantaneous and RGB/state alignment is not certified.
