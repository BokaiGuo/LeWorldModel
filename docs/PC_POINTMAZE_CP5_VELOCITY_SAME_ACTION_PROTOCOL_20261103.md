# Fixed-bank cp5 velocity ablation

Development diagnostic fixed before execution. Replay all 384 sealed plans
in the eight previously exposed candidate banks, with original float32
source-state conversion and the same ten actions. Render cp4, cp5, cp10 and
capture raw MuJoCo cp5 `qvel`. Require every cp5 and cp10 RGB and the full
cp10 state to be byte/exact-equal to previously sealed data; fail closed.

For the exact same cp5 RGB, goal, fixed initial task mask and original tail
action, compare RGB position with: zero velocity; causal cp4-to-cp5 RGB
position difference divided by 0.05 seconds; and raw cp5 MuJoCo velocity.
The third arm is a privileged diagnostic only and must never drive a deployed
policy. Encode actual cp10 RGB and report per-bank masked MSE and ranking
against the same true RGB, and physical distance at each diagnostic argmin.
Require the zero-velocity score and MSE to reproduce the previous receipt
within rtol 1e-4, atol 1e-6. Record image/state arrays, action binding hashes,
calls and model parameter hashes. Abort rather than silently skip a candidate.

Each prefix is physically observed separately, so even the causal-RGB arm's
full-bank argmin is an oracle, not a deployable selection policy. The raw
velocity is captured directly from the legacy integrator; do not assume the
rendered image is temporally aligned with that state. Image differences are
quantized and causal velocity is an average, not instantaneous velocity.
These post-outcome adaptive banks are dependent and do not establish P1,
independent generalization, or an equal-budget performance improvement.
