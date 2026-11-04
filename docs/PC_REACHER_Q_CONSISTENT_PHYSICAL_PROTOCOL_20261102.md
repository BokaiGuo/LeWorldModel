# Q-consistent selected prefixes: real-feedback development protocol

Before execution, bind the complete model-only report
`reacher_pc_q_consistent_suffix_20261102_v1/report.json` and its Q source.
On each of its two previously exposed roots execute three arms:
model-selected prefix, zero prefix and the same-bank Q-immediate-greedy
prefix. Select indices only from the sealed report; preserve duplicates and
record zero fallback if selection is null. No outcome may select a prefix.

For every arm, restore the same initial state, execute five prefix actions,
observe its own real RGB and actual action history, and replan a ten-step
suffix using the same inherited goal Q, fixed CEM budget 16 candidates x
two iterations x two packets, four elites and root seed + 5000. Execute
the independently replanned suffix and record its physical qpos distances,
actual action arrays, root/goal/successor identities, stops and prediction
counts. The goal Q is not refit from physical feedback. Encode and predictor
costs, and the additional upstream Q construction and prefix generation
costs, must be separately labeled rather than called equal total compute.

Any exception, early stop or incomplete arm is an explicit failure/censoring,
not an imputed full trajectory; save partial arm status. Simulator state is
used only for restoring the paired initial state and evaluator metrics, never
for choosing actions. This is a new development comparison on exposed roots,
not a common-suffix mechanism intervention, strong-baseline comparison or
independent P1 confirmation.
