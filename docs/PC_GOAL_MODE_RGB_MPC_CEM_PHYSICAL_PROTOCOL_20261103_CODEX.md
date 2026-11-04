# Direct-CEM RGB smoke: post-seal physical replay

Replay the sealed direct-CEM warmup and 15 executed actions on the same
MuJoCo 2.1.2.14 backend, scene, start and goal as the goal-mode physical
diagnostic. Verify action, decision, source and RGB image hashes and report
distances at the same four feedback boundaries. This is an exploratory
single-root comparison initiated after goal-mode results were observed;
equal prediction tokens do not constitute equal full compute.
