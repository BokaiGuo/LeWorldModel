# Goal-conditioned RGB MPC: post-seal physical replay

Replay the exact 15 warmup actions and 15 actions executed by the already
sealed three-decision RGB integration smoke, using the same fixed PointMaze
MuJoCo 2.1.2.14 scene, start and goal `(1.8, 2.8)` used by the prior
physical audit. Verify image, decision, action, source and scene hashes.
Report distance after warmup and at each five-action boundary, plus the full
physical trace. No candidate selection or tuning uses this replay. This is
one reused development root and has no comparable baseline; it does not
constitute PC2, PC3, independent confirmation, or a success-rate estimate.
