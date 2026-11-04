# Goal-conditioned RGB MPC and abstention: smoke result

On one reused PointMaze development root, the frozen model compiled three
15-step model plans from real RGB/action histories, executing only the first
five actions each time and recompiling from the next real RGB observation.
Each decision was sealed before its prefix was dispatched.

The selected prefixes were `goal_-1`, `zero` (abstention from a nonzero
prefix), and `goal_+1`. Each decision used 132 five-action prediction tokens.
The second decision confirms that the actual fallback branch executed; its
optimized suffix was not executed because this is receding-horizon control.

Post-seal replay on MuJoCo 2.1.2.14 gave physical goal distances of
`1.745588`, `1.471325`, `1.131087`, `0.814188` after warmup and after each
five-action prefix, respectively. The full replay trace and action/image
hash bindings are saved in
`le-wm-main/results/pc_goal_mode_abstain_rgb_mpc_20261103_codex_physical.json`.

This is a single reused root and has no matched control or independent
confirmation. Only the first five actions of each 15-step plan were executed:
the final model score does not certify the actions' realized outcome. Token
caps do not certify total compute fairness, nor does this smoke establish PC2,
PC3, an independent gain, or a physical frame-to-state correspondence.
