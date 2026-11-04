# Fixed-bank decision regret: retrospective diagnostic

Use the already exposed 36 same-action comparisons saved in
`pc_pointmaze_lookahead_same_action_20261103_DEV01.json`; verify its published
SHA-256 before analysis. For each of four roots select the smallest score
among all nine fixed candidate chains separately using open prediction,
refreshed prediction and the actual RGB-derived encoded cost. Resolve ties by
the recorded candidate index. The selected action's physical endpoint
distance minus the best physical distance in that same root's bank is
descriptive fixed-bank regret.

Actual-encoding selection and physical oracle are hindsight diagnostics:
neither is an admissible deployed planner. All scores and outcomes were
already exposed, the four roots are development data, the candidate bank is
not exhaustive, and there is no uncertainty claim or method selection.
Do not use this analysis to alter the historic experiment or admission gate.
