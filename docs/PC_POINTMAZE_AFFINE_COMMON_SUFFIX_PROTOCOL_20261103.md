# Fixed-prefix common-suffix physical intervention

This is a development diagnostic of the eight exposed roots 1888–1895 in
the existing affine-score spectral-injection and random-CEM runs. Use their
saved, model-selected five-step first packets as `spectral_cem` and
`random_cem`; add a five-step all-zero prefix. These are existing progress
planner prefixes, not certified non-greedy reconfiguration programs.

For each root and each of the three prefixes, execute the same seven
predeclared constant-action suffixes: zero, `[+0.5,0]`, `[-0.5,0]`,
`[0,+0.5]`, `[0,-0.5]`, the random arm's saved cp5 replanned action and
the spectral arm's saved cp5 replanned action. Execute each suffix for ten
high-level steps. Both saved planned tails are included for *every* prefix,
with their root and source bindings; do not retrospectively choose suffixes.
Their inclusion probes the execution-relevant scale and does not make them
a fixed action shared across roots. Record endpoints at prefix+5 and
prefix+10 as well as cp5. All trajectories reset to the same saved root.

Use the audited MuJoCo position backend with the same velocity clipping,
action bounds and high-level step clock as the original experiments.
Record physical position and velocity at start, cp5, suffix+5 and suffix+10.
The goal position is available only for offline physical evaluation, not for
candidate selection. Score Euclidean goal position error at start, cp5 and
endpoints. Report change at cp5 from initial error and from the zero-prefix
cp5 error, and absolute final error. At each suffix horizon also report
position response and goal-error change relative to the *same prefix's*
zero-suffix endpoint, which controls its passive continuation. Record all
seven suffixes, not the retrospectively best suffix.

A non-immediate-position-progress flag means cp5 error minus start error is
at least `-1e-8`; it is a diagnostic definition, not a replacement or
relaxation of any historical gate. Prefix effects on common suffixes do not
establish a superior individually optimized future, and no non-greedy claim
follows if the flag fails. These are previously exposed roots; trajectories
are paired and dependent. Do not treat suffix rows as independent roots.
Full equal-resource optimizer baselines, closed-loop replanning and source
certification remain separate requirements.

Bind source reports, score-replay receipt, dataset files, backend contract,
script, suffix definitions and outputs. Independently replay the saved
historical own-prefix+own-tail at five steps and require agreement with
the stored terminal position. Fail closed on shape, bounds, source drift,
incomplete roster or replay mismatch; retain any negative outcomes.
