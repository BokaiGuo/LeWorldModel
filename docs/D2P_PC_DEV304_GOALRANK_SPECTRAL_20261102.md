# Dev304 goal-contribution spectral modes: model and physical development run

The new run changes the sealed dev304 spectral compiler's mode ordering from
absolute damped action coefficient to damped goal-objective contribution.
It uses the same two known-root feedback contexts, anchors, identity latent
metric, action bounds, radius 0.15, two modes with both signs, and the
same seeded random controls. Selection is minimum nonlinear predicted goal
latent MSE. This is a post-outcome development experiment; no physical
outcomes entered selection. It is not a fresh-root confirmation.

Host CUDA ran the frozen model and its before/after hashes matched. All 18
planned candidates were physically executed, without execution errors, to
the ten-step tail horizon; none achieved native success. The saved verifier
checks the role and source bindings, model-winner and goal-mode rankings,
action bytes, prefix replay, finite-difference arithmetic and 252 RGB frames.
The checked finite-difference relative errors at epsilon 0.001 were
0.000817 and 0.002020; these do not certify radius-0.15 linearity.

| Feedback context | Candidate | Model latent MSE | Native reward | Object position error | Object angle error |
| --- | --- | ---: | ---: | ---: | ---: |
| original | anchor | 1.566485 | -312.309 | 207.536 | 2.7110 |
| original | selected spectral | 1.547500 | -299.061 | 204.425 | 2.7667 |
| coherent | anchor | 1.612874 | -295.760 | 205.792 | 2.9933 |
| coherent | selected spectral | 1.597999 | -299.909 | 203.127 | 3.0124 |

Both selected actions improve object position error relative to their own
anchor but worsen angle error; reward improves in one context and worsens in
the other. The two contexts share a root and have different prefixes, so
they are not independent samples. These results do not establish pose
control, a reconfiguration mechanism, equal-total-budget superiority,
independent generalization, or P1/PC2/PC3 admission. The complete upstream
planning cost remains uncertified.

Artifacts relative to `le-wm-main/results/`:

- model contract SHA-256
  `48017771016077c1d7860759c7104f46581919471f63fd169ec0fec3e94da402`;
- model report `p1_dev304_goalrank_spectral_20261102_RUN_DEV01/report.json`,
  SHA-256 `d5832908ddcc9d298b4980dd7ec1da313c92d2002e04dbcc649c00a449d4b323`;
- physical contract SHA-256
  `2a6f837213855e78da0645bea9f102e08716174324f40f38839f128ccf067f6d`;
- physical report `p1_dev304_goalrank_spectral_physical_20261102_RUN_DEV01/report.json`,
  SHA-256 `08a42bd80b0cca6a76a1aa481ef0f5ad876ba5818c838b37894f4bb8386cec3f`;
- saved verifier output `p1_dev304_goalrank_spectral_saved_20261102_DEV01.json`,
  SHA-256 `fe8e4d4fb8e95d3c3bd34f2b9d81a415a9fb158b7f5da76defb3ee4603ab3468`.
