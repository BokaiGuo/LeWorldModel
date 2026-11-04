# Moving-history cp5 independent replanning development diagnostic

After the four model choices and fixed-chain outcomes are already exposed,
compare three roles per root: selected, zero and seeded `random_0`. Retain
duplicate roles. Replay the identical 15-action warmup and the role's
sealed five-action prefix on the official 2.1.2 RGB bridge, recording
every frame/action and matching all warmup RGB hashes. At cp5 retain
actual RGB observations after warmup steps 10, 15 and prefix step 5, each
with its own immediately preceding frame for RGB velocity estimation.
Bind the actual warmup packet steps 10–14 followed by the actual prefix
packet, in chronological order and with the frozen dataset normalizer.
The real cp5 RGB and action history, not a predicted cp5 state, feed the
planner.

Each role uses the same frozen model, target, projected cost, window three,
10-action suffix horizon and two-start CEM with three generations of 16
samples and four elites, plus two final evaluations per start: 100 model
candidate scores. Root-specific seeds `2026110360 + root_index` give
common random numbers across roles. Save complete suffix searches and
plans before executing any suffix. Afterwards replay each warmup/prefix,
require exact cp5 RGB match, then execute its own suffix through the RGB
bridge and save all frames/actions. Evaluate physical terminal position
only in a subsequent post-seal native replay.

This is a development diagnostic after exposure, not a frozen independent
comparison, not a replacement for same-suffix interventions, and not a
fair whole-method budget comparison: initial candidate generation cost,
prefix search, encoding, initialization and other baseline planning still
need full accounting. The common suffix-family intervention is a separate
saved report. No simulator state is used for action generation.
