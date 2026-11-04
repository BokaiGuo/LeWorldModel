# Wall-unicycle non-oracle diverse beam: development result

The standalone cost-pruning implementation exactly matched the real
unmodified `AtlasSearch` complete programs, selected index, replay,
compiler counters and backend counters on all three exposed seeds. The
predeclared diverse policy retained the current-cost best node plus the
model-state-farthest node, without injecting the privileged waypoint chain.

| Seed | Original Atlas error | Diverse beam error |
| ---: | ---: | ---: |
| 20261103 | 0.505090 | 0.513514 |
| 20261104 | 0.503323 | 0.513514 |
| 20261105 | 0.502205 | 0.503991 |

Neither policy reached the other side; this particular diversity rule did
not recover the known feasible detour and was worse on all three repeated
searches. Both search policies used 3,320 state transitions, 316 compiler
rollouts and 31 Jacobians per seed, including 496 FD rollouts. The diverse
rule additionally computed 162 state distances per seed, so the exact
transition counts alone do not certify equal total compute.

This is one already exposed analytical task, not a learned-model result,
statistical replication, or proof that all diverse beams are ineffective.
The state scaling `[1,1,0.3]` is a task-specific development choice. The
successful random waypoint control remains a separately executed, more
structured candidate-generation baseline; its privileged exact state and
task-scale sampling range do not establish RGB-only performance.

- Protocol: `docs/PC_WALL_UNICYCLE_DIVERSE_BEAM_PROTOCOL_20261103.md`
- Runner: `le-wm-main/experiments/spectral_reachability/pc_wall_unicycle_diverse_beam_20261103.py`
- Result: `le-wm-main/results/pc_wall_unicycle_diverse_beam_20261103_DEV01.json`
- Regression set, including deterministic tie handling and unmodified
  Atlas parity: **24 passed**.
