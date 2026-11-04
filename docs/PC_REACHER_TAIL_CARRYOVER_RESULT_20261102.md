# Reacher feedback tail-carryover development result

The new runner replays the archived full-CEM first packet, verifies exact raw
first-action and observed-successor identity, and includes the archived
two-packet plan tail in the real-RGB successor CEM candidate batch. The
specified candidate count and feedback opportunity are unchanged. Both roots
selected the carried tail by successor model cost.

| Root | Original CEM feedback final qpos | Tail-carryover final qpos | Guarded spectral selected | Zero prefix |
| --- | ---: | ---: | ---: | ---: |
| 20263160 | 0.943407 | 0.558885 | 0.558267 | 0.546408 |
| 20263161 | 0.721176 | 0.795165 | 0.619442 | 0.598310 |

Lower is better. Carryover improved the first root by 0.384522 but worsened
the second by 0.073990. On the first root it nearly removes the apparent
advantage of the guarded-spectral candidate over full CEM; it remains worse
than zero on both roots. All compared methods operate on known development
roots; this is not an independent performance or mechanism test.

The original final-iteration CEM excluded the archived first-stage plan's tail
from its successor candidate set. This is an implementation/design limitation
in that baseline, not evidence of a universal defect or of the cause of all
previous failures. The carryover costs 32 suffix rollout evaluations / 64
horizon-weighted model prediction samples, replacing one random candidate per
batch; the sampled populations therefore differ. Source first-stage planning
calls are charged, but its wall time is not included in this runner and
full-resource equality remains uncertified.

This protocol was created after the old outcomes were known. Model score and
physical outcome disagree in direction on the second root; no outcome was
used by the selector. Six tests of the associated first-plan, incumbent and
carryover utilities passed, but those tests do not certify statistical
advantage, complete resource accounting or independently verified generalization.

Artifacts:

- `le-wm-main/results/reacher_pc_tail_carryover_20261102_v1/report.json`
- `le-wm-main/experiments/spectral_reachability/run_reacher_pc_tail_carryover.py`
- `le-wm-main/experiments/spectral_reachability/reacher_pc_tail_carryover.py`
- `docs/PC_REACHER_TAIL_CARRYOVER_PROTOCOL_20261102.md`
