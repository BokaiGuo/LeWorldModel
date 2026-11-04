# Two-start CEM and plan-time alignment: exposed-root follow-up

The two-start CEM baseline completed both previously exposed roots, with the
same frozen prefix, five feedback opportunities and execution horizon as the
earlier comparison. It spent its extra work on a second independently seeded
CEM search and chose the better predicted final score. Its physical outcomes
were obtained by fixed-action replay; all 18 RGB frames per branch reproduced
exactly, repeated 2.1.2 replay states agreed, and 2.1.2 and 2.2 endpoint
positions agreed exactly.

| Root | Single-start CEM | Spectral CEM | Two-start CEM |
| --- | ---: | ---: | ---: |
| r8343_a0_p2 | 0.762595 | 0.705294 | 0.828740 |
| r8345_a0_p2 | 1.708919 | 1.603303 | 1.752083 |

Values are physical final position distances, lower is better. This
development sample shows the earlier spectral advantage persists against
this particular extra-search baseline, not that it beats all stronger
optimizers or has certified equal-resource efficiency. Per decision,
single-start used 50 scored samples/four batches, spectral used the same
plus one response forward/two reverse vectors/one SVD, and two-start used
100 samples/eight batches. Recorded mean decision wall times were 0.109 s,
0.117 s, and 0.201 s respectively. These observations include different
logging and warmup costs and are neither synchronized benchmark distributions
nor calibrated FLOP or energy budgets. There are only two exposed roots.

## Five-step prediction versus one-step execution

All 30 decisions across the six branches saved a five-action plan and
executed its first action exactly. Among the 60 subsequent planned actions
whose corresponding execution times occur within the recorded runs, all
60 differed from the later replanned actions by more than 1e-6 in raw-action
L2 norm. Thus a terminal score for the original five-action plan is **not**
the prediction of the subsequently executed five-action closed-loop chain.
For decisions 1–4, the five-step prediction endpoint also lies beyond this
run's fixed terminal time. Comparing decision 0's planned score to the
actual final RGB score is descriptive only, because the executed tail changed.
This behavior is consistent with receding-horizon planning; it is not by
itself an implementation bug or a same-action model error.

The [receipt](../le-wm-main/results/pc_shadow_multistart_replay_timing_audit_20261103_CODEX.json)
contains per-decision tail comparisons and physical state trajectories.
[Protocol](PC_POINTMAZE_SHADOW_MULTISTART_DEV_PROTOCOL_20261103_CODEX.md),
[replay protocol](PC_POINTMAZE_MULTISTART_REPLAY_TIMING_AUDIT_PROTOCOL_20261103_CODEX.md),
and [runner](../le-wm-main/experiments/pc_shadow_multistart_dev_20261103_codex.py)
were saved separately from historical records. Twelve targeted tests passed.

Neither this control nor the prior RGB result changes the reconfiguration
evidence boundary: both start from an identical historical prefix. Future
experiments must choose the prefix before execution, test common suffixes and
independent replanning, and compare all methods under a prespecified shared
resource policy on independently sourced roots.
