# Causal RGB history and ten-step suffix replanning

The new predeclared development run used the same nine sealed prefixes and
four exposed roots as the earlier five-step suffix experiment. Every prefix
received 50 three-packet model evaluations before selection; its real
post-prefix RGB then received its own 50-sample, ten-action suffix
optimization, which was physically executed. All 36 branches completed.
The generalized CEM exactly matched the historical kernel on a one-packet
test; the ten-step kernel used the same population, iteration count, elite
count and historical unprojected-elite update.

No root produced a model-selected prefix satisfying the frozen nonprogress
and terminal-score conditions. Physical endpoint distances (lower is better):

| Root | Model greedy index | Zero cp2 / cp7 / cp17 | Greedy cp7 / cp17 | Best physically nonprogress prefix: cp17 |
| ---: | ---: | --- | --- | --- |
| 8343 | 1 | 1.3960 / 1.4599 / 1.0001 | 1.1791 / 0.0892 | 3: 0.9417 |
| 8345 | 1 | 2.0336 / 2.0625 / 1.6653 | 1.7870 / 0.7831 | 5: 1.6639 |
| 8349 | 1 | 1.6554 / 1.6136 / 0.7095 | 1.4952 / 0.8302 | 2: 1.5040 |
| 8351 | 3 | 2.6023 / 2.5429 / 1.8376 | 2.2317 / 0.7751 | 1: 1.8135 |

`cp2` is after two common observation actions, `cp7` after the prefix, and
`cp17` after ten suffix actions. The final column is an **outcome-exposed
oracle summary**, not a model-selected policy. No physically nonprogress
nonzero prefix beat both zero and greedy at cp17 with its own replanned
suffix. In root 8349 zero beat greedy; that does not supply a nonprogress
program. A longer search horizon changes execution time and total work, so
its physical outcomes cannot be treated as an equal-resource superiority
comparison with the prior five-step result.

The run counted 1,800 three-packet lookahead samples, 1,800 two-packet
feedback samples, 41 encodes, 612 branch physical steps, and 936 separate
position-backend replay steps. It reused the source bank; the source's 40
JVP calls and 80 FD samples are upstream costs, not free computational
work. No fixed common-suffix mechanism test or equal-total-budget strong
baseline is included.

An independent saved-output audit checked 36 complete bank/prefix bindings,
selector results, lookahead winners, feedback scores and decoded tails,
and physical distances. Decoding uses the historical normalizer receipt,
not complete normalizer provenance. A separate same-backend replay
re-executed all 612 actions and matched all 648 saved RGB frames.
Kernel, audit and regression tests reported 11 passed.

The legacy renderer timing diagnostic found step-after-step screenshots
separated by five ticks but one tick behind integrated qpos; RGB velocity
is consequently an interval proxy rather than guaranteed instantaneous
velocity. See `PC_CAUSAL_HISTORY_RGB_TICK_TIMING_SIDE_RESULT_20261103.md`.
Neither this diagnostic nor the outcome-exposed bank proves causal capability
change, independent generalization, native success or a P1/PC gate.

- Result:
  `le-wm-main/results/pc_causal_history_h10_suffix_20261103_DEV01.json`
  and bound NPZ.
- Saved bindings:
  `le-wm-main/results/pc_causal_h10_saved_audit_20261103_DEV01.json`.
- All-frame replay:
  `le-wm-main/results/pc_causal_h10_rgb_replay_audit_20261103_DEV01.json`.
- Protocol: `PC_CAUSAL_HISTORY_H10_SUFFIX_PROTOCOL_20261103.md`.
