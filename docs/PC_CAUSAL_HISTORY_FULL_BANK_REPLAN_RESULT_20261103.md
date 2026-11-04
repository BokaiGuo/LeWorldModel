# Consecutive RGB history and full-bank suffix replanning

Four previously exposed obstacle roots were run under the new
`PC_CAUSAL_HISTORY_FULL_BANK_REPLAN_PROTOCOL_20261103.md` development
contract. Each acquired two real simulator step observations after common
zero actions. The post-step image pair supplies causal velocity; simulator
state is used only for reset and offline evaluation. This avoids assuming
that a reset screenshot and a subsequent screenshot share the ordinary
step-to-step timing, but is not a certification of the legacy renderer's
observation timing or of instantaneous velocity accuracy.

The model generated nine five-action prefixes at each observed prelude
successor, used 50 two-packet CEM samples for each prefix, and locked its
selection before physical candidate execution. No root had a selected
candidate satisfying the frozen model nonprogress and terminal-score rule.
All 36 prefixes were nevertheless executed; after each prefix its own real
RGB history was encoded and its own five-action suffix was optimized with
the same CEM and 50-sample budget, then physically executed.

| Root | Eligible model prefixes | Model greedy index | Zero cp2 / cp7 / cp12 distance | Greedy cp7 / cp12 distance | Best physically nonprogress prefix: cp12 |
| ---: | --- | ---: | --- | --- | --- |
| 8343 | 2,3,4,5,6,7,8 | 1 | 1.3960 / 1.4599 / 1.3021 | 1.1791 / 0.6192 | 3: 1.2425 |
| 8345 | 2,3 | 1 | 2.0336 / 2.0625 / 1.8920 | 1.7870 / 1.2992 | 5: 1.8839 |
| 8349 | 2 | 1 | 1.6554 / 1.6136 / 1.3296 | 1.4952 / 1.2006 | 2: 1.7269 |
| 8351 | 1 | 3 | 2.6023 / 2.5429 / 2.2416 | 2.2317 / 1.4779 | 4: 2.3596 |

`cp2`, `cp7` and `cp12` are after the two-action prelude, five-action
prefix and five-action suffix, respectively. Distances use the separate
position evaluator and lower is better. Physical nonprogress here means
cp7 distance at least cp2 distance, irrespective of model eligibility. The
last column is an **outcome-exposed oracle summary**, not a policy choice.
No physically nonprogress nonzero bank prefix beat *both* zero and greedy
after its own real-RGB suffix replan in these four roots. This is a result
for this finite bank, optimizer, horizon and development roots, not proof
that no useful program exists.

The task-response JVP/FD relative discrepancies were 0.00138–0.00197.
The run recorded 40 JVP calls, 80 FD samples, 1,800 two-packet lookahead
samples, 1,800 feedback samples, 8 prelude-acquisition physical steps,
432 branch physical steps and 756 position-backend replay steps. The
shared suffix optimizer and feedback opportunity do not make the
whole method equal-compute to its controls; JVP internals and complete
wall-clock resources remain unverified against a strong baseline.

The independent saved-data audit checked all 36 prefix/action and lookahead
tail bindings, model selection, feedback score bindings and position-derived
distances. All 36 final feedback tails also exactly matched decoding of the
winning normalized CEM packet using the historical normalizer receipt; that
does not prove complete training-data or normalizer provenance. A separate
same-backend run replayed all 432 branch actions and matched all 468 saved
RGB frames exactly. These checks validate recorded bindings and replay, not
new roots or causal mechanism. Saved-audit corruption checks, existing choice
and optimizer tests, and RGB velocity tests reported 16 passed.

The result lacks a fixed common-suffix intervention, native success, a
matched-total-resource strong baseline and independent root confirmation.
No P1/PC gate is opened, and no historical gate or result is modified.

- Data and receipt:
  `le-wm-main/results/pc_causal_history_full_bank_replan_20261103_DEV01.json`
  and bound NPZ.
- Saved binding audit:
  `le-wm-main/results/pc_causal_history_saved_audit_20261103_DEV01.json`.
- Tail decode audit:
  `le-wm-main/results/pc_causal_history_tail_decode_audit_20261103_DEV01.json`.
- Full RGB replay:
  `le-wm-main/results/pc_causal_history_rgb_replay_audit_20261103_DEV01.json`.
