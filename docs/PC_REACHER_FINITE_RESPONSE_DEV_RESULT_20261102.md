# Finite-response screening: known-root model-only result

Protocol: `PC_REACHER_FINITE_RESPONSE_DEV_PROTOCOL_20261102.md`. The first
attempt stopped before screening because the new code assumed a two-coordinate
token; the actual token is a five-step, ten-coordinate action packet. No
report was produced. The corrected protocol and implementation repeat each
probe's two-axis parameter across the five packet positions; a regression
test covers that binding. The successful run is v2.

| Development root | Bank size | Retained original indices | Selected original index | Previous selected index | Screen predictor samples |
|---|---:|---|---:|---:|---:|
| 20263160 | 6 | 0, 1, 2, 3, 5, 4 | 2 | 2 | 36 |
| 20263161 | 8 | 0, 1, 4, 2, 3, 5 | 5 | 5 | 48 |

On both roots the selected parameter arrays match the previous guarded
compiler's arrays exactly. All six proposals survive on the first root; on
the second root two are screened out but the selected one survives. The
selector accounts for 396 predictor samples on both roots, compared to 396
and 528 respectively in the previous full-bank selector. Add 36 and 48
screening samples respectively; guarded compiler cost is separate. Encoding,
action encoding, and full wall-time/resource equivalence remain uncertified.
Thus this experiment does not establish an equal-budget performance advantage.

The output does not select a new action on these roots, so rerunning the old
physical actions would not create new evidence. The existing physical and
common-suffix negative findings remain applicable only to those previously
executed matching actions, not as an evaluation of this new screening
procedure on other roots. No fresh physical actions, independent roots,
native success, or P1 admission were obtained.

The implementation, guarded compiler, and successor selector regression suite
passed 16 tests, including packet shape and deterministic tie handling.
Unit tests do not constitute model or physical efficacy evidence.

Artifact: `le-wm-main/results/reacher_pc_finite_response_development_20261102_v2/report.json`.
