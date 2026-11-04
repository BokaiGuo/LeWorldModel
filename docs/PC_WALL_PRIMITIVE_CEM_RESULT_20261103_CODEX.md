# State-blind primitive CEM: exposed development result

This post-outcome test compares four-scalar turn/move programs with flat
32-coordinate CEM on the same single exposed exact wall task. The primitive
generator uses neither exact state nor wall geometry, but both methods are
scored with the exact terminal task position; this objective is privileged
for RGB deployment. The parameterization is an ordinary hand-specified
action program, not a spectral method.

| Seed | Flat terminal distance | Primitive terminal distance |
| ---: | ---: | ---: |
| 20261103 | 0.550612 | 0.471314 |
| 20261104 | 0.508483 | 0.501462 |
| 20261105 | 0.511999 | 0.500783 |

The primitive arm has lower distance in all three repeated searches. In
seed 20261103 its selected program passes around the wall without blocked
steps and ends at `[0.04181, -0.11045]`, still 0.471314 from the goal.
The other two primitive programs remain on the starting side; none reaches
the goal. This supports candidate parameterization as a development
variable, not independent generalization or spectral advantage.

Each arm evaluated 204 complete 16-step programs, for 3,264 exact state
transitions, plus a separate selected replay. Calls and transitions match,
but search dimensions and overhead do not, so equal end-to-end cost is not
established. The three flat selected actions and errors match the previous
flat CEM receipt exactly. All 204 saved proposals for each arm on the first
seed were separately replayed in tests; elite ordering and selection were
recomputed. Together with cross-suffix and wall tests, **9 passed**.

This cannot establish why the primitive arm differs: both its parameter
dimension and action correlations change. It also does not establish a
reconfiguration mechanism, learned-model performance, independent roots or
any gate.

- Protocol: `docs/PC_WALL_PRIMITIVE_CEM_PROTOCOL_20261103_CODEX.md`
- Runner: `le-wm-main/experiments/spectral_reachability/pc_wall_primitive_cem_20261103_codex.py`
- Receipt: `le-wm-main/results/pc_wall_primitive_cem_20261103_codex_DEV01.json`
- Tests: `le-wm-main/tests/test_pc_wall_primitive_cem_20261103_codex.py`
