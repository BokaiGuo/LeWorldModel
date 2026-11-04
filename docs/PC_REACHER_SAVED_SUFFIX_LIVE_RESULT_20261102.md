# Saved suffix real-feedback development verification

The pre-run protocol is `PC_REACHER_SAVED_SUFFIX_LIVE_PROTOCOL_20261102.md`.
On host CUDA the guarded compiler reproduced both known-root banks, selected
indices and all candidate future scores exactly. It saved 14 suffix
parameter witnesses; frozen parameter hashes matched before and after.

The first physical runner invocation rejected the new compiler report before
environment execution because the archived physical report binds the old
compiler report's SHA, not the new one. The runner now accepts an explicit
old physical-source report, checks that SHA, and checks the new and old
model identity, root/goal RGB, candidate banks, selected parameters and
per-candidate indices and costs for exact equality. Six tests passed around
the binding and carryover. The failure was a provenance interface mismatch,
not a negative physical experiment.

The second invocation executed all six known-root arms with saved suffixes,
verifying each imagined suffix score before executing its fixed prefix and
replanning from actual successor RGB and action history. All six saved suffix
arrays and all six reported physical metric dictionaries exactly matched the
previous recovered-tail carryover run; model hashes remained unchanged.

| Root | Selected final qpos distance | Zero final | Greedy final |
| --- | ---: | ---: | ---: |
| 20263160 | 0.558267 | 0.546408 | 0.531732 |
| 20263161 | 0.619442 | 0.730452 | 0.760234 |

The root-20263161 zero outcome differs from the original unmodified-CEM
baseline because inserting a carried tail changes CEM's elite population;
the selected tail itself was not selected in that zero arm. This reproduces
the previously documented development behavior, not a causal demonstration
of a more controllable selected state. Roots and actions are not independent
new evidence; no new common-suffix mechanism, native success or fair total
budget has been established. Each saved-witness verification costs one
two-token prediction rollout in addition to the upstream compiler and
feedback planner costs.

Artifacts:

- `le-wm-main/results/reacher_pc_saved_suffix_compiler_20261102_v1/report.json`
- `le-wm-main/results/reacher_pc_saved_suffix_feedback_20261102_v2/report.json`
