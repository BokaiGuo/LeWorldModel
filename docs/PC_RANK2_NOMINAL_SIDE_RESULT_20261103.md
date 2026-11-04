# Rank-two nominal response: complete development result

All 16 predeclared exposed exact-unicycle goals, five arms and three caps
completed: 240 selected plans. Every response candidate had two nonzero
singular values. This resolves the prior rank-one panel's inability to
compare mode selection, not the research advantage question.

## Mean final distance

| Objective-equivalent cap | Task rank-one | Gain rank-one | Full no-SVD ridge | Random | Task-bank greedy |
| --- | ---: | ---: | ---: | ---: | ---: |
| 64 | 8.176e-4 | 4.952e-6 | 9.851e-8 | 1.352e-7 | .116308 |
| 128 | 6.737e-8 | 7.926e-8 | 2.662e-8 | 1.334e-7 | 2.627e-7 |
| 256 | 6.737e-8 | 7.926e-8 | 2.662e-8 | 1.334e-7 | 2.627e-7 |

Mean incumbent error before refinement was .095469 (task), .233674 (gain),
.000064558 (ridge), .399584 (random), .138499 (greedy).
Thus better task-ranked starting error than gain-ranked starting error did
not imply better low-budget final mean error.

Descriptive paired task-arm wins/ties/losses, using a **post-hoc descriptive
1e-6 tolerance**, not an admission threshold or a significance test:

| Cap | vs gain | vs ridge | vs random | vs greedy |
| --- | --- | --- | --- | --- |
| 64 | 5/8/3 | 0/11/5 | 0/11/5 | 16/0/0 |
| 128 | 0/16/0 | 0/16/0 | 0/16/0 | 2/14/0 |
| 256 | 0/16/0 | 0/16/0 | 0/16/0 | 2/14/0 |

## Mechanism and cost

At cap 64 task had one non-improving selected prefix and one limited
mechanism witness; random had four of each. At caps 128 and 256 task had
neither, and no arm had a limited witness. These are actual jointly refined
chains, not candidate-bank qualifications. The probes compare each selected
suffix across prefixes and analytically optimized one-step suffixes against
zero and task-bank greedy. They do not replace a predeclared common suffix
family, real RGB interventions, or independent confirmation.

Generation cost was 48 charged calls for task/gain/ridge, 56 for greedy,
8 for random. Mean total calls at cap 64 were respectively
63.75/63.875/62.5625/64/43.6875; at higher caps,
72.8125/74.25/63.75/80.4375/44.25.
Greedy's additional prefix scores reduce its low-cap refinement allocation.
Its low-budget losses do not isolate an intrinsic capability deficit.
Analytic gradients, SVD/solve counts, dynamics steps and timing are separate;
equal objective-equivalent caps are not equal FLOPs or GPU-cost certification.
Rank diagnostic SVDs are explicitly post-selection, outside planning time,
and never enter the ridge solver.

## Verification and conclusion

26 related tests passed. Saved-data tests verify dependency hashes, all 240
selected-chain replays, bounds/cost logic, summaries, rank-two responses and
recomputed interventions. No shared compiler, historical gate or existing
experiment output was changed.

The task is again saturated at larger budgets. This experiment does not
support unique SVD value or stable task-truncation advantage. Full-response
no-SVD ridge remains a necessary control, and random initialization cannot
be omitted. No P1 admission or visual confirmation is claimed.

The next research decision is whether to test a separately frozen, harder
program-search task where all-action optimization does not saturate, rather
than rerun this exposed panel. Task selection must be geometric/prospective,
not based on which roots favor the method. Visual deployment additionally
requires actual RGB/history binding, full resource accounting and provenance.

Artifacts:

- `docs/PC_RANK2_NOMINAL_SIDE_PROTOCOL_20261103.md`
- `le-wm-main/experiments/spectral_reachability/pc_rank2_nominal_side_20261103.py`
- `le-wm-main/tests/test_pc_rank2_nominal_side_20261103.py`
- `le-wm-main/results/pc_rank2_nominal_side_20261103.json`
