# P2 shared-candidate full30 physical replay

This exploratory fixed comparison replays, for each of four existing base jobs,
both the selected F candidate and fixed candidate index 1. Both originate in the
same saved observed cp15 context and same paid candidate set. Two known roots
are not independent samples. This is fixed after earlier results, not
preregistered; equal total compute, FLOPs, science and P2 admission are false.
No neural work is performed here.

Each job uses a fresh native environment per arm, the original ten-step warmup,
the literal identical 15-step prefix, then its assigned 15-step suffix. The
existing exact native/RGB prefix guard must be durably emitted before dispatch
of action 16; there is no state injection or new encoding. Native early stops
retain missing checkpoints and never pad. Per job caps are constructor/close 2,
reset 4, public steps 80, simulator steps 808, set-state 8, goal setter 6,
goal render 4, goal bind 2, render 252, plus 900 seconds including input
verification and publication. Failed attempts and cleanup remain recorded.

The source F model execution is inherited once per base; the constructed
candidate-1 execution witness is explicitly derived from the saved candidate,
not a producer selection or new model call. The receipt runs the full saved
native checks inherited from the existing physical interface, including PNG
bytes, event chronology, actual actions, labels and caps; the upstream model
authentication is inherited and locally rechecked, not a fresh model replay.

CLI: prepare each fixed job to a new absolute path, verify using its actual
prepare hash, then run with explicit physics authorization and a new output
directory. Consume each report with its actual report and completion hashes.
Do not substitute historical output paths or infer missing values.

```bash
.venv/bin/python -B -m experiments.spectral_reachability.codex_p2_feedback_candidate_physical_20261030_v1 prepare --job 0 --output /ABS/codex_p2_feedback_candidate_physical_20261030_v1_PREP_JOB0.json
.venv/bin/python -B -m experiments.spectral_reachability.codex_p2_feedback_candidate_physical_20261030_v1 verify --prepared /ABS/codex_p2_feedback_candidate_physical_20261030_v1_PREP_JOB0.json --prepare-sha ACTUAL_PREP_SHA
.venv/bin/python -B -m experiments.spectral_reachability.codex_p2_feedback_candidate_physical_20261030_v1 run --prepared /ABS/codex_p2_feedback_candidate_physical_20261030_v1_PREP_JOB0.json --prepare-sha ACTUAL_PREP_SHA --output /ABS/codex_p2_feedback_candidate_physical_20261030_v1_RUN_JOB0 --authorization P2_SHARED_CANDIDATE_EXPLORATORY_PHYSICS_ONLY
.venv/bin/python -B -m experiments.spectral_reachability.codex_p2_feedback_candidate_physical_20261030_v1 consume --output /ABS/codex_p2_feedback_candidate_physical_20261030_v1_RUN_JOB0 --report-sha ACTUAL_REPORT_SHA --completion-sha ACTUAL_COMPLETION_SHA --receipt /ABS/codex_p2_feedback_candidate_physical_20261030_v1_RECEIPT_JOB0.json
```

Repeat exactly for jobs 1, 2, 3. The four paired bases share two known roots.
Missing/failed/early-stopped slots must be retained and cannot be relabelled as
full-horizon observations or evidence of scientific success.
