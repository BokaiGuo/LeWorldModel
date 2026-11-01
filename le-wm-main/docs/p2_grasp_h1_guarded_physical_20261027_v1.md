# New GRASP H1 guarded physical consumer

Consumes only the actual `p2_grasp_h1_cuda_runner_20261027_v1.report` and its
frozen protocol/prepare. Not the P3 v2/v3 reports or old physical endpoints.
No physics, CUDA, learned CPU inference, model loading or encoding is performed
by prepare/tests. Actual execution is exclusively main's explicit reviewed
`--execute-physics` command. Only new source/test/doc/results are written.

## Census and action support

The fixed census is four existing calibration roots × GRASP H1/action-Adam H1,
one physical repetition, including ties and rejected plans. Alternating source
arm order is preserved. The actual completed planning report contains eight
arms; four raw plans are outside [-1,1]. These four entire plans are persisted
as `rejected_raw_out_of_box`, with their **unchanged** original normalized
[3,10] and raw [15,2] actions in the prepared selections and zero dispatch.
There is no clipping, tanh, scaling repair, legal-prefix execution, retry,
replacement candidate or outcome-driven selection.

Rejections are written before any environment construction. Rejected branches
never reset, render or dispatch. The other four plans execute **at most 15**
raw commands each, not the old 10-step horizon. A fresh canonical RoyalBlue
RGB presence observation authorizes every selected-path raw step; absence,
termination or exceptions stop that branch without padding. This guard cannot
prevent the first out-of-view transition and is not full observability/control
safety. It is the existing immutable `durable.physical_branch/guarded_execute`
low-level path, not an old higher-level consumer or its outcomes.

## Frozen inputs and real root reconstruction

Planning prepare SHA:
`70f452342f574f94f3171372958d4542d4b5f4a3ce95095382ff8bf1b816d8a2`.
Prepare requires an explicit actual report SHA and independently checks:

- complete eight-slot source census, started census, per-arm JSON equality;
- source protocol ID, config, raw inversion and command hashes;
- all phase transitions, 900 backward calls/arm, actual module hook coordinates;
- actual full192 H1 verification start/goal0 MSE and separate H3 diagnostic scope;
- the full durable planning journal, including every submitted action, module
  entry/return, phase/context length, final-plan event and final verification
  action equality, with its actual SHA.

The actual journal is slightly larger than the generic Evidence 32-MB cap.
It is therefore independently **streamed**, bounded to 128 MiB, 200000 lines,
64 KiB/line, with continuous SHA and exact sequence checks. It is not truncated,
ignored or falsely treated as a small evidence file. All other consumed reports
retain the original bounded regular-file/source seal checks. Full software
source closure is collected by reading Python ASTs, not importing physics.

Physical roots are reconstructed with the same source seed, four roots and
oversampling factor four, and matched by full state/goal root fingerprint.
For each eligible branch, the low-level `_reset_and_warmup(history_len=3)`
reconstructs actual F1,F2,F3 frames after three five-step raw-zero packets.
The entire three-frame uint8 stack SHA and independent goal0 RGB SHA must match
the planning packet. Prepare checks the helper's original goal reference equals
the independently encoded intervention goal0; it cannot substitute a different
old goal. A mismatch refuses prepare rather than silently rerendering a new task.

Two between-frame normalized action packets are recomputed from the actual
raw warmup history using the source-bound public float32 mean/std, checked at
fixed absolute tolerance 2e-6 (float32 arithmetic only). There is no additional
division by 100. Repeated arms must have identical frame/history/state/goal
identities. Complete source full192 latents remain inherited; no new encoding
is used to pretend independent latent verification. Normalizer training-stat
certification remains missing, as in the planning contract.

## Budget, failures, partials and comparisons

Four eligible branches: selected-path **upper bound 60 steps**, plus separate
warmup **upper bound 60 steps**. Root-collection/setup steps and resets are
independently metered/journaled rather than folded into the selected-path budget.
Branch warmup/selected meter events carry root/method identity; per-branch
`.budget.json` snapshots retain both attempt/return deltas independently.
Attempts and returns are separate; an exception after dispatch never becomes
an uncharged zero-cost step. Actual totals can be lower due to guards/errors.
Each branch retains reset/guard/execution/evaluator failures and durable RGBs,
attempts, step returns, observations, trace and raw path. Render failure or a
step without a known return cannot turn the previous RGB into a fresh endpoint.
Remaining unavailable branches are explicit blocked slots, never dropped.

Planning costs are inherited and charged once: 21624 H1 transitions, 24
separate real-H3 model diagnostic transitions, 7200 backward calls. Inherited
RGB/goal encoding cost is unknown. This consumer adds **zero model loads,
predictions, backwards, or neural encodings**. It has no optional implicit
CUDA endpoint path; fresh endpoint encoding would be a separately reviewed
new extension, not an unaccounted side effect.

All four pair records remain. Only two complete 15-step branches with identical
initial context/goal enter the physical pose-cost difference (GRASP minus
action-Adam). Same-length fresh partial pairs are `guard-partial`; rejected,
blocked, unequal-length, stale or mismatched pairs are `unmatched`. Their delta
is null, not pooled into a full-path mean or success claim. Counts are per
branch/pair explicitly; no new independent roots are manufactured.

The physical replay is a real 15-step execution of **H1-planned** actions, not
an H3-initialized planning experiment and not an equal-total-compute comparison
to P3. No admitted P0/P2/P3 efficacy follows from completion. Official state
distance and pose are evaluator-only; neither controls action dispatch.

## Review commands

Non-neural tests:

```bash
.venv/bin/python -m pytest -q tests/test_p2_grasp_h1_guarded_physical_20261027_v1.py
```

Offline prepare requires main's complete actual report and exact SHA:

```bash
.venv/bin/python -m experiments.spectral_reachability.p2_grasp_h1_guarded_physical_20261027_v1 --prepare --planning-report results/p2_grasp_h1_cuda_runner_20261027_v1_cuda_main_v1/report.json --planning-report-sha256 <ACTUAL_REPORT_SHA> --output results/p2_grasp_h1_guarded_physical_20261027_v1_prepared
```

After review, main alone launches once into a new exclusive output:

```bash
.venv/bin/python -m experiments.spectral_reachability.p2_grasp_h1_guarded_physical_20261027_v1 --execute-physics --prepared results/p2_grasp_h1_guarded_physical_20261027_v1_prepared --prepare-sha256 <DELIVERED_PREPARE_SHA> --output results/p2_grasp_h1_guarded_physical_20261027_v1_main_v1
```

Source and input byte checks occur before and after execution. Setup journal
and started census are durable; a hard kill/power loss may prevent final report
creation, so a journal prefix is not completion evidence. No live execution,
physical endpoints or efficacy are claimed by the offline tests/prepare.
