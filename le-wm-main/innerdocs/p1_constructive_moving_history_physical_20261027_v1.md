# Moving-history selected physical consumer v1

Implementation-only handoff; MAIN reviews and alone executes physics. No model
imports/calls, no training, no P1/P2 scientific admission. Reused two seed/placement
recipes create NEW moving-history roots/tasks, not new independent samples.

## Minimal reuse and source contract

NEW `branch`, `capture`, `native_hooks`, source adapter and execution lifecycle.
Reuse frozen original physical `commands`/`pointer`/saved `selection` checks,
`metric`/`retain`, hashchain `journal`, bounded `Store` and predispatch `Ops`.
Never invoke original zero-history `branch`, `execute`, `load_source`, `material`
or `verify`; never monkeypatch old globals. Reuse of the original
`selection` pure validator translates only a deep-copied schema string.

Moving source code is pinned to
`9fb76ecfa054f878ab9778fc29c59bc708728b240dd2cbc3060b166e52622f55`.
MAIN's actual sourcepack is
`results/p1_constructive_moving_history_source_20261027_v1_main01/sourcepack.json`,
external SHA `c2d8d573f94884ffc081163f25f2bfa269f98ffe8155887704109219910ffec6`.
The adapter binds sourcepack, prepared envelope, execution binding, complete
journal hashchain, blind NPZ and source prepare's bindings. It reads NO
certificate/witness arrays. Evaluator-only goal state comes from the sealed
construction endpoint journal; endpoint RGB is matched to exported goal_rgb.
These are evaluation metadata, not candidate commands or planner inputs.

`raw_action_history` must be exact float32 bytes of source `warmup_commands()`:
first five `[float32(.1),0]`, then five `[0,float32(.1)]` in raw world coordinates.
Reject zero/tanh/normalized/rounded/float64 payload. For each of four fixed
root-major branches: reset original seed, repeat original sampled placement
with seed0, dispatch these ten actual raw commands individually. Reconstruct
RGB, state dtype/shape/bytes, both-body snapshots exactly at0/5/10 against source
independent replay0/replay1. Snapshot covers state bits and both agent/block
position/velocity/unwrapped_angle/angular_velocity/force/torque/is_sleeping/bits;
it does NOT certify unexported solver cache.

Only after reconstruction attach new goal through frozen source `attach_goal`:
no poststate setter, physics or render; state/body snapshot and native
`goal_pose` remain unchanged. `with_target=False`; this is NOT a custom-goal
marker experiment. Execute each arm's own literal selected raw10x2, checked
against native latest_action after every saved transition. No witness as
candidate, no poststate restoration, retry, replacement or outcome selection.

## Explicit planner handoff (do not guess future layout)

Prepare takes external SHA of a NEW selection JSON. Exact top-level fields:

```text
schema = p1_constructive_moving_history_physical_20261027_v1.selection
sourcepack, sourcepack_sha256, source_prepared, source_prepare_sha256
roots = [{root_slot:0,root_sha256:SOURCE_ROW0}, {root_slot:1,root_sha256:SOURCE_ROW1}]
arms = [ARM0, ARM1]  # exactly two unique safe opaque strings, actual producer names
plans = four rows in (0,ARM0),(0,ARM1),(1,ARM0),(1,ARM1) order
selection_fixed_before_physics = true
physical_outcome_selection = false
wall_seconds = integer 1..600
```

Each plan row has exactly:

```text
root_slot, arm, commands,
compiler_report, compiler_report_SHA, compiler_schema, compiler_status,
commands_pointer, source_slot_pointer, blind_sha_pointer, arm_pointer,
selection_digest
```

`selection_digest=source.digest(row_without_selection_digest)`; use the frozen
digest's default JSON spaces, NEVER compact separators. `commands` must be
finite, exactly float32-representable raw10x2 within[-1,1], never clipped/rounded.
Each pointer is an explicit RFC6901 path into the same SHA-sealed compiler
handoff JSON, which must expose matching schema/status and `errors:[]`.
`source_slot` is literal int, `arm`/commands exact, blind SHA must equal the
approved original slot blind NPZ SHA. No native root identity is required in
the blind compiler; roots live only in evaluator manifest.

At implementation time `moving_history_planners_cuda_20261027_v1` is not yet
frozen. Its current actual report rows do NOT contain commands. Its read-only
`consume` returns `plans` containing source_slot/arm/commands/blind_sha256 and
validates its own report/artifacts/model/exports/ledgers. Do NOT point commands
at absent report fields or use older zero-task consume. After its final MAIN
review, a minimal saved handoff may be constructed by MAIN from successful
own `consume`, for example:

```python
# MAIN only, after final producer SHA review and successful saved consume.
# out = own_planner.consume(actual_output, EXTERNAL_REPORT_SHA)
# handoff = dict(schema='moving_history_planner_handoff.v1',
#                status='commands_verified_before_physics', errors=[], data=out)
# Seal handoff in a NEW path; no alteration of original report.
# Row i pointers: /data/plans/i/commands, /data/plans/i/source_slot,
#                /data/plans/i/blind_sha256, /data/plans/i/arm
```

This consumer authenticates that explicitly approved handoff, not independently
reruns the planner consume. MAIN must retain and approve its chain to final
planner source/prepare/report/consume; merely adding `errors:[]` to unchecked
commands is NOT verification. Creating the handoff/manifest does not authorize
physics. Sourcepack/prepare/report paths must all be absolute. No planner
actuals or scientific results are fabricated by this implementation.

Exact four-row manifest example, consuming the saved envelope above (not the
old zero consume). This is executable saved-data arithmetic after MAIN has
created/approved the envelope; it prints the JSON and writes nothing. Replace
the two explicit handoff placeholders; all actual command values are copied
literally from approved `consume`, not mock commands. Successful `consume`
must precede creation of that envelope. No direct agent-message tool is
available in this session; MAIN can forward this exact contract to Poincare.

```python
from pathlib import Path
from experiments.spectral_reachability import p1_constructive_moving_history_physical_20261027_v1 as c
handoff=Path('/ABS/NEW_MOVING_HANDOFF.json')
handoff_sha='REPLACE_WITH_APPROVED_HANDOFF_SHA'
c.require(c.sha(handoff)==handoff_sha, 'external handoff SHA')
h=c.read(handoff);out=h['data']
c.require(h['schema']=='moving_history_planner_handoff.v1' and
          h['status']=='commands_verified_before_physics' and h['errors']==[] and
          out['schema']=='moving_history_planners_cuda_20261027_v1.consume' and
          out['status']=='moving_commands_verified_not_admitted', 'own successful consume envelope')
sp=c.ROOT/'results/p1_constructive_moving_history_source_20261027_v1_main01/sourcepack.json'
sp_sha='c2d8d573f94884ffc081163f25f2bfa269f98ffe8155887704109219910ffec6'
b=c.read(sp.parent/'execution_binding.json')
d=c.load_source(sp,sp_sha,b['prepared'],b['prepare_file_sha256'])
arms=['line_search_GN','full10_conditional_D2P']
c.require([(p['source_slot'],p['arm']) for p in out['plans']]==
          [(s,a) for s in (0,1) for a in arms], 'four fixed own plans')
rows=[]
for n,p in enumerate(out['plans']):
    row=dict(root_slot=p['source_slot'],arm=p['arm'],commands=p['commands'],
        compiler_report=str(handoff.resolve()),compiler_report_SHA=handoff_sha,
        compiler_schema=h['schema'],compiler_status=h['status'],
        commands_pointer=f'/data/plans/{n}/commands',
        source_slot_pointer=f'/data/plans/{n}/source_slot',
        blind_sha_pointer=f'/data/plans/{n}/blind_sha256',
        arm_pointer=f'/data/plans/{n}/arm')
    row['selection_digest']=c.source.digest(row)
    rows.append(row)
m=dict(schema=c.NAME+'.selection',sourcepack=str(sp),sourcepack_sha256=sp_sha,
    source_prepared=b['prepared'],source_prepare_sha256=b['prepare_file_sha256'],
    roots=[dict(root_slot=s,root_sha256=r['root_sha256']) for s,r in enumerate(d['references'])],
    arms=arms,plans=rows,selection_fixed_before_physics=True,
    physical_outcome_selection=False,wall_seconds=600)
c.selection(m,d)  # exact commands/slot/arm/blind SHA/pointers/digests validation
print(c.source.canonical(m))
```

## Resource/durability contract

Full four-branch cap: 8 resets, 80 public steps, **816 actual internal physics
steps** (16 reset setters x1 +80x10), 16 reset-only state setters, 8 reset goal
renders, 4 offline goal binds, 88 explicit renders/PNG. Each complete branch
saves warmup0..10 and post-bind plan0..10, including duplicate root. Independent
attempted/completed counts include nested reset physics and steps that throw.
Model/encoder/projector/AD calls all zero; physical differences are measured by
MAIN, not inferred here. No equal-budget superiority claim.

Store: exclusive NEW output dir, output <=8MiB including journal/final,
256KiB final reserve and >=200MiB free floor; prepared JSON <256KiB. No retry
when wall/cap/storage fails. Every returned valid RGB is persisted with state
and snapshot before guards. Invalid numeric images get NPZ receipts. Native
terminated/truncated stops dispatch, never pads to10. Failed/not-started rows
remain in four-row denominator. Exceptions thrown inside native step retain
the LAST successfully observed frame, not an invented exact failure scene.
Close/restore errors preserve primary error. Final seal/census/journal/report
failure cannot return success. Runtime needs MAIN external watchdog as well.

## MAIN CLI (not executed by implementer)

From `/home/gbk/code/LeWorldModel/le-wm-main`, substitute approved external SHA
values and NEW absolute paths; output parent must already exist.

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_constructive_moving_history_physical_20261027_v1.py

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_moving_history_physical_20261027_v1 prepare --manifest /ABS/NEW_SELECTION.json --manifest-sha MANIFEST_SHA --output /ABS/NEW_PREPARED.json

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_moving_history_physical_20261027_v1 verify --prepared /ABS/NEW_PREPARED.json --prepare-sha PREPARED_SHA

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_moving_history_physical_20261027_v1 run --prepared /ABS/NEW_PREPARED.json --prepare-sha PREPARED_SHA --output /ABS/NEW_PHYSICAL_DIR --authorize-physics
```

Run returns2 for `partial_or_failed`, zero only for
`moving_history_physical_recorded`; that status means records/census complete,
NOT success or P1 admission. Outputs execution_binding.json, all PNG/invalid
NPZ, journal.jsonl and report.json. The journal's run_started.prepare contains
the exact source closure/selected commands; report.json contains sourcepack
SHA, full four rows, attempted/completed resources, native/wrapped diagnostic
metrics and errors.

## Implementation checks and remaining prerequisites

39 pure array/mock tests passed; this is not physical evidence. Tests cover nonzero
dispatch, independent own commands, all three boundary reconstructions, body
and signed-zero-bit drift, failed guard persistence, raw substitutions,
render mutation, native stops, exceptions/cleanup, forbidden goal side effects,
budget/storage/auth gates, source receipt drift and explicit pointer joins.
Actual sourcepack above passed saved-data adapter smoke: two references,
float32 raw10 and three snapshots each; 0 new physics/model calls, no files.
MAIN's independent source audit is separate evidence, not performed here.

Remaining: MAIN final planner review and actual model plans; successful own
consume plus sealed explicit handoff/manifest; MAIN consumer code review,
prepare/verify and actual physics. Mock tests and source reach do NOT certify
planner reach, P1 direct-stagnation prerequisites, fresh independent roots or
strong-baseline scientific comparability.
