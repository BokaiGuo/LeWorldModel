# Moving-history observed-prefix suffix physical replay v1

NEW implementation only. MAIN reviews, prepares and alone runs physics; no
implementer physics/CUDA/CPU neural. Known-root saved-prefix feedback calibration,
not online selection from an old terminal, not independent root discovery.
`P1_admission=false`, `P4_admission=false`; no mechanism gate is bypassed.

## Frozen contract and minimal reuse

Original four branches are fixed root-major `(0,line_search_GN)`,
`(0,full10_conditional_D2P)`, `(1,line_search_GN)`,
`(1,full10_conditional_D2P)`. Never filter by outcomes or replace a branch/root.
Original physical report:

```text
/home/gbk/code/LeWorldModel/le-wm-main/results/p1_constructive_moving_history_physical_20261027_v1_main01/report.json
SHA256 a5322dc4f483e82b5ed9d7b3996cf8a63845261d65774f90a4ae341773016aaf
```

Frozen moving physical tool source SHA is
`44d90599eab881c773d5efa09177dfe6b395fa3f4eef41070340681fea5ec96e`.
Reuse its immutable Store, base Ops billing, raw/command validators, saved source
loader/verify, metric, capture and hashchain helpers. NEW replay branch and
lifecycle: NEVER invoke or monkeypatch its branch/execute, nor any old zero
branch. New Ops adds lexical operation activity over unchanged frozen billing;
new hooks reject post-reset state setters/goal-bind physics/render and restore
all replaced methods. A failed goal_bind remains charged but leaves its lexical
scope, so attempted/completed deficit is not a permanent lock on later branches.
No changes to sealed files, shared docs, sourcepacks or old results.
This is an exception-path precaution only: OLD actual goal-binds all completed;
it is NOT an explanation of old zero-success results.

Per branch:

1. Repeat original seed reset and original placement recipe, then actually
   dispatch source's ten float32 raw warmup commands: first5 `[.1,0]`, last5
   `[0,.1]`. Never tanh, normalize, round, pad zero or set a poststate.
2. Compare every warmup0..10 returned frame to that branch's OLD saved PNG
   byte SHA and pixel SHA, observed state and full two-body snapshot. At0/5/10
   also compare original approved blind RGB/state/snapshot/raw source history.
3. Attach the same approved goal offline: no step/reset/render/poststate setter;
   dynamics and native default goal_pose unchanged. Compare post-bind plan0.
4. Execute original raw prefix5 commands unchanged, comparing each plan1..5
   PNG byte/pixel SHA, state and full snapshot to OLD saved physical receipts.
   Check raw latest_action, native reward/terminal and returned/live goal at
   every step. Native stop during prefix is a failed replay, never permission
   to continue. At exact prefix5, check same goal state/RGB, set prefix_exact
   and emit `suffix_admitted_after_exact_prefix` BEFORE dispatching step6.
5. Continue in that SAME env with own approved new suffix5. No reset/state
   restore between prefix/suffix, no retry, no zero-history branch.

Old snapshots contain state/state_float64_bits plus agent/block position,
velocity, unwrapped_angle, angular_velocity, force, torque, is_sleeping and
float64_bits. No claim about unexported solver warm-start cache. Old receipts
do not separately store goal_pose; this implementation checks goal_pose is
unchanged by its own bind, NOT that an independently logged old goal_pose exists.
Pure capture persists new frame/state/snapshot before paired mismatch rejection.
Exceptions thrown inside native step retain the last successful observation;
that is not a claimed exact failure scene.

## Paired labels, not control reruns or selection

Old suffix terminal10 and metrics are read from the SHA-sealed OLD report and
receipts. They are saved controls, NEVER replayed as a separate control arm,
used to choose new commands/roots or injected into the env. New report exposes
`saved_old_terminal`, `saved_old_final_metrics` and
`paired_native_distance_delta = new_native_distance - saved_old_native_distance`.
This is native distance, NOT latent Q or object-success proof. New selected
suffix is allowed to equal original: `selected_suffix_matches_original` reports
that literal action identity rather than forcing a change. In that case the
selected path coincides, but still no EXTRA old-control run was launched.

MAIN supplied real RGB report
`results/p1_constructive_moving_history_rgb_cuda_20261027_v1_cuda_main01/report.json`
SHA `7ec03c0a6a6e8411ff0dd0115044a872d099106edc95e4529b97c38c2b9b8f71`.
That is upstream compiler context evidence, not read by this physical executor.
Compiler must bind it and obey prefix-only model inputs under its own protocol.
This executor does not pretend to audit model/history semantics or re-run neural
consume. MAIN must review the final NEW compiler and successfully run its own
saved consume before approving the manifest below.

## NEW manifest and final publisher interface

Exactly the previous moving physical selection fields, with NEW schema plus
three extra fields. Common fields:

```text
schema = moving_history_observed_suffix_physical_20261027_v1.selection
sourcepack, sourcepack_sha256, source_prepared, source_prepare_sha256,
roots, arms, plans, selection_fixed_before_physics=true,
physical_outcome_selection=false, wall_seconds=integer1..600
```

source/root/goal/arms must equal original physical manifest exactly. Four plans
must retain original root-major order and original arm names. Extra fields:

```text
original_physical_report = absolute path of ORIGINAL report
original_physical_report_sha256 = a5322dc4...016aaf (FULL64 hex above)
compiler_authority = {
  source, source_sha256, report, report_sha256,
  consume, consume_sha256, consume_schema, consume_status,
  mappings: [MAPPING0,MAPPING1,MAPPING2,MAPPING3]
}
```

`source/report/consume` are absolute paths to the reviewed NEW compiler source,
actual NEW compiler report and MAIN-saved successful NEW own-consume JSON stdout.
Their three SHAs are external approval inputs, all bound/rechecked. No dynamic
module import, model invocation or execution of arbitrary manifest code.
Read current publisher `moving_history_observed_suffix_cuda_20261027_v1.consume`:
its own top-level schema is `moving_history_observed_suffix_cuda_20261027_v1.consume`,
status is exactly `observed_suffix_commands_verified_not_admitted`, errors is[];
new_physics=0, P1_admission=false, P4_admission=false. These are REQUIRED, not
optional or MAIN-added. Source basename must be that NEW compiler's `.py`.
All four literal `plans[i].whole_run_report_sha256` must equal authority.report_sha256.
Final source/report/consume external SHAs still come from MAIN, never invented.
The executor checks linkage, not a substitute implementation of compiler consume;
MAIN must actually run the publisher's successful own consume. Neither labels
nor files constitute scientific admission.

Each MAPPING contains exactly these four RFC6901 pointers into RAW saved
own-consume stdout. Full four-row authority.mappings example, in fixed order:

```json
[
  {"commands_pointer":"/plans/0/commands","source_slot_pointer":"/plans/0/source_slot","blind_sha_pointer":"/plans/0/blind_sha256","arm_pointer":"/plans/0/arm"},
  {"commands_pointer":"/plans/1/commands","source_slot_pointer":"/plans/1/source_slot","blind_sha_pointer":"/plans/1/blind_sha256","arm_pointer":"/plans/1/arm"},
  {"commands_pointer":"/plans/2/commands","source_slot_pointer":"/plans/2/source_slot","blind_sha_pointer":"/plans/2/blind_sha256","arm_pointer":"/plans/2/arm"},
  {"commands_pointer":"/plans/3/commands","source_slot_pointer":"/plans/3/source_slot","blind_sha_pointer":"/plans/3/blind_sha256","arm_pointer":"/plans/3/arm"}
]
```

These paths were confirmed by reading the current publisher's own consume and
model_plan construction, including its new status/blind fields. Own consume exposes
literal FULL raw10x2 commands (unchanged original prefix5 + selected suffix5),
literal int source_slot, original arm name and approved original blind SHA for
each fixed branch. If it emits only suffix5, this consumer will fail closed;
MAIN must not synthesize full commands and call that producer-certified.
Pointer values must match the approved physical selection rows exactly.

Each `plans[i]` has exactly frozen moving physical's fields:

```text
root_slot, arm, commands,
compiler_report, compiler_report_SHA, compiler_schema, compiler_status,
commands_pointer, source_slot_pointer, blind_sha_pointer, arm_pointer,
selection_digest
```

Here `compiler_report` MUST name authority.consume itself, and compiler_report_SHA
MUST equal authority.consume_sha256. compiler_schema/status MUST be its own exact
schema/status above. Plan-row pointers use the SAME four literal paths as its
authority mapping. No MAIN handoff wrapper or added consume metadata is allowed.
There is no implicit field search. `selection_digest` uses frozen source.digest
over the entire row except itself, with default JSON spaces (not compact).
All commands finite exactly float32-representable raw10x2 in[-1,1], no clip/round.
First5 must equal old own prefix5 in raw float32 bytes before any env creation.

Exact construction pattern (RAM/print only; MAIN saves untouched own-consume
stdout, creates selection via apply_patch, and seals actual files):

```python
from pathlib import Path
from experiments.spectral_reachability import moving_history_observed_suffix_physical_20261027_v1 as c
# authority has exactly the nine keys listed above, using real external SHAs
# and the four literal mappings above, AFTER final new-compiler review/consume.
def make_selection(authority):
    pr=c.ROOT/'results/p1_constructive_moving_history_physical_20261027_v1_main01/report.json'
    d=c.load_original(pr,c.ORIGINAL_SHA)
    old=c.read(d['original_prepared']['manifest'])
    consumed=c.read(authority['consume'])
    rows=[]
    for i,a in enumerate(authority['mappings']):
        row=dict(root_slot=c.base.pointer(consumed,a['source_slot_pointer']),
                 arm=c.base.pointer(consumed,a['arm_pointer']),
                 commands=c.base.pointer(consumed,a['commands_pointer']),
                 compiler_report=str(Path(authority['consume']).resolve()),
                 compiler_report_SHA=authority['consume_sha256'],
                 compiler_schema=consumed['schema'],compiler_status=consumed['status'],**a)
        row['selection_digest']=c.source.digest(row)
        rows.append(row)
    m={k:old[k] for k in ('sourcepack','sourcepack_sha256','source_prepared',
                          'source_prepare_sha256','roots','arms')}
    m.update(schema=c.NAME+'.selection',plans=rows,
             selection_fixed_before_physics=True,physical_outcome_selection=False,
             wall_seconds=600,original_physical_report=str(pr),
             original_physical_report_sha256=c.ORIGINAL_SHA,
             compiler_authority=authority)
    c.selection(m,d)  # recheck all authorities/pointers/literal prefix before printing
    print(c.source.canonical(m))
```

No new compiler source/report SHA or actual new commands are invented here.
The compiler's own consume must authenticate its model,
prepare/export/context/ledger chains; MAIN audits those semantics separately.

## Budget, durability and report interface

No neural/encoder/projector/AD/training calls. Full four-branch cap identical
to frozen moving physical: 8 resets, 16 reset state setters, 8 reset goal
renders, 4 offline binds, 80 public steps, **816 actual internal physics steps**
(16 reset setter steps +80x10), 88 explicit renders/PNG. This includes replay
warmup/prefix cost, not just five new suffix actions. Native suffix terminal or
truncation stops early; actual attempted/completed counts determine usage,
never padding. No claim of equal total cost with the saved old control.

Store is exclusive NEW output, total<=8MiB including partial/final journal,
256KiB final reserve, >=200MiB free floor; prepared JSON<256KiB. Predispatch
wall/cap/storage checks, durable operation receipts, complete four-row
denominator even on failure/not-started. No retry/replacement. Any final
seal/census/journal/report failure makes return partial. MAIN external watchdog
still required. Journal initial prepare binds current source/tests/doc, original
source/prepared/manifest/report/journal/all88 PNG and new compiler authorities.
Original PNGs are read/hashed, not copied into new output.

New report schema `moving_history_observed_suffix_physical_20261027_v1.report`;
complete record status `moving_history_observed_suffix_physical_recorded`, else
`partial_or_failed` (CLI exit2). Complete record is not physical success.
`rows` fixed four; identity `(root_slot,arm,selection_digest)`; also
`original_selection_digest`, prefix_exact, prefix_goal_exact,
selected_suffix_matches_original, saved_old_terminal/saved_old_final_metrics,
paired_native_distance_delta. `checkpoints` plan0/5/10 when actually reached,
receipts use same phase/step/png/file_sha256/pixel_sha256/state/snapshot/
observation_state fields as frozen moving physical. Do not synthesize missing
checkpoints. New RGB stage adapters must explicitly accept THIS schema;
do not pretend it is the old frozen report.
New checkpoint PNGs resolve under NEW output; saved_old_terminal.png resolves
under ORIGINAL physical output. Do not confuse these directories.

## MAIN commands and implementation evidence

From `/home/gbk/code/LeWorldModel/le-wm-main`; replace uppercase external SHA
and absolute NEW paths. No commands below were used to run physics by implementer.
CLI entry `if __name__=='__main__':main()` is present; `python -B -m MODULE --help`
was actually run successfully without importing neural/physics modules.

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_moving_history_observed_suffix_physical_20261027_v1.py

MOD=experiments.spectral_reachability.moving_history_observed_suffix_physical_20261027_v1
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m "$MOD" prepare --manifest /ABS/NEW_SELECTION.json --manifest-sha MANIFEST_SHA --output /ABS/NEW_PREPARED.json
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m "$MOD" verify --prepared /ABS/NEW_PREPARED.json --prepare-sha PREPARED_SHA
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m "$MOD" run --prepared /ABS/NEW_PREPARED.json --prepare-sha PREPARED_SHA --output /ABS/NEW_PHYSICAL_DIR --authorize-physics
```

RAM tests use an artificial numeric state machine, not a simulator/model;
cover literal warmup/prefix/new suffix, per-step PNG and body-bit mismatch,
goal drift, failed frame retention, action substitution, native stops, no state
restore/extra goal work, primary/cleanup errors, independent scope after a
failed bind, resource census, source/consume SHA and explicit mapping mutations,
all-four denominator, terminal labels not influencing commands, final durability.
Actual saved-data adapter smoke loaded four fixed branches, 11 warmup and6
prefix receipts each, bound115 paths including old PNGs; 0 new physics/neural,
no prepare/output directory. This is interface implementation validation, not
an independent scientific audit or a new physical result.

Remaining: final NEW compiler source/report and own successful consume,
external SHA-approved raw consume+manifest, MAIN code review/prepare/verify/physical
execution, explicit new-schema RGB consumer if wanted. P1/P4 remain false.
