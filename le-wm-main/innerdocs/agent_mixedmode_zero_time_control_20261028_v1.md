# zero5 matched-time control v1 — MAIN-only physical implementation

## Fixed intervention, not another model witness

This NEW schema implements all four original `(root_slot,condition)` tasks in
order `(0,off),(0,on),(1,off),(1,on)` and, within each task, the original ordered
five probes `zero,+x,-x,+y,-y`. Each probe is held for five steps, preserving its
literal float32 amplitude (0 or ±0.1). All 20 branches are independent episodes.
Every branch reconstructs the original seed + sample-state placement recipe,
executes the original warmup10 `[.1,0]×5,[0,.1]×5`, then **actually dispatches five
positive float32 `[0,0]` actions**, then dispatches the original probe5. It never
resets to a saved post-state, reuses an endpoint, omits no-ops, pads after native
termination, or replaces a failed branch. No greedy group is present.

`intervention=zero5_time_control`, `model_selected_witness=false`, and the own
`job_digest` distinguish these branches from the original before/after jobs.
`parent_after_job_digest` and `parent_selection_digest` identify ancestry only.
The parent mixed selected prefix is NOT dispatched and no new selection_digest
is invented. Even where a parent's selected prefix happens to be zero, a fresh
20-step branch must execute in full. No outcome, cost, rank, or success enters
the intervention generator.

This matches commanded elapsed time to the selected-prefix **after** condition.
It does not hold intervening state, velocity, contact, or solver cache fixed.
Comparisons remain descriptive on two known roots (four rendering tasks), not
independent confirmation, a derivative/capability theorem, or P1 admission.
Native stops are honored even if an official native-angle formula is imperfect.
Full state7 is retained; object-only response is not full task success.

## Exact resources and predispatch permissions

The new meter seals these aggregate attempt caps (complete runs must equal them
in both attempted and completed columns):

| Axis | Fixed maximum |
|---|---:|
| constructor / close | 20 / 20 |
| reset | 40 |
| public step | 400 = 200 warmup + 100 zero + 100 probe |
| internal physics step | 4080 = 4000 public internals + 80 reset internals |
| set_state / goal_setter | 80 / 60 |
| goal_render / goal_bind | 40 / 20 |
| explicit render | 1320 |

The inert native constructor source/AST is checked before dispatch; constructor
internal work is unknown if constructor initialization fails before hooks.
Public callbacks and native reset/_set_state/space.step/render/goal hooks are
all durably intent/return/exception metered. The own meter does not modify old
helper globals or reuse the old40 caps. Only nonphysical close can bypass a
fault/exhausted wall; it still has its own20 cap and terminal journal evidence.

The official224 RGB observability guard authorizes **each primitive**. Separate
warmup and task sequences both observe steps0..10 and dispatch10. There are
22 observation triplets per branch: warmup0..10, newly bound root0, zero1..5,
probe1..5. Each is off/on/off with invariant full snapshots and exact off replay;
the selected condition feeds its own guard. The immediately preceding zero5
fresh frame serves as probe0 (no extra render). All1320 PNGs and all400 original
returned info NPZ arrays are retained losslessly, not just checkpoints. Context
history3, root7, original goal7, goal image, renderer and source warmup traces
must reproduce before intervention. No model/encoder/AD/training call exists.

Limits are1800s including saved verification; own output512MiB, final reserve
32MiB, JSONL **uncompressed**64MiB, report8MiB, prepare64MiB and free-disk floor
200MiB. Startup requires512+200MiB free for a new output; no files are deleted.
512MiB is a maximum, not a predicted actual size. Raw RGB payload alone is
198,696,960 bytes before PNG compression; info arrays/journal/report consume
additional bytes, so the original tiny-output limits would be inappropriate.
The writer accounts actual partial bytes, no-overwrite and no symlinks, and
performs final full name/inode/size/SHA census. Journaling failure stops further
physical dispatch, preserves close/final failure evidence and never hides the
first execution error behind a later source-hash/write exception.

## Interface and source closure

The source is
`experiments/spectral_reachability/agent_mixedmode_zero_time_control_20261028_v1.py`;
tests are `tests/test_agent_mixedmode_zero_time_control_20261028_v1.py`.
Only these and this innerdoc are newly written. The public frozen capability
physical helper supplies placement/render/step primitives and its durable
writer; the own module supplies20-job generation, meter, intervention episode,
execution, prepare/run/verify/consume and validators. Old modules are not patched.

Input is the actual full probe-v2 prepare:
`results/agent_mixedmode_capability_probe_20261028_v2_prepare_MAIN01.json`, SHA
`e8e4554ea2448a8c4bcf4e916934800fa8c7afe3eda335174056a551620da3d5`.
Its full original mixedmodel/task/raw/Q/normalizer/source closure is regenerated
and SHA checked, including all original artifact paths; arrays are not recopied.
The new prepare references this input with an external file SHA, its canonical
digest, all20 own jobs, own limits, installed-native constructor source and all
own/helper source/test/doc SHAs.64MiB prevents the prior8MiB metadata-cap defect
without trimming any of the ≈44k closure bindings. Actual input availability and
complete source verification remain hard preconditions, not mock certificates.

New schemas are `${NAME}.prepare`, `${NAME}.report`, `${NAME}.consume`, where
NAME is `agent_mixedmode_zero_time_control_20261028_v1`. The report has own
`execution_binding`, `contract`, ordered20 `rows`, `counts`, `errors`, complete
`artifact_sha256`, journal SHA/tail, elapsed wall, source SHA and P1=false.
Each row retains intervention/ancestry/context/raw5, probe attempt/return lower
bounds, actual root/start/end7, raw returned snapshot/NPZ journal evidence and
six probe frames. A failed/native-stopped/unstarted endpoint is null, never a
zero-filled response. A returned but inconsistent buffer remains evidence of
failure, not a successful zero action.

The own strict consumer regenerates prepare/closure, checks external report SHA,
all flat artifacts and census, binding receipt, full event chain and resources,
all attempted raw buffers and ordered actual returns, full root/goal/renderer,
guard sequence, decoded official PNG pixels/off replay/source warmups, every
info NPZ inventory/dtype/shape and frame-to-journal joins. Complete requires
exact totals on **all** resource axes; partial retains all20 denominator rows.
It does not call the old40 consumer or rename this report as old evidence.
It returns `zero_time_control_responses_verified_not_admitted` or
`partial_zero_time_evidence_preserved_not_admitted`. No signed-response analysis
or scientific admission is manufactured by execution/consume.

## MAIN commands (worker must not execute these)

Run from `/home/gbk/code/LeWorldModel/le-wm-main`; all result paths below are
reserved for MAIN. No actual prepare/physical result is created by this delivery.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
 .venv/bin/python -m pytest -q -p no:cacheprovider \
 tests/test_agent_mixedmode_zero_time_control_20261028_v1.py

.venv/bin/python -m experiments.spectral_reachability.agent_mixedmode_zero_time_control_20261028_v1 prepare \
 --probe-prepared results/agent_mixedmode_capability_probe_20261028_v2_prepare_MAIN01.json \
 --probe-prepare-sha e8e4554ea2448a8c4bcf4e916934800fa8c7afe3eda335174056a551620da3d5 \
 --output results/agent_mixedmode_zero_time_control_20261028_v1_prepare_MAIN01.json

.venv/bin/python -m experiments.spectral_reachability.agent_mixedmode_zero_time_control_20261028_v1 verify \
 --prepared results/agent_mixedmode_zero_time_control_20261028_v1_prepare_MAIN01.json \
 --prepare-sha <ACTUAL_NEW_PREPARE_FILE_SHA>

.venv/bin/python -m experiments.spectral_reachability.agent_mixedmode_zero_time_control_20261028_v1 run \
 --prepared results/agent_mixedmode_zero_time_control_20261028_v1_prepare_MAIN01.json \
 --prepare-sha <ACTUAL_NEW_PREPARE_FILE_SHA> \
 --output results/agent_mixedmode_zero_time_control_20261028_v1_MAIN01 --authorize-physics

.venv/bin/python -m experiments.spectral_reachability.agent_mixedmode_zero_time_control_20261028_v1 consume \
 --output results/agent_mixedmode_zero_time_control_20261028_v1_MAIN01 \
 --report-sha <ACTUAL_NEW_REPORT_FILE_SHA>
```

No consumer output file is implicitly written; MAIN chooses any durable receipt
destination. RAM mocks are not real resets/physics/render counts or P1 evidence.
Personal implementation-goal completion means this three-file implementation and
RAM tests only. Actual20 execution and strict acceptance remain MAIN's work.

## Actually performed pre-handoff checks

The focused suite passed **38 RAM tests** (15.51s): complete20 mock episodes,
exact400/4080/1320 accounting, twenty independent constructors/closes, unchanged
literal warmup/zero/probe buffers, full hashchain, real PNG/NPZ codecs, own strict
consumer, adversarial context/ancestry/raw/count/guard/payload mutations,
terminal/truncation/throw/substitution at zero1 with all20 retained, source-error
precedence, close reservation and actual partial-write bytes. An initial RAM
consumer fixture overpatched `Path.stat` for a virtual directory; that fixture
was corrected rather than changing production assertions or guards. No actual
physical/neural negative result arose from that test engineering failure.

The actual saved-only `material()` and canonical JSON round-trip also completed
successfully against the fixed real probe prepare: **8,467,100 bytes,44,055
closure bindings,20 own jobs**, below64MiB. This check constructed metadata in
memory only; `wrote_prepare=False`. It did not create results or run the simulator.
Its repeatable read-only command is:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
 .venv/bin/python -c 'from experiments.spectral_reachability import agent_mixedmode_zero_time_control_20261028_v1 as m; p,_,_=m.material("results/agent_mixedmode_capability_probe_20261028_v2_prepare_MAIN01.json","e8e4554ea2448a8c4bcf4e916934800fa8c7afe3eda335174056a551620da3d5"); b=m.canonical(p); import json; assert json.loads(b)==p; print(dict(bytes=len(b),bindings=len(p["bindings"]),jobs=len(p["jobs"]),cap=m.PREP_CAP,wrote_prepare=False))'
```

Remaining dependencies are MAIN source review, fresh own prepare SHA, runtime
storage/wall permissions, actual20 execution, external final report SHA and own
strict consume. No actual report/prepare SHA for this NEW runner is fabricated.
