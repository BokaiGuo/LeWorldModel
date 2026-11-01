# P1 fixed witness physical calibration consumer

NEW consumer only. Never modifies sealed source/direct/witness artifacts. MAIN
alone constructs the native simulator. Import/prepare/verify and tests do not
run a simulator or neural model. There is no CUDA/encoder/predict option.

## Fixed selection and identity

Prepare requires externally supplied SHA256s of actual complete witness report,
witness prepare, original direct actual report/prepare, and calibration2 sourcepack.
Original seals and input closure are verified, root reports/journals are bound by
top-level report hashes. Stable root_sha256/task_id joins, not array position,
connect sourcepack, tasks, both direct methods, and witness roots. Complete source
census and original RGB/context/Q/normalizer/model identity validation is inherited
from the sealed upstream verifiers. No future physical outcomes select commands.

Root order is source_plan slot order; each root has exactly these arms in order:
zero raw, full_H2_GN final incumbent, full_H2_gradient final incumbent, two_stage
original stable-first argmin winner. Prepare stores all eight exact 10x2 float32
command arrays, selection digests, task/goal/Q/context identity and per-arm search
costs. No clipping/reselection/shortening. Direct four-iteration incumbent chains
and all64 witness candidates/own-prefix contexts and independent H2 winner replay
are checked with pure math against saved evidence. The consumer does not pretend
to independently rerun a model or certify producer hardware merely from JSON.

## Reset and physical execution

Each branch uses original selected random reset seed, original sample_state from
that random reset's requested goal and polygons, then original placement seed and
requested options. NEVER reset(savedpoststate). No unselected proposals are rerun
here: this is the selected-root physical continuation, not another source scan.
Before any command: exact current state, requested goal, current RGB, goal RGB,
root SHA, and actual state at render must match the selected sealed source NPZ.

Actual native env class source is bound to the source closure; native constructor
requires relative=True, action_scale=100, rgb_array, render_size224. Dispatch is
original raw float32 action; no /100, no normalization, no dispatch through model
tokens. Normalized action history is part of the upstream model/task contract only.

Every step uses the existing RgbObservabilityGuard with a fresh observed frame,
checks raw finite box, index/attempt/return chronology, exact returned
versus actual successor state. Each attempt and return is fsynced before subsequent
processing. Official info (excluding duplicate full goal image), reward, termination,
truncation, observation and actual state are saved. Stop on termination/truncation,
identity mismatch, invalid observation, I/O or dispatch exception; no retries or
replacement arms. Full8 denominator and all partial rows remain. Close errors are
separate, never suppress branch first errors. Original official eval_state metric
is retained beside a separately wrapped state delta; do not relabel it as a new
official success criterion or silently fix old angle conventions.

### Native angle bug: preserve, diagnose, do not override

Native eval_state uses `min(abs(goal_angle-current_angle),2π-abs(...))` without
modulo first. For raw differences beyond2π its angle term becomes negative.
Every saved successor therefore has the actual native eval_state result, the
unaltered native formula/negative-term flag, independently wrapped angle/error,
object-only position error, agent/allxy errors and wrapped state distance. Also
reported: native-versus-wrapped same-allxy-threshold success disagreement,
terminated-versus-native-eval disagreement and native-eval-versus-formula
disagreement. Independent diagnostic thresholds use the same20/allxy andπ/9
only for attribution; they are not a changed official criterion or P1 admission.
The actual returned terminated/truncated flags are always honored: even a suspected
native false-positive stops that branch, with partial denominator retained.
Termination flags, observations, actual state and official info are fsynced before
evaluation, so evaluator exceptions cannot erase the returned evidence.

Saved calibration2 selected initial states, independently recomputed with NumPy:

| source candidate | goal angle | actual angle | native raw angle gap | native angle term | wrapped angle error | allxy error | native/wrapped success |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | -0.630803879835935 | 2.6347248116516298 | 3.2655286914875647 | 3.0176566156920215 | 3.017656615692022 | 242.21766921325218 | false / false |
| 1 | -4.24634682134344 | 0.2543251582600785 | 4.500671979603519 | 1.7825133275760674 | 1.7825133275760676 | 319.2764760895218 | false / false |

Thus negative goals ARE present, but neither saved initial state has a negative
native angle term or a success disagreement. The requested physical80steps have
NOT been executed by this agent; whether this bug affects any later actual
successor/termination is unknown, not guessed from goal signs. ±2π invariance
tests cover the independent diagnostic; a separate pure counterexample preserves
native noninvariance/false-positive behavior instead of incorrectly asserting
that the official implementation is invariant. No action is tuned to these data.

### MAIN-authorized 88-render guard protocol

Before prepare/seal, MAIN explicitly chose the existing fresh-RGB-per-command
guard and authorized88 explicit renders instead of the initial24-render proposal.
This unsealed v1 was updated; no previous prepare or source seal is reused.
Each branch renders initial and EVERY post-step observation (11×8=88); the last
fresh frame is consumed once by the next command's guard.dispatch, with no extra
precommand render and no stale-frame reuse. Existing canonical RoyalBlue guard
fails closed on missing/invalid/out-of-order RGB. Its scope remains appearance
presence only, not full observability or prevention of the first unseen transition.

Initial/step5/step10 official224 uint8 RGB PNGs (24 total) carry both byte SHA and
PNG file SHA, plus actual state and step index. Other64 observed RGB frames are
not retained as images, but their exact pixel SHA and state/step are fsynced in
the journal before guard.observe. All88 guard observations are receipted.
PNG checkpoints are fresh artifacts for later MAIN
CUDA encoding, never borrowed endpoints. Partial endpoints are marked separately;
RGB is fresh only when the last saved checkpoint equals attempted=returned steps.
No extra render is invented after off-checkpoint termination/failure.

## Resource and output contract

Successful upper envelope: 8 branches,16 public resets,80 selected public steps,
88 explicit renders; reset source implies16 internal goal renders and32 internal
physics steps. No warmup, no source scan repeated. Internal physics is implied
from frozen env source, not counted by native-substep hooks. Failures retain
attempted/completed and lower/upper accounting, never refund attempts. Two fresh
resets per branch; env construction/close itself is not a reset.

Zero arm adds no search; direct GN/gradient EACH retain their own30predict+
30action_encoder+3J+576mathVJP, with GN3SVD/3solve and gradient0SVD/0solve, per root.
Witness retains120predict/120action_encoder+17J/17SVD/17solve/3264mathVJP producer
PLUS6predict/6action_encoder winner replay per root. Source generation and native
image acquisition/direct total are separately inherited once, not charged eight
times or called free. These methods are NOT equal-total-compute, independent
confirmation or P1pass; no compute-matched win-rate conclusion is authorized.

One exclusive output directory; global8MiB cap and200MiB free floor. Launch reserves
the whole8MiB plus floor, writes fsync and enforce quota/floor. Expected output is
small PNGs + JSON/JSONL, no upstream source/report copy. An unwritable/full disk
cannot guarantee a final report; prior fsynced journal remains, stdout reports
final-write failures. No deletion/download or shared progress-doc edits.

## Commands (MAIN after review; no agent physics)

From `/home/gbk/code/LeWorldModel/le-wm-main`, independently verify file SHAs first.
Actual complete witness report SHA currently read from disk:
`d85ad0999f316db9860bc5358f787488cadfd76c5bc8e2a0ff5a59439e5903c5`.
Prepare is saved-data-only, not physical execution:

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.spectral_reachability.p1_witness_physical_20261027_v1 prepare --witness-prepared results/p1_two_stage_witness_20261027_v1_prepared01 --witness-prepare-sha 1ade88fd784f2a1217ede182d8e46f99b3007b902406c97413f46ed70cc16106 --witness-report results/p1_two_stage_witness_20261027_v1_cuda_main01/report.json --witness-report-sha d85ad0999f316db9860bc5358f787488cadfd76c5bc8e2a0ff5a59439e5903c5 --direct-prepared results/p1_direct_cuda_calibration_main_20261027_v2_prepared01 --direct-prepare-sha 24eaf8b02873b3d93ebeeadbead9ca56c2caa079cea8c30b308357ddd8a3d2e6 --direct-report results/p1_direct_cuda_calibration_main_20261027_v2_cuda_main01/report.json --direct-report-sha fad2c2ff4f409762fa5925f142f9b4d3edc371645fc4c114bfa69c996483f3d4 --sourcepack results/p1_conditional_calibration2_main_20261027_v1 --sourcepack-sha 728f14cb83c4fcd0ea4f5348bad2bff5b5dd372ab0185b6e6b3a26dbfda58057 --output results/p1_witness_physical_20261027_v1_prepared01
```

Then MAIN takes prepare.json SHA from that command and reviews the saved8 plans:

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.spectral_reachability.p1_witness_physical_20261027_v1 verify --prepared results/p1_witness_physical_20261027_v1_prepared01 --prepare-sha PREPARE_FILE_SHA
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.spectral_reachability.p1_witness_physical_20261027_v1 run --prepared results/p1_witness_physical_20261027_v1_prepared01 --prepare-sha PREPARE_FILE_SHA --authorize-physics --output results/p1_witness_physical_20261027_v1_main01
```

Pure mock tests (official224 RGB fakes, no physics/neural):
`PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_p1_witness_physical_20261027_v1.py`.
Saved actual input selection validation is separately performed read-only; mock
correctness or successful preparation is not a measured physical consequence.
