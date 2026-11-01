# Fixed remaining-seed native factual panel v3

Prepare/source/tests only by agent; main reviews and executes physics using the
installed isolated overlay process. No neural calls, training, outcome reads,
old-source edits, import shims, native patches, candidate optimization or action
selection. This collector does not consume v10/v11 reports or sort their costs.
The experiment-code skill informs explicit source binding/logging/test contracts;
the requested bounded prepare-only scope overrides generic experiment/plot runs.

## Frozen panel

ALL remaining seeds from the original runtime manifest, in order:
2026102702, 2026102703, 2026102704. Seed2026102701 is excluded only because it was
the predeclared initial pilot, not because of its observed ordering. No collision,
short/done, mismatching, failed or poor root may be replaced or silently excluded.
These are three fresh deterministic calibration roots, not a claimed population
generalization test or archived dataset restoration.

Exact sealed v2 schedule (15 native u64 commands, three five-step xy packets):

```text
[[ .12, .04], [ .08,-.06], [-.05, .09], [ .11, .03], [-.04,-.07]]
[[-.07, .10], [ .03, .13], [ .09,-.02], [-.11, .05], [ .06,-.08]]
[[ .05,-.12], [-.09, .07], [ .13, .02], [ .02,-.05], [-.06, .11]]
```

For EACH seed: construct two independent official envs (roles a/b), explicitly
seed before reset, capture post-reset frame00, execute the frozen15 commands and
retain all16frames per role. Native reset advances one .01s substep; each command
advances ten .01s substeps. Native target = pre-step agent xy +100*u; this is a
relative target-position PD command, NOT LeWM velocity. Compare all16 paired
state/proprio/RGB/full-public-state/transition SHA256 receipts exactly, no tolerance.
Expected per-root: 2construct,4seed (including constructor seed(None)),2reset,
30control,302Space.step,32render,32get_obs,2close. Three-root complete panel totals:
6construct,12seed,6reset,90control,906Space.step,96render,96get_obs,6close.
Public-state/RNG receipts do NOT serialize private contact caches.

## Source and failure boundary

The v2 manifest and source have explicit SHA checks. All inherited sealed sources
are verified before prepare/run and before/after each root. Runtime's original
four-seed declaration is independently checked from its frozen manifest. The
collector imports only SHA-bound v1 helpers and v2 numeric functions in prepare;
no physics import/constructor/reset occurs. Actual run invokes the unchanged
native source with constructor-code provenance and isolated runtime probe.
There is no reassignment of v1/v2 seed globals and no source monkeypatching.

Reuse v1 canonical JSON/fsync exclusive writes, hash-chain journal, profiling,
public-state/RNG/numeric helpers, operation intent-before-execution, comparisons;
reuse v2 source-bound normalization/stat/array-read functions and exact schedule.
v3 adds explicit seed parameters, budgeted capture/export and panel orchestration.
Every native operation has durable intent before execution; every command file
is durable before its step. An exception can leave an unmatched native intent:
retain it as indeterminate, do not retry/resume it. Closing an env is attempted
in finally and is independently journaled, never a claim that failed work completed.

Each ordinary root failure is retained and subsequent fixed roots still attempted.
Both roles are attempted even if one fails, except global runtime/source/storage
constraints. Source/runtime/budget precondition failure leaves all remaining roots
not_run. All roots have independent journals/reports; top-level final report and
exclusive progress snapshots retain the full three-root census. Unexpected executor
escape marks that root failed/physics-indeterminate and later roots not_run, not
false no-execution. Complete or failed outputs are exclusive and never overwritten.
On process kill, panel_intent/progress_00 and all journal paths are already durable;
use latest progress plus journal intents, not missing final reports as success.
No auto-resume or replacement after interruption.

## Stable consumer schema

Default root: `le-wm-main/results/p5_jepa_factual_panel_20261027_v3/factual_main_v1/`

```text
panel_intent.json               # full ordered census, protocol/manifest SHA
schedule_env_u.npy              # one fixed15x2 float64 schedule before runtime
runtime_probe.json
progress_00.json .. progress_03.json
report.json                    # final rows, never filters failed/not_run roots
seed_2026102702/                # likewise2703/2704
  journal.jsonl
  report.json                  # status, rows16, roles, per-root count receipts
  schedule_env_u.npy
  a/                           # identical layout for b
    constructor_rng.json
    pre_reset_rng.json
    command_01.json .. command_15.json
    frame_00/ .. frame_15/
      rgb.npy                  # uint8[224,224,3], unresized native return
      state.npy                # float32[7]
      proprio.npy              # float32[4], state[[0,1,5,6]]
      full_public_state.json
      transition.json
    consumer/
      inputs_manifest.json     # schema=v3, layout_schema=v2, actual seed
      raw_env_commands.npy     # float64[15,2]
      raw_pixel_displacements.npy       # float64[15,2] =100*u64
      dataset_pixel_displacements.npy   # float32[15,2]
      normalized_packets.npy           # float32[3,10]
      controls_packets.npy             # float32[4,3,10]
      states_frames.npy                # float32[4,7]
      raw_proprio_frames.npy           # float32[4,4]
      normalized_proprio_frames.npy    # float32[4,4]
      context_proprio.npy              # float32[3,4]
```

Panel rows have seed/root/status/report/journal and terminal file hashes. Root
report contains consumer_inputs[a|b].path relative to THAT seed directory;
consumer manifest frame/packet/proprio paths are also relative to that seed
directory, NOT to the panel directory or the consumer subdirectory. Downstream
join must use seed identity, retain all three census entries, and fail closed on
any missing/failed/not_run root; no successful-root-only cohort.

Context frames0/5/10, target15. Frame00 is post-reset. command01 advances00->01;
packets advance00->05,05->10,10->15. Packet row-major five xy pairs, shape3x10.
Native float32 projection: pixel32=(100*u64).astype(float32),
z=(pixel32/float32(100)-native_mean32)/native_std32. Normalizedzero is native
mean, NOT rawzero. All4 downstream arms preserve first2packets/context proprio;
actual/normalizedzero/negative/reversed change only packet2. Controls are model
inputs only, NOT physically executed branches. The target15 state/proprio/RGB
must not leak into predictor history or select schedule. Official encoder/cache
provenance remains conditional; no official/HF equivalence claim.

Statuses: root success `fresh_native_panel_root_recorded`, root failure
`fresh_native_panel_root_failed`, skipped root `not_run`. Panel success
`fresh_native_nonzero_panel_recorded`, otherwise `fresh_native_nonzero_panel_failed`
and exit2. No model accuracy/control benefit/P5 admission follows from collection.

## Storage and launch

Hard admission on every writer: <=40,000,000 payload bytes, <=1024file/directory
nodes. Reserve2,000,000bytes and32nodes for final failure/terminal/progress reports.
Conservatively budget8KiB per node (tail/inode/directory overhead): maximum
48,388,608bytes, below50,000,000bytes, on ordinary4KiB filesystems. No image
duplication or extraction; RGB remains only in raw frame files. Budget exhaustion
fails closed and retains partial roots; no cleanup/retry/downsampling. Filesystem
compression/COW/metadata policies are host-specific; main can additionally check
actual allocated size. Typical output is much smaller than the hard ceiling.

From project root, tests/prepare in original venv (no overlay needed):

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_p5_jepa_factual_panel_20261027_v3.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m experiments.spectral_reachability.p5_jepa_factual_panel_20261027_v3 prepare
```

Prepare prints JSON only. Agent saves new manifest_frozen.json before main run.
Main-only physics launch (replace FROZEN_SHA with delivered FILE SHA256):

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/.jepa-physics-site:$PWD" SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy .venv/bin/python -m experiments.spectral_reachability.p5_jepa_factual_panel_20261027_v3 run --confirm-physics --manifest-sha256 FROZEN_SHA
```

There is no CUDA/model/training/optimization entrypoint. Tests use numeric and
fake-env I/O fixtures only, not source-native physics or neural evidence.
