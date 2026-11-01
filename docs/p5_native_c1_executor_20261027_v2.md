# Native prospective C1 three-branch executor v2

Source/tests/prepare ONLY by agent. Main supplies independently generated goal,
CUDA compiler and actual frozen bridge packet; only main executes physics. No
neural imports/calls, training, outcome-based selection or sealed-source edits.
Implementation follows experiment-code source-binding/logging/test guidance;
no generic experiment reruns or figures are in this bounded scope.

## Authority and interfaces

Executor source path for new bridge ready prepare:
`le-wm-main/experiments/spectral_reachability/p5_native_c1_executor_20261027_v2.py`.
Default prepare only seals source/helper closure and grants NO branch authority.
It verifies the unchanged Ohm bridge v1 SHA and its existing source seal, plus
sealed panel v3 budget/capture and v1 runtime/journal/profile helpers. No env
import, native construction, reset or step occurs during either prepare mode.

The `ready` command consumes actual bridge freeze output containing
freeze_receipt.json, all_branches.json and selected_packet.json. Verify their
file hashes, source seal, original ready protocol and actual CUDA compiler receipt.
Rebuild the original ready bridge using its sealed goal/compiler/executor args.
Validate exact `bridge.frozen_branches(prepared,receipt)` reproduction. Require
the ready protocol to name THIS exact executor file/hash. Verify every compiler
artifact binding, exact CUDA selected replay, model identity and source closure
through unchanged bridge validation. No replacement packet, candidate, goal,
native offset or threshold may be invented by executor.

Source manifest: `results/p5_native_c1_executor_20261027_v2/manifest_source_frozen.json`.
After actual CUDA and Ohm freeze, save a separate manifest_ready_frozen.json
from `ready`. Main must review and supply its FILE SHA to `run`.

## Fixed physical protocol

Exactly compiled, nominal, factual_reference, in that order; all slots retained,
including when compiled equals nominal. Each constructs a distinct unmodified
native env, explicit seed2026102701 BEFORE reset, identical native constructor.
No live env branch reuse, snapshot restoration, packet adjustment, retries,
root replacement or reoptimization after opening physical outcomes.

All3branch manifests, selected packet, exact15command float64 arrays and ordered
dispatch intent are fsynced BEFORE first runtime/env construction. For each arm,
replay exact10frozen prefix commands. Capture/gate every frame0..10: RGB/state/
proprio/full-public-state/transition FILE bytes must equal archived factual rolea,
not merely tolerate a state norm. Mismatch aborts this arm BEFORE any suffix.
All11 gates are required immediately before EVERY suffix command. Suffix commands
are the exact frozen raw native u64 packet, no normalization or LeWM velocity
interpretation. Native target is current agent xy+100*u. Nominal is raw native
zero, NOT normalized zero, NOT None. factual_reference is the fixed old suffix;
old factual frame15 is a reference diagnostic, never a prospective goal.

Expected each complete arm:1construct,2seed,1reset,15control,151Space.step,
16render,16get_obs,1close. All3:3/6/3/45/453/48/48/3, exactly bridge contract.
Native reset itself advances .01s; each control advances10substeps dt=.01.
Only unchanged native step returns are captured, no extra render/get_obs calls.

An exact prefix mismatch is `integrity_failure`; native done is separately
`environment_termination`; operational/count/budget guards are `guard_abort`.
Missing runtime leaves arms not_run. v2 routes BOTH close intent and return through
final-budget journal admission, not the exhausted ordinary capture budget.
Each arm reports close call attempted/completed, return durable, original/cleanup
errors. A close exception, missing dispatch or undurable return blocks ALL later
arm dispatch; later slots remain not_run. A constructor that returns before its
return receipt fails still has its env handle retained for cleanup. A constructor
that raises leaves construction indeterminate and blocks subsequent dispatch. No endpoint padding. Finally attempts close;
if close fails alongside an original error both are retained in journal, original
error is not replaced. Incomplete native intents are indeterminate, never resumed.
Source drift blocks further dispatch. Every arm has independent durable journal/
report and partial frame assets; top dispatch/progress files retain all3slots.
After process kill, use durable intents/progress; do not infer success from a
missing final report or auto-resume an output directory.

## Output and interpretation

`results/p5_native_c1_executor_20261027_v2/native_main_v1/`

Top all_branches.json, selected_packet.json, dispatch_intent.json, progress_00..03,
runtime_probe.json and final report.json. Each arm has journal.jsonl, report.json,
frozen_env_commands.npy, constructor_rng.json, pre_reset_rng.json,
command_01..15.json and frame_00..15/{rgb.npy,state.npy,proprio.npy,
full_public_state.json,transition.json}. Budgeted writer reused unchanged from
sealed v3:40MBpayload/1024nodes with reserved terminal receipts; <50MB conservative
budget on ordinary4KiB filesystem. Complete three-arm output usually <12MB.

At valid frame15, separately report agent xy/velocity, object native origin and
angle, XY goal error, principal wrapped angle error, native final/max coverage.
No mixed agent/object state norm, invented agent success goal or image-MSE
effectiveness claim. Native done is hardcoded false in current source; completion
is not success. Successful execution status `native_c1_three_branch_execution_recorded`
means protocol execution only, not C1 effectiveness, C2, P5 or generalization.

## Commands (no physics executed by agent)

From project root in original venv:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_p5_native_c1_executor_20261027_v2.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m experiments.spectral_reachability.p5_native_c1_executor_20261027_v2 prepare
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m experiments.spectral_reachability.p5_native_c1_executor_20261027_v2 ready --freeze-dir /ABS/ACTUAL/OHM/FREEZE
```

Commands print JSON; save ready output exclusively and review before main run.
Source prepare alone is NOT a runnable manifest. Main-only physics launch after
actual ready manifest freeze:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/.jepa-physics-site:$PWD" SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy .venv/bin/python -m experiments.spectral_reachability.p5_native_c1_executor_20261027_v2 run --confirm-physics --manifest-sha256 READY_FILE_SHA
```

Tests are fake-env/numeric/I/O contracts only, not physical or neural evidence.
