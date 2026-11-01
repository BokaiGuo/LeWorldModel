# Geometry-prefix physical consumer v2 — prefix v3 verification contract

V3 READY: the fresh `results/pusht_geometry_prefix_20261027_v3_cuda_v1/report.json`
is `plans_sealed`; real consumer `prepare` and model-free CLI preflight passed
against it with the complete 96-branch census. There is no remaining HOLD.

Consumer imports `pusht_geometry_prefix_20261027_v3.verify_plans` with the
unchanged report/source API. Physical basename remains v2. The main writer
diagnosed six v2 action-reference comparisons exceeding the fixed 1e-7 bound
because NumPy float32 tanh itself approximates the reference; v3 uses float64
tanh with the SAME 1e-7 bound. This consumer never reconstructs commands and
adds no tolerance override. Old v2 report status and source are untouched;
an `integrity_fail` report is still rejected, not relabeled or admitted.

Known-root calibration only: four official roots, geometry and stopK arms,
twelve saved prefixes per arm, one physical replay each (96 branches, at most
960 selected-path step attempts). This is not P1 admission, efficacy evidence,
a total-budget comparison, or permission to train/change outcomes or gates.

The consumer imports and calls planning `verify_plans(report, source)`; it
does not implement gradients, generate candidates, call predict, or reselect.
It independently binds each verified plan, including its saved candidate and
selected/greedy IDs, with `selection_sha256`. Commands are consumed verbatim.
Planning source seals must verify at launch; no running planning CUDA CLI is
needed for consumer development. Main completed the fresh v3 host-CUDA rerun;
the physical run remains exclusively main-scheduled.

`prepare` follows the full-packet consumer: official `validate_inputs`, exact
protocol bindings, every planning source seal and manifest/args binding,
planning report hash, unique intervention bank/upstream hash, and exact four
RGB references. Reset checks use the whole three-frame stack and goal RGB.
All loaded experiment/model source dependencies, every planning seal source,
shared physical helpers/guard, installed evaluator, this consumer, its test
and this document are sealed and verified before collecting physical roots.
Inputs/plans are checked again before physics, optional encoding and at exit.
Existing planning/source seals are read-only; only NEW output seals are made.

Outputs must be in a NEW directory whose basename starts with
`pusht_geometry_physical_20261027_v2`; existing paths/files are never reused.
The protocol, planned census and physical JSON use exclusive writes + fsync.
Shared `durable.physical_branch` journals/fsyncs each attempt before env.step,
then each return and observed RGB. PNGs are exclusive/fsynced too; directory
entries are synced after each branch and before encoding. Failures retain all
96 slots, with unexecuted branches materialized as blocked. Both
`all96branchcensus.planned.json` and `all96branchcensus.physical.json` enumerate
every root/method/selection hash; the final report contains the same census.
Reset/warmup and root collection costs are separate from path step counters.

Only after the complete durable physical report and all 96 physical files
match the census may optional host-CUDA encoding begin. The shared endpoint
encoder reads hash-valid journal/PNG observations at EXACT step 5 (prefix) and
10 (terminal). At most 192 endpoint samples; no context encoding, prediction,
padding, shortened-endpoint substitution or CPU neural fallback. Guard-aborted
branches keep their prefix/terminal slots, blocked if the endpoint does not
exist. A saved step-5 endpoint remains eligible after a later guard abort.
Even an endpoint rendered at an abort is an observation, not a guard success.
Encoder/setup failures retain physics and per-endpoint failure accounting.
The model is frozen eval, predict is forbidden, and state hashes are checked.

## Commands

Run from `/home/gbk/code/LeWorldModel/le-wm-main`. Supply the finalized sealed
planning report at the exact path below. Main must first finish the fresh v3
run with status `plans_sealed` (now verified); these commands never bypass it.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_pusht_geometry_physical_20261027_v2.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_pusht_geometry_physical_20261027_v2.py tests/test_pusht_geometry_prefix_20261027_v3.py
.venv/bin/python -m experiments.spectral_reachability.pusht_geometry_physical_20261027_v2 --planning-report results/pusht_geometry_prefix_20261027_v3_cuda_v1/report.json --output results/pusht_geometry_physical_20261027_v2_preflight --preflight-only
```

Preflight performs input/seal/plan validation and prints the explicit 96-branch
census; it neither creates output nor makes an environment or loads a model.
The tests are nonneural contract fixtures, not physical efficacy evidence.
V3 integration validation: 46 consumer contracts plus 12 existing prefix-v3
tests (58 total), including the actual sealed-v3 prepare/preflight test.
The real-artifact test forbids environment/model construction and CUDA
allocation and checks that preflight creates no output directory. The
saved-CUDA action regression reads existing
artifacts only; it does not run a neural model or alter the v2 failed report.

Only the main host-CUDA scheduler may launch the following, with a fresh output:

```bash
.venv/bin/python -m experiments.spectral_reachability.pusht_geometry_physical_20261027_v2 --planning-report results/pusht_geometry_prefix_20261027_v3_cuda_v1/report.json --output results/pusht_geometry_physical_20261027_v2_cuda_v1 --encode-terminals
```

Without `--encode-terminals` this records physical receipts only. Blocked slots
are retained even when final integrity fails; do not interpret retained census
as complete physics or complete encodings. Source checks remain content seals,
not continuous monitoring or proof of execution against adversarial mutation.

## Freeze after nonneural tests

Freeze only the three new owned files after the nonneural test run. Record
their exact SHA256 values with this command; the execution seal additionally
freezes all current dependencies/test/doc before physics:

```bash
sha256sum experiments/spectral_reachability/pusht_geometry_physical_20261027_v2.py tests/test_pusht_geometry_physical_20261027_v2.py docs/pusht_geometry_physical_20261027_v2.md
```

Real finalized-v3 `prepare` and CLI preflight were run during consumer
finalization; neither allocates a model nor executes physics. No host-CUDA
experiment or physical execution was run by this consumer developer.
The source-seal doc path `Path(__file__).resolve().parents[2]/'docs'/...` is
intentional: this owned document actually lives in `le-wm-main/docs`, not the
outer repository's `docs`. Both the path and sealed-file inclusion are tested.
Do not reuse earlier source hashes after any edit. The latest freeze is a NEW
directory; previous source freezes and the original failed v2 report remain
unchanged. A source freeze alone is not permission to execute physics.
