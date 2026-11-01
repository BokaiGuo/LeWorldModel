# Observed-history direct CUDA calibration v1

NEW driver/tests only; no old sealed compiler, source, direct core or evidence
modified. This implementation thread runs only NumPy/metadata tests and source
audits. It does not run CUDA, autograd, image/model inference, resets or physics.

## Actual observed-source audit and sample boundary

Bound actual sourcepack SHA:
`0b89d77494acc6cf03bb0110e87231c1c240a253e844326acc668b7e4f93d546`.
Bound observed producer prepare SHA:
`ecc5f4a488290b6d0eade4d3ff2034d256729de94c4d3a14b12d6eebad783385`.
Full successful two-row terminal is REQUIRED before any model dispatch. Checks:
external SHA, producer source/input bindings, complete hash-chain journal,
nested attempted/completed operation pairing, per-reset/per-public-step actual
physics counts, every checkpoint SHA/schema/state/RGB/goal/snapshot, every final
source SHA/schema, all10 zero actions and nonterminal/nontruncated receipts.
Both original selections, actual goal frame and ancestral source SHA are bound.
No old padding loader, fake task IDs, old source census/admission call or simulator
execution is used. Legacy parent NPZ arrays are read only to verify ancestry and
numerical equality; the NEW observed loader is used for actual bridge inputs.

Read-only audit passes for this actual run: 4reset/20public/6explicit render/
8setter/4goal render/**208 actual internal physics steps**, all attempted=completed.
Both roots' observed at0/5/10 state/RGB are identical; final root SHA equals its
parent SHA. RGB/state/raw-action arrays also match the old padding NPZ numerically.
These are nevertheless real observed frames backed by public steps and checkpoints,
not padding relabelled as observations. **No new independent roots/samples** are
created (`novel_root_count=0`, `independent_sample_increment=0`); this remains the
same two parent seed families, not N=4 or a new confirmation panel. The stationary
history is evidence against a padding-vs-observation numerical difference on THIS
batch, not a general demonstration that padding is valid or the model is accurate.
Zero controls/distribution are not changed to manufacture motion or novelty.

Native and wrapped terminal diagnostics are retained for at0 and every public
step, preserving native front4 position criterion. Actual discrepancies=0 in this
batch. The known abs-angle>2pi native bug is NOT repaired or overridden; a failed
source, including native false-positive safety rejection, would fail this gate.
Contact geometry is checked through saved source-domain receipts; this audit
does not rerun a simulator to independently reconstruct collision distances.

## New bridge and fixed direct protocol

Four RGB-only native encoding calls per root (actual0/5/10 plus original actual
goal frame), total **8 encoder +8 projector samples** even when RGBs are identical.
The existing sealed `screen.native_image_encoder` and native ImageNet/Resize224
transform recipe are used. No caching to discount repeated-frame acquisition.
Transform/model/library/source bytes are hash-bound; source-closure files are
hashed in place and NEVER copied into prepare.

Task root_context has actual encoded at0/5/10 native projected CLS192 states and
the two normalized time-major action packets between them. Saved raw commands
are retained and normalization is rechecked exactly float32. Public constants are
source-bound but checkpoint training-normalizer provenance remains unverified.
Privileged state/pose/hidden velocities only support provenance; the encoder sees
uint8 RGB only, and direct optimizer receives only the validated latent/action task.
Same Q=I192, same original goalRGB, original model/config, H2=2 packets of5 actions,
raw tanh commands and per-packet logit L2 cap `.75*sqrt(5)`.

Two methods: `full_H2_GN` and `full_H2_gradient`, B1=4/B2=8, one zero restart,
step_scale=1, GN damping=.002, zero extra line-search trials. Reuse the sealed
`screen.NativeCuda` genuine joint H2 derivative and `screen.run_direct`, unchanged.
No CPU neural fallback, training, parameter update, environment construction or
new physical steps. Existing NumPy SVD/solve algebra remains as in sealed core;
this is not a claim that all algebra executes on GPU.

| Phase/resource | Exact successful budget |
|---|---:|
| Previously paid source-only acquisition | 4 resets,20 public steps,208 physics; separately reported |
| New native visual acquisition | encoder8,projector8 |
| New direct predict/action_encoder | 280 each |
| New direct Jacobians / mathematical VJP directions | 32 / 6144 |
| New SVD / linear_solve | 16 / 16 |
| New proposals / selections | 32 / 32 |
| New physics/training | 0 / 0 |

Per root-method:70predict/70actencoder/8J/1536VJP/8proposal/8selection; GN adds8SVD
and8solve. The70 includes three separately recomputed H2 replay paths per evaluated
candidate plus two ADforward packets per J. VJP directions are not kernel/FLOP
equivalents. Source acquisition is not silently charged again as new direct work.

## Small prepare, hard disk gate and durable evidence

prepare is one small JSON containing paths/SHA/config/source audit, no full raw
source, Q192 arrays or copied dependency/weights closure. It requires only200MiB
plus its own bytes. It can succeed while CUDA is **not** allowed for space.
Run requires free **at least256MiB =56MiB complete output cap+200MiB floor** before
creating output/importing CUDA. MAIN must recheck immediately before launch. No
automatic reduction of budget, cap/floor, roots, saved evidence or cleanup/deletion.
If insufficient: remain prepare-only, request external storage resolution.

Shared journal preserves every original screen event losslessly as JSONL and
fsyncs each append: native packet/context paths, all raw J rows, full logit J,
gradient, proposal, selection trajectory, attempted/completed module work and
terminal report. No core changes or evidence thinning. Reports/tasks/image journal
and all component journals share the same56MiB cap; every write checks actual
recursive output size and200MiB floor. Unlike old8MiB-per-component sink, this
does not erroneously cap an8-iteration component at the earlier3-iteration limit.
If cap/floor is hit, run is partial_or_failed, no retries. Source/model hashes are
rechecked in finally. Missing final report from an I/O failure is not completion;
stdout returns failure and already persisted partial receipts remain evidence only.

Top report keeps source acquisition, native images and direct phase separate,
model before/after hash, actual resources, root-equality/sample boundary, errors,
global_freshness=unknown, P1_admission=false/model_task_eligible=false.
**Eight iterations do not certify plateau/stationarity**, nor provide an independent
legal two-stage attainable witness, frozen confirmation thresholds, non-greedy
physical mechanism, or task efficacy. No old numerical equivalence is assumed
for the new CUDA run before actual terminal/saved-array audit.

## MAIN commands

From `/home/gbk/code/LeWorldModel/le-wm-main`. All output names must be NEW.
Metadata-only audit (optional):

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.spectral_reachability.p1_observed_direct_cuda_20261027_v1 audit-source --sourcepack results/p1_observed_history_source_20261027_v1_main01 --source-sha 0b89d77494acc6cf03bb0110e87231c1c240a253e844326acc668b7e4f93d546 --source-prepare results/p1_observed_history_source_20261027_v1_prepare01.json --source-prepare-sha ecc5f4a488290b6d0eade4d3ff2034d256729de94c4d3a14b12d6eebad783385
```

Metadata-only small prepare:

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.spectral_reachability.p1_observed_direct_cuda_20261027_v1 prepare --sourcepack results/p1_observed_history_source_20261027_v1_main01 --source-sha 0b89d77494acc6cf03bb0110e87231c1c240a253e844326acc668b7e4f93d546 --source-prepare results/p1_observed_history_source_20261027_v1_prepare01.json --source-prepare-sha ecc5f4a488290b6d0eade4d3ff2034d256729de94c4d3a14b12d6eebad783385 --output results/p1_observed_direct_cuda_20261027_v1_prepare01.json
```

MAIN-only actual CUDA (replace PREPARE_FILE_SHA with prepare command's file SHA;
gate failure must NOT be bypassed):

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.spectral_reachability.p1_observed_direct_cuda_20261027_v1 run --prepared results/p1_observed_direct_cuda_20261027_v1_prepare01.json --prepare-sha PREPARE_FILE_SHA --output results/p1_observed_direct_cuda_20261027_v1_cuda_main01
```

Pure tests:

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_p1_observed_direct_cuda_20261027_v1.py
```
