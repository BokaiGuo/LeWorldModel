# Isolated source-native PushT runtime and fresh calibration proposal

New files only. Archived episode1 blocked report remains immutable. Missing
archived hidden state is not a reason to reject freshly generated roots.
This delivery prepares an installer plan, a non-physics dependency/native-import
probe and a prospectively fixed fresh-root proposal, NOT a physics/neural runner.

## Runtime boundary and exact installer requirements

Use existing project `.venv/bin/python` with a separate target directory
`/home/gbk/code/LeWorldModel/le-wm-main/.jepa-physics-site`.
Set explicit `PYTHONPATH` to that directory for a standalone **non-neural**
collector/probe subprocess only. Do not replace/extend inherited PYTHONPATH.
GPU main retains its original interpreter/environment without this overlay.
No mutation of existing gym/pymunk, no native source patch or compatibility shim.

Version lock (not yet an authenticated downloaded-artifact hash lock):

- Runtime: gym0.23.1, pymunk6.8.0, scikit-image0.25.2.
- Additional missing transitive packages: lazy-loader0.4 and tifffile2025.6.11.
- Build tool: wheel0.45.1 for gym's small source package. Reuse bound existing
  setuptools81.0.0; disable build isolation to prevent hidden build downloads.

The five non-gym packages require binary wheels. Gym alone may build from
unmodified source. All installs use `--no-deps`; no torch downloads, extras or
whole-repository pip install. Prepare captures installed base dependency
versions/sites and recursively checks their active metadata constraints.
After installation probe recursively checks the **actual installed** locked
package metadata too; unexpected missing/incompatible requirements fail closed.
NumPy2.4.6/SciPy1.17.1/OpenCV4.13.0.92 and other base imports are reused, not
silently downgraded. Native import success alone does not prove runtime physics
or numerical/ABI behavior; main must execute the factual pilot separately.

Main commands, project root `/home/gbk/code/LeWorldModel/le-wm-main`:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m experiments.spectral_reachability.p5_jepa_native_physics_runtime_20261027_v1 bootstrap
```

This only prints exact argv/environment, not install. After review, create
fresh target and `results/p5_jepa_native_physics_runtime_20261027_v1/bootstrap_artifacts`
temporary directory (do not reuse partial attempts), then use its two argv:

```sh
PYTHONPATH=/home/gbk/code/LeWorldModel/le-wm-main/.jepa-physics-site PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 UV_NO_CACHE=1 TMPDIR=/home/gbk/code/LeWorldModel/le-wm-main/results/p5_jepa_native_physics_runtime_20261027_v1/bootstrap_artifacts uv pip install --python /home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python --target /home/gbk/code/LeWorldModel/le-wm-main/.jepa-physics-site --no-deps --no-cache --only-binary :all: wheel==0.45.1 pymunk==6.8.0 scikit-image==0.25.2 lazy-loader==0.4 tifffile==2025.6.11
PYTHONPATH=/home/gbk/code/LeWorldModel/le-wm-main/.jepa-physics-site PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 UV_NO_CACHE=1 TMPDIR=/home/gbk/code/LeWorldModel/le-wm-main/results/p5_jepa_native_physics_runtime_20261027_v1/bootstrap_artifacts uv pip install --python /home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python --target /home/gbk/code/LeWorldModel/le-wm-main/.jepa-physics-site --no-deps --no-cache --no-build-isolation gym==0.23.1
PYTHONPATH=/home/gbk/code/LeWorldModel/le-wm-main/.jepa-physics-site PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 .venv/bin/python -m experiments.spectral_reachability.p5_jepa_native_physics_runtime_20261027_v1 probe
```

Main must monitor combined target+staging size, limit **150,000,000 bytes**,
stop on exceedance and retain failure evidence. A printed install plan cannot
guarantee downloads' size, hard quota or success; probe refuses success above
cap. No network/install command has been executed by this agent. Probe records
installed target-file and imported dependency hashes before/after, origin
priority, source hashes, versions/active dependency constraints. This is not
authentication of a remote wheel/archive. Neither `probe` nor `prepare`
constructs/resets/steps Space/Body/PushTEnv or imports neural libraries.

## Fresh-root proposal (main must separately freeze execution consumer)

Four fixed seeds2026102701..2026102704, retain all, no replacement or favorable
contact/root filtering. Pilot seed2026102701 first. Official direct PushTEnv,
explicit constructor config in manifest: T, legacyFalse, no CoG/damping override,
relativeTrue, scale100, render224, with_velocity/targetTrue. Constructor's
nondeterministic seed(None) must be recorded then overridden with fixed seed
**before** reset. Root is post-reset, including native extra .01s physics.

Factual pilot: explicit raw-relative zero xy argument for15 control ticks,
NOT `None`, NOT normalizedzero. A tick holds current target with PD damping;
it does not freeze velocity/contact dynamics. Expected per trajectory:
1reset/15step calls,151physics substeps/1.51s including reset,16RGB observations.
Repeat from an independently constructed same-seed/config environment and
check exact state/full-public-state/RGB identity; retain mismatch, stop before
using any counterfactual branch. Repeat doubles these counts.

Unit distinction: native env argument is `u`; dataset raw pixel displacement is
`100*u`; predictor normalized action is `(u-mean)/std`, equivalently
`(raw_pixel/100-mean)/std`. Never divide env argument u by100 again.

Capture lossless RGB, float64 body/shape/space states, RNG state, raw targets,
contacts/coverage, source/runtime hashes and every timing/count receipt before
opening outcomes. Keep native float32 observation/proprio separately. Public
body properties are not complete Chipmunk contact-cache serialization: preserve
live roots or independently replay deterministic seed+prefix and verify branch
identity; do not claim that seven numbers or a property JSON restore solver
state. Full snapshot requirements and this boundary are explicit in manifest.

Only then propose a separate pre-outcome fixed **nonzero** native-raw factual
schedule for action-fidelity at contexts0/5/10,target15. All-zero actual/reverse
packets coincide, so zero-only factual receipts cannot establish action
fidelity. Official encoder/proprio/dynamics consumer would use actual vs three
fixed controls, preserving history/proprio; actual factual target is not a
counterfactual ground-truth label. No GPU work in this runtime process. No
future control/benefit/P5 claim, no reuse of v9 observed-target fitting as planning.

Tests and prepare:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_p5_jepa_native_physics_runtime_20261027_v1.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m experiments.spectral_reachability.p5_jepa_native_physics_runtime_20261027_v1 prepare
```
