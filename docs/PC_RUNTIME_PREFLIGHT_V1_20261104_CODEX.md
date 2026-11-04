# Runtime preflight for future PC development runs

The stopped v13 Reacher feedback panel wrote a manifest before discovering
that the sandbox did not expose NVIDIA devices. That panel remains stopped:
this preflight does not retry it or substitute a root.

`experiments/pc_runtime_preflight_v1.py` is an infrastructure-only entry
point for future protocols. It checks a real CUDA allocation and declared GPU
UUID, optionally creates the PID and network namespaces required by the
isolated workers, and optionally resets and renders the actual Reacher
environment using EGL. The render uses an infrastructure-only seed, takes
no planning actions, and produces no saved observations or panel results.
It writes no experiment manifest. Namespace creation alone does not verify
the full worker mount policy or model execution.

On 2026-11-04 the combined check passed outside the restricted application
sandbox on GPU `GPU-4e8973ff-0278-53f3-2bae-2615d461e441`: allocation sum
4.0, Reacher RGB `uint8` frame shape 224 × 224 × 3, and namespace creation
passed. This required explicit sandbox escalation. The worker isolation
policy itself has not been disabled. Gym emitted a render-mode metadata
warning; an RGB frame was nevertheless returned. Two unit tests verified
ordering and that a failing check prevents the provided start callback from
creating a manifest.

This is a standalone utility, not a retrofit of the sealed Reacher runner.
Future runners must explicitly call `run_before_artifacts` with all required
checks before any output directory, manifest, task root, or scientific
observation is created. An isolated-worker experiment also needs its own
complete worker/mount and model preflight. The utility does not validate
task feasibility, method correctness, source independence, or PC0–PC5.

Example infrastructure check, subject to the host's permissions:

```sh
CUDA_VISIBLE_DEVICES=GPU-4e8973ff-0278-53f3-2bae-2615d461e441 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl .venv/bin/python -m experiments.pc_runtime_preflight_v1 --gpu-uuid GPU-4e8973ff-0278-53f3-2bae-2615d461e441 --isolation --reacher-render
```

Run from `le-wm-main`. Do not infer that this command ran an experiment.
