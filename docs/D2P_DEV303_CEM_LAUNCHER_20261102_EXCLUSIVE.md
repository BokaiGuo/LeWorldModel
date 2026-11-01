# dev303 full-horizon CEM exploratory launcher

This launcher connects the previously added CEM and native host bridge. It is
one exploratory run on the already explored dev303 root, not an independent
confirmation, an official external-method reproduction, or an equal-budget
comparison. No author-reported performance is consumed.

From `le-wm-main`, with a fresh ASCII-alphanumeric tag:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m experiments.spectral_reachability.dev303_cem_launcher_20261102_exclusive prepare --tag TAG
```

Preparation reads fixed source receipts, fixed source images, the pinned task,
the checkpoint files, and listed source files; it does not create a model or
environment. It writes an exclusive preparation file and prints its SHA-256.
It does not authorize an experiment. The launcher requires the existing regular,
single-link cooperative lock file, which must never be replaced; the lock is
advisory and only coordinates cooperating processes.

Only after separate authorization of both CUDA and physical execution, a
future operator may use the printed prepare digest:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m experiments.spectral_reachability.dev303_cem_launcher_20261102_exclusive run --tag TAG --prepare-sha256 PREPARE_SHA --authorize-model-cuda --authorize-development-physics
```

Run requires CUDA allocation, the pinned frozen checkpoint tensor identity,
exact native environment source, one reset, the dev303 reset/goal identity,
ten actual zero warmup steps and exact sealed 0/5/10 state, snapshot and RGB
checkpoints. It encodes all three actual warmup frames and the goal anew.
The full CEM plans H3 at step 0, executes five commands, encodes actual CP5
feedback, then plans H2 and executes up to ten more commands. Native termination
stops immediately; no retry or padding occurs. Candidate banks, scores, elite
updates, replay, individual attempted/returned model calls, dispatch/return,
frames, latent files and setup counts are recorded. Initial and CP5 encoder
timings synchronize CUDA; per-model-call bridge times are host-side and
unsynchronized, not CUDA kernel time. Model-internal calls, FLOPs, energy and
complete transitive software/hardware provenance are not measured or certified.
I/O failure can leave an incomplete directory without a final report.

The saved report digest printed by the run is an external input to:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m experiments.spectral_reachability.dev303_cem_launcher_20261102_exclusive verify --tag TAG --prepare-sha256 PREPARE_SHA --report-sha256 REPORT_SHA
```

Verification is saved-only: it checks the listed current source bindings,
external digests, journal chain, warmup and control order, saved frame and
latent digests, CP5 ordering, optimizer replay with saved scores, winner
selection and outer-call counts. It does not re-run the model or physical
environment and cannot prove that a hostile host honestly executed the logged
calls or authenticate hardware. A partial/failed run is preserved but is not
accepted as complete by the verifier. The 32 MiB output cap and disk reserve
are inherited from the existing native store.

This entry point has only been exercised with CPU fixtures and source preflight.
Its actual CUDA/PushT execution and real saved output remain unvalidated.
