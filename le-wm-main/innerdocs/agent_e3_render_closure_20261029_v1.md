# E3 render dependency preseal: fixture-only

The prior CPU03 authority retains the separately presealed model-startup source set. This independent extension does not construct or load a neural model. It imports the same gym registry, stable_worldmodel and PushT class as the native factory, checks the original frozen PushT source hash and registration, calls Pillow's full plugin initialization, and sends a synthetic 224x224 RGB array through the real E3 `CaptureStore.frame` PNG path and reads it back. It does not construct an environment, reset, render a real environment, step physics, call an encoder, or run CUDA. It snapshots project/venv file-backed loaded modules and seals canonical file hashes.

Real artifacts, exclusively created:

* Source authority `results/agent_e3_startup_authority_20261029_v1_CPU03.json`, SHA `61a778391cc5ba2e89ca2b478c2901504f3a3d152364bc8af2e9a7a83017ee46`.
* Supplemental closure `results/agent_e3_render_closure_20261029_v1_FIX01.json`, SHA `77ecec9c288956cc76f9fef6263a0b2962bf2cfac74e7904130530fbd28aa863`; 3083 names, 3051 paths, including `PIL.BmpImagePlugin`.
* Reconstructed authority `results/agent_e3_render_authority_20261029_v1_FIX01.json`, SHA `4214c43495376c158378520f7e4ef92cda1d7165e3c4594417288955685eafc3`.

`collect`, `prepare`, `verify` and `precheck` are available from `python -m experiments.spectral_reachability.agent_e3_render_closure_20261029_v1`; the CLI requires explicit source/closure/authority paths and SHAs. The fresh-process precheck passed, rehashing and reconstructing the complete authority and repeating the fixture dependency probe. The module intentionally has no production `run` entry; precheck is a separate operator step, not a guard automatically enforced by the old host. Retain both closure and authority with their SHAs for audit. Original runtime source guard and before/after module census remain unchanged.

This is not an exhaustive environment/render closure: constructor, real backend rendering, image formats chosen by actual runtime, other lazy imports, unloaded transient modules, standard library and non-project native libraries are not observed. New modules loaded in a later actual run can still fail census; passing this fixture cannot certify a future run. Neither the failed MAIN02 evidence nor its scientific status changes. No claim of production readiness, full saved strict, or scientific admission is made. The original CPU03 model-probe closure was reused as sealed input, not re-executed during this task.
