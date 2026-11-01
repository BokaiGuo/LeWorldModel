# Dynamic cp10 fail-closed transport

This is **not** a completed production CUDA writer or a complete saved consumer. `run_cuda` invokes the physical adapter first, before any torch import, output directory or backend dispatch. The current adapter always reports independent source authority and execution authorization false, so `run_cuda` always blocks. Even if those flags were changed, the function separately blocks because no reviewed actual CUDA backend or full saved consumer has been installed. A caller-supplied boolean must never be promoted to authorization. This prevents historical source or fixture data from being dispatched as a future run, but does not establish fresh roots or trusted time.

`Store` is a create-only, flat, byte-counted disk primitive with an 8 MiB final reserve, file and directory fsync, no overwrite and no refund. `record_fixture` is explicitly fixture-only and writes an error and persisted-prefix hash list on failure; it performs no inference. Reserve exhaustion, disk failure and arbitrary interruption can still prevent partial publication. This is not the real native writer's wall/GPU/resource enforcement.

`verify_saved` checks a caller-supplied complete file census and bytes (including orphan rejection), native seal and all 52 ordered events, then calls the new own-namespace native IO transcript validator. It does **not** authenticate the caller's prepared object, report, source/physical/producer receipts, model, wall clock, GPU limits or complete scientific authorization. The result explicitly sets `full_saved_consumer=false`. These missing checks, actual CUDA model backend/tracing/precision/resource accounting and true failure transport remain implementation work before MAIN can run. The old producer source hardcodes historical schemas and cannot simply be relabeled as future.

Temporary disk tests reuse the previous fake-tensor native fixture; these are not CUDA, model or physical evidence. No production, neural or physics is executed.

```sh
cd /home/gbk/code/LeWorldModel/le-wm-main
.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p3_dynamic_cp10_cuda_transport_20261029_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.agent_p3_dynamic_cp10_cuda_transport_20261029_v1 status
```
