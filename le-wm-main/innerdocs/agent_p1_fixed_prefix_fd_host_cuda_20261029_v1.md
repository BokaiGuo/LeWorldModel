# Fixed-prefix FD CUDA host

This own-schema host binds the actual fixed-prefix producer prepare, report,
persisted receipt and complete native/source strict, then preseals all 64
groups, 2560 pair slots and each group's valid-pair resource budget. The
producer full consume is reconstructed and must be byte-identical to the
persisted receipt. Source, tests, docs, installed class sources and frozen
model/class/precision are bound; zero model training, physical calls or new
reverse calls. Selector inputs are model-only. Inherited source verification
may access old physical lineage files; it must not feed physical outcomes into
the pair roster.

The fixed scales and both raw/chart coordinates are as in the frozen FD
kernel. Float32 actual denominators, boundary nulls and all pairs are preserved.
MAIN's read-only preflight reports raw 1280/1280 and chart 408/1280 valid:
if the same full authority is supplied, this implies 1688 valid pairs and 3376
new predicts; prepare independently recomputes it, does not assume these
numbers, and preserves 872 chart nulls. Each null additionally retains all
applicable explicit causes (input/output
float32 collapse, raw-domain crossing, or chart-ball boundary), recomputed
without altering the frozen kernel or hiding overlapping causes. The planned
cap is at most 5120 predicts, no new AD; an invalid pair is not an unpaid
successful forward.
Existing fixed-prefix 1216 predict/512 reverse and earlier ancestor resources
are inherited, not hidden. Reverse/VJP/backward axes are nested and no FLOPs
or equal-budget claim is made. Tangent or one-sided probes are separate future
experiments, not substituted here.

Engineering protections, not a guaranteed fit or runtime prediction: 8 GiB
total output including 512 MiB final reserve, 200 MiB disk floor, 14400 s
watchdog sampled through pre-report finalization (report fsync excluded), GPU
3 GiB allocated/4 GiB reserved. Full source verify time is in that sample.
No byte-census proof establishes that all full native IO necessarily fits.
Failure stops dispatch without refund; all 64 groups and 2560 pairs remain in
the report. Complete saved consume verifies every native IO and model math;
partial consume validates existing hashchains and ledger, does not invent
missing results. Failure before output binding or failure to persist despite
reserve may leave no consumable report. There is no automatic execution resume.

The worker ran RAM/temporary-disk fixture tests only, not actual prepare,
full strict, CUDA or physics. Tests replace `verify` with a fixture and cannot
certify upstream authority, model identity, device or precision. MAIN must
review and test before production. This is model gradient diagnosis, not
numerical proof of derivative correctness, fresh-root evidence or P1.

From project root:

```
.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_agent_p1_fixed_prefix_fd_host_cuda_20261029_v1.py tests/test_agent_p1_fixed_prefix_fd_host_component_20261029_v1.py tests/test_agent_p1_fixed_prefix_fd_20261029_v1.py
.venv/bin/python -m experiments.spectral_reachability.agent_p1_fixed_prefix_fd_host_cuda_20261029_v1 prepare --host-output /ABS/HOST_OUTPUT --host-report-sha REPORT_SHA --host-prepared /ABS/HOST_PREPARE --host-prepare-sha PREPARE_SHA --host-receipt /ABS/HOST_RECEIPT --host-receipt-sha RECEIPT_SHA --output /ABS/results/agent_p1_fixed_prefix_fd_host_cuda_20261029_v1_prepare_MAIN01.json
.venv/bin/python -m experiments.spectral_reachability.agent_p1_fixed_prefix_fd_host_cuda_20261029_v1 verify --prepared /ABS/results/agent_p1_fixed_prefix_fd_host_cuda_20261029_v1_prepare_MAIN01.json --prepare-sha ACTUAL_PREPARE_SHA
.venv/bin/python -m experiments.spectral_reachability.agent_p1_fixed_prefix_fd_host_cuda_20261029_v1 run --prepared /ABS/results/agent_p1_fixed_prefix_fd_host_cuda_20261029_v1_prepare_MAIN01.json --prepare-sha ACTUAL_PREPARE_SHA --output /ABS/results/agent_p1_fixed_prefix_fd_host_cuda_20261029_v1_cuda_MAIN01 --authorization MAIN_P1_FIXED_PREFIX_FD_HOST_CUDA_ONLY
.venv/bin/python -m experiments.spectral_reachability.agent_p1_fixed_prefix_fd_host_cuda_20261029_v1 consume --output /ABS/results/agent_p1_fixed_prefix_fd_host_cuda_20261029_v1_cuda_MAIN01 --report-sha ACTUAL_REPORT_SHA --receipt /ABS/results/agent_p1_fixed_prefix_fd_host_cuda_20261029_v1_consume_MAIN01.json
```
