# Cell10 cooperative exploratory feedback entry

This entry targets `r1.g0.h3.dDxy`, an existing known root and saved control5.
It starts the existing native host in a fresh interpreter after an exclusive
cell10 marker and the existing CPU checkpoint/source precheck. The original
runner owns its physics, encoder, numerical ledger and publication. Nothing
in this entry changes scientific admission or turns a transport-complete
early stop into a complete closed loop.

The operator must coordinate with other local users and check the relevant
attempt history before executing. Only cooperating callers using the same
marker are excluded; old entrypoints, direct imports, other marker names,
other hosts and filesystem aliases can bypass it. No global fence or
cross-host exclusion is claimed. If the marker or output exists, do not
delete or retry automatically: an explicit recovery decision is required.
Actual GPU-loaded modules must still pass the native runtime census; the CPU
precheck cannot prove its full runtime closure.

The parent journal is separate from the native output. It records monotonic
phase boundaries, return codes and exceptions, fsyncing each record. It does
not retain subprocess stdout/stderr. Failures opening or fsyncing the journal
can leave a claim without a complete journal. The native runner's own durable
partial/output records must be inspected independently. Parent journal data
is not in the native runner's 512 MiB cap; disk used by upstream control5 and
checkpoint/cache reads is also outside it. The parent's 1800-second timer
starts just before claiming and bounds each direct child by remaining time;
it does not include interpreter startup/import before `run`, and terminating
the direct child on timeout does not prove termination of descendants.
GPU time, FLOPs, device energy, peak resource use and full model-loading cost
remain unknown. This is not an equal-compute comparison.

Only a coordinating operator, after checking these limits and existing
attempts, may choose to run:

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -m experiments.spectral_reachability.e3_cell10_cooperative_run_codex_20261030_v1 \
  --confirm CELL10_SINGLE_HOST_EXPLORATORY_ONCE
```

The command above is production and was **not run** in this work. Do not set
`CUDA_VISIBLE_DEVICES` to an empty value or `-1` for an actual authorized
attempt; GPU/device selection remains the operator's responsibility.

Offline fixture tests, without GPU or physics:

```bash
env CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_e3_cell10_cooperative_run_codex_20261030_v1.py \
  tests/test_e3_cell10_durable_claim_codex_20261030_v1.py \
  tests/test_e3_live_wiring_cpu_codex_20261030_v1.py \
  tests/test_agent_e3_continuous_dispatch_20261029_v1.py
```
