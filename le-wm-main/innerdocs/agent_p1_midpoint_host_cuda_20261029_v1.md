# Actual-cp5 midpoint compiler host

This is a proposed model-only adapter for MAIN review; it does not alter P1,
grant physical release, train, or select roots by outcome. Four original
actual-cp5 slots and all nine original iterations remain fixed. Eight extra
midpoint evaluations per slot use the same original context, task and goal.
The exact strict midpoint rule and abstention are defined by the frozen method.

Per-slot native ledger: 26 predict and 8 scalar reverse; total 104/32, including
the 72/32 original compiler and 32 additional midpoint predict. Other native
axes are retained in full, nested resources are not FLOPs. The first failure
stops dispatch; all four rows remain, with failed partial and unreached nulls.
Output cap 256 MiB including 32 MiB final reserve, disk floor 200 MiB, wall
1800 s, CUDA allocated/reserved 3/4 GiB. This is not equal-budget comparison.
The saved wall sample is taken before the final report write/fsync; it is not
proof that the complete final report persistence finished inside 1800 seconds.

Prepare reconstructs the frozen observed recompile's actual full midpoint
authority and source closure, then binds this adapter and method. Run requires
the literal MAIN authorization and checks model/source/precision before-after.
The consumer checks full file census, native ledger, 26 native IO paths and
midpoint decisions. It first verifies the full real ledger, then derives a
read-only prefix view solely to invoke the original auditor's full optimizer,
AD IO, chronology and plan provenance checks; that view is not a receipt.

CLI from project root (supply actual frozen authority SHA values):

```
.venv/bin/python -m experiments.spectral_reachability.agent_p1_midpoint_host_cuda_20261029_v1 prepare --midpoint-output ABS_OUTPUT --midpoint-report-sha SHA --midpoint-receipt ABS_RECEIPT --midpoint-receipt-sha SHA --output ABS_NEW_PREPARE
.venv/bin/python -m experiments.spectral_reachability.agent_p1_midpoint_host_cuda_20261029_v1 verify --prepared ABS_NEW_PREPARE --prepare-sha SHA
.venv/bin/python -m experiments.spectral_reachability.agent_p1_midpoint_host_cuda_20261029_v1 run --prepared ABS_NEW_PREPARE --prepare-sha SHA --output ABS_NEW_OUTPUT --authorization MAIN_P1_MIDPOINT_HOST_CUDA_ONLY
.venv/bin/python -m experiments.spectral_reachability.agent_p1_midpoint_host_cuda_20261029_v1 consume --output ABS_NEW_OUTPUT --report-sha SHA --receipt ABS_NEW_RECEIPT
```

MAIN must audit and independently test before any run; fixture tests are not
evidence of actual CUDA, model quality, physical benefit, freshness or P1.
