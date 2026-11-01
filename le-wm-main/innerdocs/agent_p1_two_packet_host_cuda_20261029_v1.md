# Two-packet four-slot host

This adapter binds the actual full mixedmode winner consumer via the separately
reviewed winner-authority component, including its original float64 solver
witness and literal float32 chart. It preserves all four known root/condition
slots and both fixed arms; no physical outcome is a planner input. Each arm
pays the same 36 native forward / 16 scalar reverse operations, for 288/128
across eight arms. Identical counts are not equal FLOPs or equal wall time.

Native IO and exact operations are journaled per arm. Complete saved consume
reconstructs the original whole source authority, validates every native IO and
AD/callback, replays the algorithm from saved outputs, checks both arms' exact
same initial context/winner, full output census and model/source/precision
closure. Independently, MAIN's pinned scalar audit recomputes all 16 stage and
suffix AD identity-Q scalar losses per sealed arm from saved endpoint and goal;
its float32 reduction bound is not a gradient or model-parity claim. Its source
and test are bound in the preparation. First error stops execution; failed and unstarted slots remain in the
fixed 4/8/72 denominator with no retry or fallback. Scalar reverse, VJP and
callbacks are nested axes and must not be summed. No gate or training change.

Output limit 512 MiB including 64 MiB final reserve, disk floor 200 MiB,
wall watchdog 3600s and allocated/reserved GPU limits 3/4 GiB; wall sample is
before final report fsync, not proof final persistence finished inside limit.
No claim is made that the output bound is a prior proof the run will fit.

```
.venv/bin/python -m experiments.spectral_reachability.agent_p1_two_packet_host_cuda_20261029_v1 prepare --source-output ABS_MIXED_OUTPUT --source-report-sha SHA --output ABS_NEW_PREPARE
.venv/bin/python -m experiments.spectral_reachability.agent_p1_two_packet_host_cuda_20261029_v1 verify --prepared ABS_NEW_PREPARE --prepare-sha SHA
.venv/bin/python -m experiments.spectral_reachability.agent_p1_two_packet_host_cuda_20261029_v1 run --prepared ABS_NEW_PREPARE --prepare-sha SHA --output ABS_NEW_OUTPUT --authorization MAIN_P1_TWO_PACKET_HOST_CUDA_ONLY
.venv/bin/python -m experiments.spectral_reachability.agent_p1_two_packet_host_cuda_20261029_v1 consume --output ABS_NEW_OUTPUT --report-sha SHA --receipt ABS_NEW_RECEIPT
```

MAIN must independently review and test before production. No physical release,
freshness, P1 success, or model improvement is certified by this adapter.
