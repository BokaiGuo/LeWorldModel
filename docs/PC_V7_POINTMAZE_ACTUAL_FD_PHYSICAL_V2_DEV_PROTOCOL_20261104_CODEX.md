# PointMaze actual probes: replay precision correction

The original diagnostic failed its native-prefix state check after converting
the historical warmup to float32. Preserve its partial manifest and original
script. This new run follows the original
[protocol](PC_V7_POINTMAZE_ACTUAL_FD_PHYSICAL_DEV_PROTOCOL_20261104_CODEX.md),
but retains the warmup actions in their recorded original precision; actual
dispatch and probe actions remain their recorded float32 values. Compare all
26 native prefix states with the earlier native replay at the original strict
tolerance. No physical probe outcomes were saved by the failed run. This is
an input-fidelity correction, not a change to the experimental hypothesis.
