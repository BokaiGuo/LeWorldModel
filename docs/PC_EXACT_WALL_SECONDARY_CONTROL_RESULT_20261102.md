# Exact wall secondary controls: development result

The post-pilot protocol is `PC_EXACT_WALL_SECONDARY_CONTROL_PROTOCOL_20261102.md`.
This is one deterministic analytic task, not an independent benchmark or a
learned-world-model experiment.

| Method | Terminal distance | Objective evaluations | Simulated transitions |
| --- | ---: | ---: | ---: |
| Random bridge | 0.0 | 64 | 768 |
| Finite-difference L-BFGS-B, four starts | 0.5001078828 | 8,904 | 106,848 |

Eight of the 64 bridge candidates reached the goal to numerical tolerance.
Replaying the selected bridge reproduced zero terminal distance and zero
blocked transitions. All four L-BFGS-B starts terminated abnormally;
this nonsmooth case does not establish a general gradient-planning failure.
The resource counts are not a complete equal-budget comparison.

The initial v1 report stored SciPy's reported `fun` as the returned program's
cost, but replay of its selected program gave 0.5001078828 rather than
0.5000989398. On abnormal line-search termination these values can refer to
different points. The runner now explicitly scores each returned program,
counts those four extra evaluations, preserves `optimizer_reported_fun`, and
selects by the replayed cost. The original v1 output is retained; v2 is the
corrected result. Seven secondary and parent dynamics tests passed.

The random bridge control solves this particular detour without explicit wall
geometry, so the earlier failure of the tested CEM and beam configurations
does not show that a structured compiler is necessary. No D2P program,
future-control-capacity mechanism, fresh roots, or real RGB feedback was
tested here; P1 and the overall research closure remain unproved.

Saved outputs:
`le-wm-main/results/pc_exact_wall_secondary_development_20261102_v1/report.json`
and `le-wm-main/results/pc_exact_wall_secondary_development_20261102_v2/report.json`.
