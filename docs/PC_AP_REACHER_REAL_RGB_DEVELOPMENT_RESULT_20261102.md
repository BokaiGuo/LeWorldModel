# AP-CEM observed-target vs final-goal: real-RGB development result

The predeclared two-root development protocol is in
`PC_AP_REACHER_REAL_RGB_DEVELOPMENT_PROTOCOL_20261102.md`. It does not use
official AP queries or memory, and it is not an independent evaluation.

The first run failed in all four arms before any planned physical action:
the `ObserveSteps` wrapper hid the goal image from the snapshot routine,
which rejected the incomplete scene binding. The original failed report is
preserved at `results/pc_ap_reacher_real_rgb_development_20261102_v1/report.json`.
The wrapper was changed to expose the unchanged goal to snapshot binding and
a regression test was added. This was an engineering failure, not a negative
control result.

After the fix, all four predeclared arms executed ten actions. Every pair
had identical initial RGB/goal hashes; each second decision used the RGB hash
recorded after its own first five actions. Frozen model parameters matched
before and after. Every block evaluated 34 model predictions. A saved-array
check reproduced all eight executed-action hashes; no arm reported physical
termination or truncation in these ten steps.

| Known root | Initial joint distance | AP target, terminal | Final target, terminal |
| --- | ---: | ---: | ---: |
| 20263160 | 1.003500 | 0.793302 | 0.810738 |
| 20263161 | 1.182600 | 1.244493 | 0.924930 |

Distances are evaluator-only joint-space distances, not a certified native
success metric. On the first root AP is lower by 0.017436; on the second it
is higher by 0.319563. These two roots have been used in development and are
not independent evidence. AP memory contains only three short episodes; its
relationship to the checkpoint training data and reset provenance is
unknown. Neither the 16-by-2 search nor this synthetic goal construction is
the original AP evaluation protocol.

Measured prediction counts are equal, but retrieval, memory construction,
encoding and wall time have not been certified as equal total resources.
After V2, the runner was extended to save per-episode memory RGB hashes and
its own source hash in future runs; the V2 report predates that extension and
does not contain those hashes. It must not be treated as source-sealed.
The goal offset of 25 is a specified retrieval-control input, not a measured
offset to the synthetic rendered goal. This run demonstrates the
execute/reobserve/replan plumbing under these conditions, not AP-CEM's
official performance, D2P reconfiguration, generalization, or P1 admission.

Artifacts:

- `le-wm-main/results/pc_ap_reacher_real_rgb_development_20261102_v2/report.json`
- `le-wm-main/experiments/spectral_reachability/run_pc_ap_reacher_rgb_physical_dev.py`
- `le-wm-main/tests/test_pc_ap_reacher_physical_dev.py`
