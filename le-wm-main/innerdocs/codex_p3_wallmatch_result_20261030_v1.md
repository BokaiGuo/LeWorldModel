# P3 common-wall-cap expanded-baseline exploration

Six independently launched CUDA slots and twelve downstream physical branches
completed and passed their corresponding saved consumers. The model batch
report SHA-256 is
`cd1b9189bdd3ffbf1021ad86a234e1a2c653ce92eaedf97e39fdcd07386cdefa`.
Physical report hashes are P20
`460f54b9ec551f4aac31431ba593611e98dac15ac332daa7e5679e4c9a568168`
and P30 `aab0445f78c5ba9845ca581b9aa22dee83598ee8ff8ebdd159bf2bafca0546e1`.

The 30-second cap includes process startup, cooperation lock queue, model load,
computation and logging. At point 20 the uniform/signal/reverse wall times
were 21.596/23.189/23.102 seconds; at point 30 they were
24.692/24.998/25.424 seconds. Uniform evaluated 40/60 candidates, while
signal and reverse each evaluated 20/30 and paid their separately measured
signal cost. Uniform did not use all available time; this is not equal
observed time or proof of globally optimal use of the available budget.

At points 20/30 the common per-arm forward-axis caps are 80/120 and metric
caps 40/60. Actual uniform use is forward 80/120 and metric 40/60; each signal
or reverse arm uses forward 48/68 and metric 20/30. For every point each
signal/reverse arm used Jacobian 2/2, VJP 384/384, grad API 384/384, raw
backward hook 384/384, SVD 2/2, scheduler 1/1. Uniform used zero for the five
signal axes and scheduler 1/1. Forward counts are per axis for action_encoder,
predict, predictor and pred_proj, not a sum. See model report for each slot,
including memory, queue time and I/O.

Physical final object-position / wrapped-angle errors:

| point | root | uniform | signal | reverse |
|---|---|---|---|---|
| 20 | 0 | 70.787 / 0.476 | 65.096 / 0.441 | 70.787 / 0.476 |
| 30 | 0 | 56.604 / 0.351 | 67.312 / 0.695 | 70.787 / 0.476 |
| 20 or 30 | 1 | 76.403 / 0.082 | 76.403 / 0.082 | 76.403 / 0.082 |

Angles are radians, position errors use task coordinates. All 12 returned
10/10 planned steps with no termination or truncation; native success 0/12.
Per point, attempted/completed physical counts are construct 6/6, close 6/6,
reset 12/12, set-state 24/24, goal-render 12/12, goal-bind 6/6, public-step
120/120, physics-step 1224/1224 and explicit-render 132/132. Physical execution
wall 6.030/6.442 seconds excludes source reconstruction. Public-step includes
warmup and physics-step is nested; no axes are summed.

Root 1 uses the same winner in all arms and budget points. These repetitions
are separate executions of one known instance, not independent roots. The
physical roster was fixed during model execution and before inspection of new
winners, not preregistered before the model run. Both roots and earlier outcomes
were known. The common wall cap is a resource constraint, not equality of actual
compute, GPU isolation, FLOPs, or energy; neither method is scientifically
admitted. Saved validation is not a separate model or physical re-execution.
The copied kernel module docstring still mentions the old four-extra-candidate
scheme; its executable allocation, fixed manifest, tests and journal instead
use the expanded allocations above. The source is left unchanged after sealing.
