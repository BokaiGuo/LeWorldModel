# Fixed-prefix finite-difference kernel (not a CUDA host)

For all 64 fixed groups, use the already-paid suffix-iterate-0 gradient and
exact same fixed model-predicted midpoint; no group is selected by stage or
physical result. For each of 10 coordinates, evaluate centered plus/minus
perturbations at the two fixed requested scales `2^-8` and `2^-10`, separately
in raw action and float32 chart-logit spaces. This is 40 pairs / at most 80
forwards per group and 2560 pairs / at most 5120 new forwards in the full panel;
new scalar AD calls are zero. Existing parent gradient expenses are inherited,
not refunded. Invalid pairs remain explicit nulls, not replaced or clipped;
actual attempts/completions must be metered in any host.

Actual float32 plus/minus distances and denominator are saved, not assumed
equal to the requested value. Raw probes crossing [-1,1], chart probes
crossing the original float64 ball, or quantization collapse are null. Chart
probes do not apply the projection, so this does not estimate a derivative of
the nonsmooth projector; norms, boundary and the null pattern are reported.
The 64 selected gradient points are fixed before observing differences.

The original saved gradient is of a native float32 loss; scores here are the
saved-protocol endpoint identity-Q MSE accumulated in float64. Differences
therefore include precision/reduction and finite-step error; two scales do not
prove a gradient bug. The chart-chain reference is separately reported from
the raw AD gradient and uses the frozen method's `1-raw^2` expression, not a
claim that float32 casting/tanh has an exact real derivative. No new tolerance,
P1 gate, result-based action or scientific admission is defined.

This is a callback kernel and RAM tests **only**. It is not a CUDA host,
authority prepare, durable native ledger/consumer, or production CLI. Real
production needs a separate own-schema host with full source strict, 64-group
saved native IO/ledger and partial failure, model/class/precision closure,
hard resource/output/time limits, and MAIN review. No CUDA, CPU neural,
physics or production run was executed.

Test: `.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_agent_p1_fixed_prefix_fd_20261029_v1.py`.
