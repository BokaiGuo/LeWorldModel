# Four-arm mechanism versus own-optimizer performance

This is a CPU-fixture-verified preparation component, not a launcher, fresh
experiment, physical authorization, strong-baseline certification, or proof
of budget fairness. It does not access the running spectral8 job or its output.
Task/model/hardware identity values supplied to `make_contract` are not
independently authenticated by this component or evidence of preregistration.

## Mechanism layer

`make_contract(task, config, model_sha, hardware_sha)` constructs a single
shared root/task/context/goal/normalizer/action-domain identity, an eight-ID
H3 policy and common raw-gradient suffix configuration. Call
`run_mechanism(leaf, arm, task, config, policy, contract, device=...,
emit=durable_sink)` with a separate already-initialized durable driver for
each arm and level `common`. The sink must persist before returning and fail
closed; this interface does not itself authenticate persistence. An actual
host launcher must verify checkpoint and root, precise feedback delivery,
source/dependency closure, device/precision, disk and wall caps, event
contents, and all terminal receipts.

Fixed initialization rules:

* `structured`: existing spectral8 H3 initializer, with its actual 192-row
  raw Jacobian, one SVD, original two radii/modes/signs and extra degeneracy
  rejection; it is not an original-R equivalence or P1 finding.
* `random`: eight seeded raw clipped-Gaussian samples, scale 0.4.
* `immediate_progress`: sixteen independently seeded samples from the same
  distribution; score the first packet for all sixteen and pass the best
  eight, sorted by `(score,index)`, to the common suffix.
* `native`: zero then seven seeded clipped-Gaussian samples, scale 0.4.
  This is a native-style *initialization variant*, not the installed CEM
  solver, a native performance baseline, or a certified strong baseline.

Every initializer uses the same task/latent history/action history/goal
contract; all remaining candidates receive the same per-candidate gradient
suffix and fresh H3 final replay. This isolates an engineering initialization
mechanism, not each method's best performance. All generated candidates and
each immediate-screen score are emitted before the corresponding later stage;
partial failures are not success. Exact seed and clipping are fixed in source.

For two suffix steps, source-derived complete-run explicit counts are:

| Arm | action-encode / predict samples and calls | backward / VJP rows | Jacobian / SVD | replay / selection |
| --- | ---: | ---: | ---: | ---: |
| structured | 75 | 208 | 1 / 1 | 8 / 8 |
| native variant | 72 | 16 | 0 / 0 | 8 / 8 |
| random | 72 | 16 | 0 / 0 | 8 / 8 |
| immediate progress | 88 | 16 | 0 / 0 | 24 / 24 |

These are prospective counts, not measurements. The common axiswise cap is
the maximum of arm requirements, not equal work, equal runtime, FLOPs or
energy. The underlying schema's `including_attributable_setup` scope is not
proof that upstream observation collection/encoding, matrix operations,
sampling, serialization or internal kernels are accounted. Reserve the whole
declared arm path before model dispatch; failures and over-cap conditions
remain failures and are not converted to successful endpoints.

## Performance layer

`check_performance_draft` separately checks a declarative set of four
arm-specific initializer and optimizer hashes, one shared feedback hash,
per-arm setup/initialization/search/final-replay/physical cost vectors and a
common cap. It reports each declared total and remaining cap without
equalizing them, and rejects extra outcome fields, different feedback and cap
overruns. It cannot verify supplied hashes, source semantics, completeness or
costs and always returns `implementation_ready=false` and
`fairness_proven=false`. It never launches an optimizer.

Real performance comparisons still need separate actual implementations of
each method's optimization (including a bound installed native CEM adapter
if that is the baseline), prospectively fixed configurations and feedback
schedule, equal accessible observations at each decision, actual event-level
metering of all model and physical calls, resource-compatible budget curves,
full final replay, source closure, raw action and RGB guards, and predeclared
failure/denominator handling. Shared gradient suffix results must not stand
in for those optimizer-specific results. Known roots are not independent
confirmation; no holdout is opened by this component.
