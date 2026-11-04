# FD prefixes to actual-cp5 common CEM: development bridge

## Existing assets and missing link

The fixed FD report and its saved consumer bind model-selected prefix 4 and
parameter-zero prefix 0 for each of the two exposed roots. The physical v2
common-tail experiment has already used those roots; it did not replan its
suffixes. Existing `_two_stage` already replans from actual successor RGB and
executed history. This bridge calls that implementation once per fixed arm;
it does not introduce another optimizer or select a prefix using physical
outcomes.

The archived v2 report contains root and cp5 images and warmup actions, but
does not contain the three actual warmup history frames required by
`_prepare`. In particular the cp5 model RGB context must contain warmup step
10, observed root, and observed cp5, with the actual intervening action
packets. A cp5 image plus privileged qpos is not a substitute. The bridge
therefore reconstructs warmup via the existing collector from fixed packets,
checks root/goal RGB and warmup hashes, and uses the collector's actual
three-frame history. `_two_stage` physically advances each fixed prefix,
rebuilds its successor RGB/history, and calls the same existing CEM.

## Fixed development comparison

Both roots `[20263160, 20263161]`, arm order `[selected, parameter_zero]`,
prefix parameters/raw bytes, warmup packets and CEM configuration are fixed
by `prepare`. The two tails use the same solver and seed `root+5000`:
two packets, 16 samples, two iterations, four elites, 32 candidate rollouts
and 64 horizon-weighted predictor samples per arm. `parameter_zero` means
zero *parameter*, not necessarily raw zero action.

The bridge uses only the model report, its saved model audit, and the earlier
physical **action plan**; it does not read v2 physical outcome values to pick
or order an arm. qpos remains inside the existing evaluator-side physical
metrics and is not an input to CEM. The bridge checks frozen model parameter
hash, root/goal/history identity and the exact decoded prefix before physical
execution, then checks returned action witnesses, nominal CEM candidate count,
and model parameter hash after both arms. Each arm exception is retained.
The existing `_two_stage` also executes an evaluator-only zero counterfactual
per arm and computes observed-latent diagnostics; charge those operations,
encoders, source reconstruction, physical execution, wall time, and GPU memory
separately. The nominal CEM count is not total-compute equality.

## Host integration and boundary

`prepare(model_path, audit_path, physical_plan_path)` is CPU-only and returns a
fixed manifest. `run_root(...)` is the actual bridge to existing
`_two_stage`; it rejects CPU neural execution and unfrozen/non-CUDA models.
There is intentionally no autonomous launcher or execution command yet:
the host owner must first build an exclusive launcher and saved consumer that
seal the manifest and this code/dependencies, authenticate checkpoint and
action statistics, load the exact frozen CUDA model without a privileged-state
readout or wrapper, acquire the coordinated GPU
lock/device identity, reserve an exclusive output, persist root/arm failures
incrementally, and independently validate saved results. The root collector
must not be given a confirmation root. The host must coordinate any physical
execution and GPU use before running.

No GPU or physical execution occurs on import or in the tests. Unit tests
cover the sealed four-arm manifest, causal warmup binding, source tampering
and CPU neural refusal; they do not establish that the GPU/physical bridge
runs end-to-end. The existing `_two_stage` result records cp5 identity hashes
but not the complete three RGB frames/action context; an independently
verifiable live run still needs an exclusive instrumentation/launcher layer
that durably captures those actual inputs without adding privileged state to
the planner. These exposed roots are development only, the original
strict selector is null, and an observed suffix score or paired qpos outcome
cannot itself establish the P1 mechanism, independent generalization or
equal-budget superiority.
