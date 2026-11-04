# Reacher P1 robust accounting recheck

This is a development plumbing recheck, not P1 admission or independent
confirmation. It uses the previously observed seeds 20263160 and 20263161,
and the witness threshold is one for this diagnostic. The frozen qpos
readouts were separately trained, so this run is not evidence for the
training-free Planning Compiler main method.

The robust selector previously omitted the two perturbed prefix rollouts and
their subsequent reachability computations for every eligible candidate from
its reported model-work ledger. The accounting fix does not alter selection.
For seed 20263160, the same candidate index 5 is selected before and after
the fix; 24 eligible candidates receive two perturbation evaluations each.
The selector receipt changes from 1152 to 1536 forward rollout evaluations,
2048 to 2768 model-predict sample evaluations, and 512 to 704 backward VJP
rows. These counts describe the selector only, not total GPU cost, wall time,
encoding, physical execution, or a certified equal-budget comparison.
Seed 20263161 has no eligible candidate and its reported selector counts
are unchanged.

The host-CUDA recheck passed its declared plumbing checks; it does not show
that the P1 mechanism holds in physical task coordinates or beats baselines.
The old immutable receipt is not modified:

- Original: `le-wm-main/results/reacher_d2p_p1_preflight_zero_reference_20261102_cuda/report.json`
- Recheck: `le-wm-main/results/reacher_d2p_p1_preflight_accounting_recheck_20261102_cuda/report.json`

Selector source changes invalidate source-hash bindings to older preflight
receipts. Fresh, contract-matched preflight and calibration are required for
any future outcome run. Neither this recheck nor the existing separately
trained readout branch satisfies the main training-free claim.
