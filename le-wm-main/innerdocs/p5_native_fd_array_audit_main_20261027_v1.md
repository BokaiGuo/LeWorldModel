# Main independent native finite-difference saved-array audit

No compiler numerical helper, model, torch or simulator import. Pure NumPy
rebuilds two fixed central-difference matrices from forty saved successor
outputs and checks all forty perturbed float32 inputs. Recomputes matrix errors
against saved JVP; verifies SHA-chain dispatch order, rounded displacement,
counts, unchanged model identity receipts and all artifact hashes.

The actual host-CUDA run has relative errors .0003638283/.0005198091 for
eps=.01/.005. These are descriptive, with no fitted acceptance threshold.
Saved arithmetic agreement is not a new neural run, exhaustive source audit,
physical effectiveness, P1 mechanism or P5 admission.

CLI requires an explicit receipt FILE SHA and exclusive new output JSON:
`python -m experiments.spectral_reachability.p5_native_fd_array_audit_main_20261027_v1
--receipt PATH --receipt-sha256 SHA --output NEW_JSON`.

Nine arithmetic/input-validation tests passed before the actual saved audit;
actual report has 91 checks all true. Readout of zero-norm directional error
is null rather than a falsely perfect match. File hashes bind bytes, not causal
truth of the original neural runtime.
