# Constrained search: saved-result numerical audit

On the SHA-bound completed development report, independently recompute all
strict selector flags, tie ordering and null selection, validate
normalized/raw action conversion and original action bounds, and verify
candidate indices, counts, model and plan postchecks. Check that saved
endpoint-replay scores agree with the compiler receipts.

The original draft also proposed independently recomputing Q from saved
predicted coordinates. Inspection before implementation found that the
goal latent coordinate was not saved in the report or its Q source.
Consequently the CPU-only audit must mark that check **unavailable**, not
infer or fit a target from observed scores. A separately sealed follow-up
may reconstruct the same goal RGB and frozen checkpoint, perform exactly one
goal encoding per root on host CUDA, save the resulting target and then
recompute all endpoint scores. All source and root hashes must match before
this check, and it must not change the original search report.

This CPU calculation does not run a neural model or physics, inspect new
outcomes, choose a new candidate, change a gate, or verify full runtime
provenance. Its role is to detect saved-score, selector, and action-chart
bugs before further experiments. A success only supports the saved numerical
claims, not their scientific correctness or independent generalization.
