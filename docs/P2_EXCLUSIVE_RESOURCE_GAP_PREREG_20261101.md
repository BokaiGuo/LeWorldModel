# P2 six-slot saved-resource gap audit: preregistration

Scope: CPU-only, read-only analysis of the already observed two-root H15
six-slot saved pairing. This is a retrospective diagnostic preregistered
before running the new audit, not a preregistration of the original study
or a fresh cohort. No outcome-based selection, model, GPU, training, or
physical execution.

Input is exactly `le-wm-main/results/p2_saved_fair_pair_20261029_MAIN02.json`,
expected SHA-256 `b45348e4dd8e74ee735d6c165148539b346fd21dd3b9aeb8e552ad524ec0ae1b`.
The audit requires the original schema, all 18 combinations of roots 0/1,
checkpoints 5/10/15, and unordered pairs of joint, composition, GRASP;
duplicates, missing entries, malformed resource values or digest mismatch fail.
The two original incomparabilities must remain missing; the input's original
`comparable` flags are reported, not independently validated.

For each arm/root, require the resource vector to be identical across its
three appearances. Compare joint (a no-composition iterative control) with
composition on `predictor_samples`, `AD_primal`, `VJP`, `J`, and `FLOPs`.
Report unknown values separately from mismatches; never treat null as zero.
Also report resource axes absent from this saved vector, including complete
physical, wall-time and total-compute accounting. No equality or scientific
admission follows from equality on these five fields.

Output is a newly created exclusive JSON file with input and implementation
hashes, full per-root vectors, checks and `P2_admission=false`. It does not
certify upstream receipts, temporal sealing, resource instrumentation,
candidate-level equality, actual outcome comparability, or hardware. The
pre-existing physical pair data are not recomputed. Independent fresh roots,
complete common-protocol instrumentation and P1 gate remain external blockers.
