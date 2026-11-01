# Independent prospective native ridge saved-array audit

Main-authored pure NumPy arithmetic, not a neural or physics experiment. Source
preparation binds this module, tests and doc before the actual CUDA receipt is
available. Audit takes an explicit compiler receipt byte SHA and never imports
the compiler, bridge, model or simulator. All eight original numeric arrays must
be regular files in the receipt directory, SHA-bound, shape/dtype/header/payload
checked before read-only mmap. No pickle, symlinks, trailing data or replacement.

Independently reconstruct J.T@J/D, J.T@(goal-nominal)/D and the .01-damped solve.
Compare saved Gram/rhs/delta at atol1e-10/rtol1e-8 (arithmetic consistency only).
Recompute both nonlinear MSEs and nominal-first exact argmin; selected successor
must equal its CUDA replay bitwise. Retain failed checks with exit2. Native-zero
nominal and bounded candidate packets are checked. Independently reconstruct
centered native-u=delta64*std32, exact nominal/selected packet, source float32
dataset projection for nominal/history/ridge. The source statistics are supplied
by the authenticated compiler receipt; their historical training provenance is
not established by this audit. Bridge source/input checks remain separate.
The three saved normalized-input arrays are independently bound and compared
bytewise with these reconstructions, not just with receipt assertions. Replay
compares dtype/shape/bytes, distinguishing signed zero. Finalization errors must
be exactly an empty list, not null or another falsey value.

This does NOT independently regenerate features, verify JVP by finite differences,
validate object semantics, test physical control or open P1/P5. Original neural
work is inherited, not re-executed or counted as zero overall compute. The audit
performs ordinary array reductions and a 10x10 solve on CPU; no neural fallback.

From project root, run the named tests. CLI `prepare --output NEW_JSON` freezes
source only. Then `audit --prepared SOURCE_JSON --receipt CUDA_COMPILER_JSON
--receipt-sha256 RETAINED_SHA --output NEW_AUDIT_JSON`. All outputs exclusive.
Eight roles/shapes match the frozen bridge compiler contract: response_columns
[98304,10] float32; goal_features/nominal_successor/ridge_successor/selected_replay
[1,256,384] float32; gram[10,10]/rhs[10]/ridge_delta[10] float64.
