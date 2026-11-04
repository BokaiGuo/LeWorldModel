# Common-tail tangent: fixed host-CUDA model-only development run

Bind CPU preflight `_02`, its source, checkpoint, action normalizer, proposal
code and the two listed exposed roots. Reconstruct each root and goal through
the existing executed-history adapter; reject identity mismatch. Run the
unchanged common-tail tangent compiler exactly once per root, with no
parameter adjustment or alternative selection after failure. Preserve
failure separately per root.

For each successful compiler output, reevaluate all seven prefixes with all
five fixed tails and save their predicted coordinates. This adds seven prefix
and 70 suffix-token prediction samples beyond the compiler's nominal
9 prefix and 90 suffix-token samples per root. Compute the non-gating Q
response census from these saved values; bind bounded normalized and physical
raw actions. Record compiler and diagnostic time, source hashes and model
parameter hashes. The diagnostic does not change the selector. Goal encoder,
history encodes and previously constructed Q VJPs are separately identified
as outside nominal prediction counts; this is not an equal-total-budget
comparison.

No physical outcomes are input to candidate generation. The two known roots,
inherited goal metric and fixed configuration are development-only. No
physical rollouts, native success or independent P1 admission follow from
this model-only run.
