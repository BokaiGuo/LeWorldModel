# CEM32 six-chain action decoding parity diagnostic

Use exactly the six previously selected development action chains and their
saved CEM suffix parameters. On host CUDA, recompute the bounded normalized
actions and raw environment actions using the existing `_to_raw_actions`
implementation. Recompute the same operation on CPU, and compare each to the
raw tail actually recorded in the sealed physical plan. Record all values,
bitwise differences and maximum absolute differences. This performs no model
inference, optimization, simulator actions or training.

The model selector evaluates normalized bounded actions, not raw environment
actions; the CUDA raw values here are a reconstruction, not a tensor saved
by the original selector. Therefore equality proves this reconstruction's
CPU/CUDA parity on the saved parameters, not original execution identity or
correctness of the full model-to-environment interface. Roots are exposed,
and neither a nonzero nor a zero difference constitutes P1 admission.
