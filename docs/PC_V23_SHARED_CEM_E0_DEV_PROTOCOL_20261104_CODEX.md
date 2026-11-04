# v23 shared-optimizer exact-Wall development protocol

Before running, fix eight tasks with starts `(-0.5, ±0.24, ±1.13)` and
goals `(0.5, ±0.16)`. Reject exact overlaps with the enumerated previous
development panels. Broader provenance remains unknown.

Compare the v23 response compiler (`max_modes=2`) with the v23 shared-bank
multistart CEM baseline. Both use the same proposal bank, bounded float32
actions, deterministic exact-dynamics objective, shared CEM implementation,
four starts at most, paired zero-prefix comparison and a 1920 prediction-token
cap. The compiler spends part of its budget on response and recursive search;
the baseline may spend that budget on CEM. Different seed allocations are
part of the methods and do not isolate a spectral causal effect. This is a
development comparison, not an independently tuned strong-baseline claim.

Record every scored action and its model output. Verify the saved traces with
the alternative implementation of the same wall dynamics, per-call and total
cost, common first bank, and selection against all scored full chains. Report
all eight roots per arm, failures included, terminal distances and success
under the existing 0.05 threshold. Stop after these tasks; no replacement,
threshold changes, or repeated tuning on this panel.

The compiler is a new implementation version and the old v22 conclusions
remain unchanged. This test neither proves the PC2 mechanism nor passes PC3.
