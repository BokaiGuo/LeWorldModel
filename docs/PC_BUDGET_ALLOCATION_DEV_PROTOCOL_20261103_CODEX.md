# Budget allocation development protocol

This is a new development protocol; prior artifacts remain unchanged. It is not
an independent confirmation and cannot open any scientific gate.

## Full-method comparison

Use the four existing development roots, budgets 900 and 1800 prediction-token
samples, and a total horizon of 15 actions. Compare spectral, matched random,
nominal-only, and full-horizon CEM with the existing candidate generator and
RGB/action history interface. Every method replans after 10 executed actions
using its own real RGB and the same 96-sample, five-action optimizer. At the
root, deduct candidate generation from the token cap, then divide the remaining
complete-trajectory samples equally in stable candidate order; the first
remainder candidates receive one additional sample. A final CEM batch can be
smaller than 12. Never allocate based on outcomes. Record generation and search
tokens, encoding, SVD and wall time separately; token equivalence is not FLOP
or wall-time equivalence. Shared goal encoding and model loading are reported
separately. All plans must be saved before their corresponding physical
evaluation, with model, source, protocol, action and RGB bindings.

## Candidate-quality diagnostic

Separately evaluate the previously sealed 900-budget spectral and random banks,
including their nominal candidates, on each of the same four development roots.
Deduplicate exact action arrays together with their prefix lengths, in source
order spectral then random. Every fixed prefix receives 48 complete-trajectory
samples, the same optimizer and root-specific common random seed. Save all
model scores and plans before physical evaluation. This does not include the
cost of generating the fixed banks and is not a fair full-method comparison.

## Mechanism and stopping

The development sample is the eight spectral-method decisions from four roots
and two budgets. The actual selected prefix, not a post-hoc substitute, is
eligible only if physical distance to the fixed goal does not decrease at its
preselected 5- or 10-action boundary. Report this explicitly as a position-only
criterion, not a general task-progress criterion. For each root and budget,
compare that prefix, zero and same-bank immediate-greedy prefixes of equal
length under a predeclared common suffix family: zero and four five-step
constant actions (+x, -x, +y, -y), each magnitude 0.25. Report absolute final
distance and distance change relative to that prefix's zero suffix. Independently
replan the remaining execution horizon from each prefix's real RGB with the
same optimizer and budget and report final distances and costs.

If no actual selected chain among the eight meets the non-improvement criterion,
shows a more favorable common-suffix active response than both zero and greedy,
and beats both zero and greedy under separately replanned suffixes, stop this
candidate construction and reassess task/mechanism fit. These are development
screening conditions, not a claim of statistical confirmation. A positive result
does not replace independent roots, complete provenance, fair strong baselines,
or the existing admission criteria.
