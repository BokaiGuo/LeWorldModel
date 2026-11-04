# Feasible spectral prefix backtracking: development protocol

After the soft-penalty development experiment produced no eligible refined
prefixes, test a new method on the same two exposed roots. Keep its four-step
refinements and the entire original bank. For each of the four refinements,
evaluate the interpolation `initial + alpha * (refined - initial)` at fixed
alphas 1, 0.5, 0.25, 0.125 and 0, in that order. Evaluate immediate goal
cost and the same five common one-packet suffixes on the frozen model.
Among points whose immediate cost is at least neutral plus 1e-6, retain the
one with smallest soft-min future cost (temperature 0.01), ties resolved by
the fixed alpha order. If none is eligible, append no candidate for that
seed. The neutral threshold is exactly the existing model selector's
development eligibility, not a claim about real physical non-greediness.

Run the unchanged predicted-successor CEM selector with its prior per-candidate
budget over original and newly retained candidates. Record every evaluated
point, whether it was feasible, exact alpha, bounds and separate costs;
charge all refinement, backtracking and suffix CEM calls. This is not an
equal-total-budget test. No simulator state may enter candidate selection.
These exposed roots cannot be used as independent confirmation.
