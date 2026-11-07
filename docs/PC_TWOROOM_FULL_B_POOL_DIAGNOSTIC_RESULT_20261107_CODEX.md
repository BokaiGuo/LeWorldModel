# Fixed B bank: exhaustive evaluator-only development diagnostic

## Protocol and verification

The previously exposed roots 10–49 and their existing 64 B chains per root
were fixed in advance. All 2,560 model scores were computed and sealed before
physical replay. The evaluator replayed every chain for all 25 steps from the
recorded reset and goal, using the earlier evaluator's success predicate.
Physical truth was not an input to planning or scoring. The protocol and
source, input, score, and data hashes are bound by the seal; individual
distance, termination, and truncation trajectories are in the physical NPZ.

Previously queried B scores differed by at most 0.00015258789; the minimum
among the previously queried union remained the same on all 40 roots. This
does not establish complete pairwise ranking or closed-loop equivalence.
Three repeated replays (38/5, 39/17, 45/0) reproduced distance and termination
exactly in the same evaluator, not an independent physics implementation.
Three unit tests passed. The environment emitted Gym observation type/casting
warnings; these should not be interpreted as independent interface
certification.

## Results

| Metric | Roots, out of 40 |
| --- | ---: |
| At least one B chain ever successful | 39 |
| At least one successful B in previously queried union | 37 |
| Full B model-best ever successful | 30 |
| At least one B chain successful at final step | 38 |
| At least one success after an unsuccessful first 10 steps | 37 |
| Successful B model-scored below best source incumbent | 1 |

The union is the union of all four arms' queried B candidates, not the
budget of any single arm. Full B selection is an additional 64-score
evaluator diagnostic, not the original deployed algorithm.

Both roots on which all four original arms failed contain successful B chains:

* Root 38: 3/64 ever succeed; none was in the queried union. The model-best B
  chain succeeds and scores 11.9295 below the best source incumbent. Its
  success occurs after an unsuccessful first 10 steps. This is a concrete
  coverage/retention loss in this fixed bank.
* Root 39: 4/64 ever succeed; one was queried by the shuffled arm. The
  model-best B chain fails, and the best-scored successful B has rank 3 within
  B and scores 191.3954 above the best source incumbent. This illustrates a
  model-score/selection mismatch, not a diagnosis of dynamics versus the
  latent task score.

## Interpretation

The prior result that no B candidate was selected cannot be explained as
absence of physical successes in this bank. However, successes in 39 roots
do not by themselves improve the earlier 38/40 selected-chain success; roots
and methods were exposed during development. The first-ten-step filter only
excludes previously successful prefixes; it is not a mechanism intervention.
Further work must use legal model-only access and charge any additional
candidate evaluations. These results establish neither descriptor contribution
nor PC0–PC5 passage, and do not support an independent generalization claim.
