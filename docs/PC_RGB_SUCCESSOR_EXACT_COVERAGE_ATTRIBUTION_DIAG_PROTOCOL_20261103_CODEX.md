# Exact successor coverage and retention diagnostic

On all 24 sealed development tasks, replay the first-bank five-step prefixes,
the four composed ten-step prefixes, every recorded complete 15-step model
sample, and each final selected chain. Check bounded normalized actions,
position, score, and source/receipt hashes. Count candidates and roots that
reach the far side of the wall (`x > 0`), touch a wall, or achieve the
predeclared terminal distance `<0.05`. Identify, for each root, the lowest
recorded terminal score and whether its exact action chain survived final
selection. Report the actual final-chain selection relation separately from
the best sampled chain because the final zero-prefix counterfactual can
replace the pre-comparison selection. Treat conversion to float32 as part of
the exact predictor; do not infer equality of distinct requested actions from
equal rounded model inputs.

This is a post-outcome engineering diagnostic only. No new plans, tuning,
candidate injection or root selection are authorized. Results do not prove a
global search upper bound or independent planning superiority. A retained
sample with a higher score must be reported, not silently corrected.
