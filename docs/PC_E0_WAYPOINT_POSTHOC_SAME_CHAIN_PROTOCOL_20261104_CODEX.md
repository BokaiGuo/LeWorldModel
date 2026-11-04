# E0 waypoint-CEM: post-hoc same-chain intervention

This protocol is written **after** the eight-root waypoint search and its
outcomes were seen. It is a fixed follow-up on all eight final selected chains,
not a preregistered PC2 confirmation, not a new planner experiment, and not a
test of v11 or a spectral method. The stopped planning panel is not reopened;
no candidate generation or thresholds are tuned based on the results here.

Use each sealed final chain after joint refinement and paired abstention. The
same-bank prefix bank consists of the zero prefix and both saved optimized
waypoint prefixes, in that order; zero may coincide with another prefix.
Greedy means the bank prefix with smallest immediate goal distance; lookahead
means the bank prefix with the best result from a separate common-seeded
119-sample, four-knot suffix CEM; random is the existing fixed-seed random
bridge. These controls can equal the selected prefix; ties or duplicates are
reported, never removed.

Reuse the existing v5 intervention implementation and its published fixed
numeric rules and seeds. For selected/zero/greedy/lookahead/random, replay the
same suffixes: zero, four fixed constant-action sequences, and the selected
actual ten-step tail. Report absolute terminal distances and each difference
relative to that prefix's zero tail. For each prefix, independently run three
480-sample, four-knot suffix CEM searches with matched seeds; include the
initial zero tail. Calculate task-distance gradients and endpoint response
matrices around the selected tail with first five tail actions clipped to
[-0.9, 0.9] at finite-difference scales 0.01 and 0.05, recording clipping and
branch stability.

For descriptive comparison only, retain the inherited v5 predicate:
nonprogress, nonzero actual prefix and tail; selected relative-zero gain on
the selected tail and selected mean replanning terminal distance each strictly
better by `1e-10` than each designated control; response matrix and
goal-gradient gaps each above `1e-6` at both scales; all relevant finite
difference batches branch-stable. Base compares zero and greedy; extended adds
lookahead and random. A zero count does not prove absence of mechanism, and a
positive count cannot establish the original gate because the method, roots,
and this intervention were exposed and the waypoint prior is hand-designed.

Save all generated actions, predicted positions and scores with dependencies
and hash bindings. Rebuild all results deterministically in a separate audit;
also verify the final selected chain equals the sealed plan and recompute all
common-tail distances and full-action replay with the independent wall
intersection implementation. Abort on inconsistent source or inputs. Keep
failures and all eight-root denominators. No result-conditioned sample
additions or repeated adjustments.
