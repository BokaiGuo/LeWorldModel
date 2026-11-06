# Fixed-bank successor response retention: E0 development protocol

This prospective diagnostic tests whether target-weighted successor
response and response diversity retain more useful prefixes than simple
rules on a common automatically generated bank. It does not assume that
response retention wins, and it is not PC2, PC3, or an end-to-end
equal-cost comparison.

Four tasks: start `[-0.5, y, theta]` with `y` in `{-0.23,0.23}` and
`theta` in `{-1.0,1.0}`, goal `[0.5,-0.21]`. Check exact tuples against
the panels enumerated by the v15 runner; wider provenance remains unknown.
The unchanged v13 compiler (1,920 token cap) produces ten first-layer
prefixes on each task. No witness or privileged trajectory enters the bank.

For each prefix, evaluate the zero 10-step suffix and central finite
differences at `±0.05` for each of its 20 action coordinates, using exact
15-step dynamics. The resulting `2×20` matrix and the zero-tail endpoint
define a local target residual. Compute the unconstrained ridge residual
`min_w ||r - G w||² + 0.1 ||w||²`; this is an unconstrained local
heuristic, not a physical reachability guarantee. Charge all 41 full-chain
evaluations per prefix. Save raw actions, endpoints and scores.

Compare five retained-prefix rules, each selecting four:

1. `greedy`: the four lowest immediate prefix scores.
2. `state`: the v12 rule, two best immediate followed by farthest endpoint
   features.
3. `response`: immediate best, lowest ridge residual, then farthest raw
   successor response matrix from the retained set; stable index ties.
4. `lookahead`: for every prefix evaluate 24 bounded per-step random
   suffixes using its prescribed seed, then retain the four lowest best
   full-chain scores (including zero suffix). These samples are distinct
   from the subsequent CEM samples and charged separately.
5. `random`: four candidates drawn with predeclared root seed.

Every one of the ten prefixes, not just the retained ones, receives the
same three-start, 120-proposal-per-start 10-knot suffix CEM and zero-tail
incumbent. Use identical start seeds across prefixes; no physical optimal
index is fed to any selector. Save every proposal and score. For each rule
report selected indices, best suffix distance, success `<0.05`, and gap
to the diagnostic all-bank best. Record generation, response, and suffix
token charges separately, including all lookahead evaluations. The all-bank oracle is only a retrospective
diagnostic; equal suffix opportunity does not equalize full pipeline
costs. A separate full-method comparison is required if this provides
positive development evidence.

Use four roots only; stop after all four attempts without replacement,
rerun, or post-outcome tuning. Check source/dependency hashes, first-layer
identity, common suffix sample counts, raw actions, scores and response
matrices by replay with alternate same-environment geometry. This checks
implementation consistency, not independent replication, task
feasibility or a PC gate. Preserve negative results.
