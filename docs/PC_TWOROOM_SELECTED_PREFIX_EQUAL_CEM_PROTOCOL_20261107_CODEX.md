# Selected-prefix equal-budget tail CEM: development protocol

All 40 exposed TwoRoom roots and the descriptor prefix in the original
sealed common-tail plan are included. Compare the original prefix, a
same-duration zero prefix and its action order reversed. Each prefix stays
fixed for all optimization iterations; physical state is never used for
planning.

For each root and arm optimize the 15-step tail with four iterations of 64
bounded samples, eight elites, mean and standard deviation estimated from
the elites with a 0.05 standard-deviation floor. Start at mean zero and
standard deviation 0.6. The first iteration contains the four shared B tails,
a zero tail, and 59 Gaussian candidates (generated as 64 draws with the first
five overwritten); subsequent iterations contain 64 Gaussian draws. Clamp to
[-1,1]. Each arm uses the same root-specific random-number
stream, though adapted distributions diverge. Select the lowest observed
model endpoint cost across all 256 actual queries (stable first tie).
Score full 25-step chains using only the initial and goal RGB. Record all
candidates, scores and selected chain, batching at most 32.

Seal all model choices and fingerprints before opening physical truth.
Replay one selected chain per arm from identical simulator initial states;
check initial rendered RGB, record full step-by-step distance and success.
Also report the original selected-chain physical success from the previous
sealed intervention; do not replace it with the newly optimized chain.
Compare per-root successes and paired exclusive wins. Query budget is equal
across arms but does not include the cost of originally discovering the
selected prefix. This tests an optimizer- and budget-specific alternative,
not oracle optimality, non-immediate mechanism, discovery attribution, or an
independent closed-loop gate. Preserve all roots and any negative results.
