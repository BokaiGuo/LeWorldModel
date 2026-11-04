# Fixed-bank physical suffix diagnostic, development only

Fix the four zero-velocity starts, goal, XML and persistent MuJoCo 3.1.6
action clock from the preceding geometry-feasibility experiment. This
protocol is separate and freezes its own five-action prefixes and ten-action
suffixes. All four starts are retained, including failures.

Prefix bank, in order: zero; the first five previously recorded actions of
the privileged one-step direct controller for the same start; and each of
eight fixed directions at magnitude one, repeated five times. The direct
prefix is a privileged reference, not an RGB planner. Never select a prefix
or modify this bank using a physical suffix result.

For each prefix, report initial and prefix-end Euclidean goal distance and
the whole physical trajectory. A non-improving prefix has prefix-end
distance at least the initial distance, with no post-hoc tolerance or
earlier truncation. This is a position-only diagnostic, not a claim about
other task metrics.

Apply the identical six open-loop ten-step suffixes to every prefix of each
root: zero, and five seeded independent uniform `[-1,1]` action sequences
(NumPy default RNG seed 2026110317). Report absolute end distance and
change relative to that same prefix followed by the zero suffix. A
beneficial response relative to zero suffix is not by itself proof of
improved controllability.

Separately optimize each prefix's suffix against privileged physical terminal
position distance, with common random numbers across prefixes within each
root: two starts, three generations of 16 sequences, four elites,
per-coordinate standard-deviation floor 0.02, action clipping `[-1,1]`.
Initial means are zero and a fixed seeded uniform sequence respectively;
initial standard deviation is 0.5. Each start also evaluates its final mean
and the all-zero suffix, for 100 physical objective evaluations per prefix.
Include all evaluations and resulting best actions. All candidate sequences
replay from the same original state through the entire prefix, using the
same persistent action clock. Ties use first index.

This is finite privileged CEM, not an upper bound on achievable control, not
a deployable method, and not an independent test. Compare the optimized
terminal distance against zero and direct prefixes, and retain the full
bank whether or not a non-improving prefix wins. A candidate that improves
terminal distance only by differing initial position is not a mechanism
finding. The observed results cannot modify historical contracts.
