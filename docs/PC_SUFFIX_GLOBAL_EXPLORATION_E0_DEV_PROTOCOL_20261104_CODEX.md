# Fixed-prefix suffix optimization: global-exploration development diagnostic

Hypothesis: preserving a fixed 25% uniform-action component in each CEM
batch can improve discovery of a usable suffix when the locally sampled
distribution converges prematurely. This is a new exploratory optimizer,
not the frozen v22 method; the hypothesis can fail.

Use exactly four new exact-Wall tasks with start `(-0.5, y, theta)`,
`y ∈ {-0.24, 0.24}`, `theta ∈ {-1.16, 1.16}`, goal `(0.5, -0.22)`.
Reject overlap with explicitly checked previous panels. Run the existing
fixed-height privileged geometric witness constructor once per root; choose
its first eligible successful 15-step witness, or record the root
ineligible. This witness is diagnostic-only and never a planner input.
Freeze its first five actions; evaluate the zero suffix once.

Compare the existing full-action suffix CEM and the new bounded
global-exploration full-action CEM, each with three starts of 120 suffix
samples and the same seed sequence per root. Both use batch size 12, four
elites, initial scale 0.35, minimum scale 0.07 and incumbent preservation.
The new arm devotes floor(25% of each full batch) to uniform suffix
actions and performs elite updates on executed bounded actions. Algorithms
differ in more than uniform samples because the existing CEM updates
unclipped proposal coordinates; this comparison is not a single-factor
causal ablation. Report each score, success `<0.05`, samples, and all
ineligible or failed tasks. Zero-tail costs three prediction tokens shared
by the two arms; suffix optimization costs 1,080 per arm per eligible root.

Independently replay every scored chain with alternate same-environment
geometry, verify prefix, bounds, counts and saved score. Both evaluators
represent the same environment, not independent replication. Stop after
these four roots and three starts; no replacement roots or extra tuning.
Unknown task history and privilege preclude PC2/PC3 performance claims.
