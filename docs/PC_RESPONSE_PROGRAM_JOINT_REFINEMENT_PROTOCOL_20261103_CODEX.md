# Response programs with joint refinement: exact-dynamics development

This is a new response-program method contract, not a replacement for any
historic gate or previously stopped method. It does not claim SVD-specific
benefit. The model is the existing exact two-step unicycle, not an RGB world
model, and the task panel is not independently certified.

Before outcomes, define 16 goals at radii `0.65` and `0.95` and angles
`2*pi*(j+0.238)/8`, `j=0..7`. Use the same bounded action coordinates and
two-step horizon for all arms. Compare, at caps 64/128/256 model-equivalent
objective calls, four starts per arm:

* `response`: for nine evenly spaced turn nominals, calculate the 2x2
  velocity-to-position response using centered finite differences, solve the
  `1e-4` ridge system without SVD, clip velocities, and score the resulting
  full chains. Refine the four lowest terminal-cost chains.
* `same_bank_greedy`: construct the identical bank and refine the four
  lowest *prefix position-error* chains, with stable ordering. Prefix scoring
  is charged to this arm.
* `random_bank`: generate nine uniform bounded full chains using seed
  `2026111000+index`, score them and refine the four lowest full-chain costs.
* `flat`: use four uniform bounded full-action starts with seed
  `2026111100+index`, without prior bank scoring.

The same bounded full-action L-BFGS-B and exact objective/gradient refine
all starts. Split the remaining evaluations equally over starts in stable
order; an early-converged allocation is not reassigned. Bank evaluations,
including finite differences and greedy prefix scores, are charged. Persist
the best pre-refinement chain among the four starts as an incumbent; the final selected chain
is the best of that incumbent and all refinement evaluations. Save the
actual selected actions, not the pre-refinement qualification. Count finite
differences, full and prefix scores, optimizer objectives and gradients,
environment/dynamics steps, ridge solves and wall time separately; the common
cap is not a claim of equal FLOPs or total hardware cost.

For each final selected chain, report initial and post-prefix task distance,
the same selected suffix following its own, zero, and the independently
selected same-bank-greedy prefix, plus each relative to its own zero suffix.
Also solve the one-step bounded suffix separately for those three fixed
prefixes. Mechanism observations are exact-model post-selection diagnostics:
require a non-improving selected prefix and more favorable active suffix
response and optimal-suffix error versus both controls. This is not a
real-RGB intervention or an independent PC2 result.

Report all four arms, all 16 tasks and all caps without picking a favorable
budget. This panel may inform development only. Do not declare PC2 or PC3,
freeze a confirmation method, use the outcomes to modify the historical gate,
or equate program value with the value of SVD.
