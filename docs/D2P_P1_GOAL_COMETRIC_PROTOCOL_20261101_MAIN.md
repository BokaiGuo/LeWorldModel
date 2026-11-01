# Goal-only co-metric diagnostic

Reuse the existing frozen Reacher goal-encoder metric recipe on the single eligible
return-task goal. This is a measurement diagnostic, NOT yet a deployed controller.
No fitting to physical outcomes, no trained critic/adapter and no environment state
input. All neural operations run on host CUDA.

Compute the full 192-row input Jacobian of the original processed goal image.
Retain the original .01 ridge, three seeded directions, steps .001 and .003, and
FD tolerances relative error <=.15, cosine >=.98. Seed is20261101; no retries or
tolerance selection on failure. Exactly one unperturbed encoding plus12 finite-
difference encodings and192 VJP rows. Verify unperturbed embedding bitwise against
the existing target receipt and unchanged checkpoint afterward.

Q=.01*(J J^T+.01 I)^-1. Its square-root transforms latent residuals. This pixel-
tangent co-metric is a proposed planning objective, NOT a certified physical task
metric. Passing derivative checks is not P1 passage, and improved retrospective
score alignment would not establish prospective control improvement.

Before any controller experiment, consistently integrate the same objective into
candidate initialization, suffix optimization, winner refinement, initial selection
and carry selection; bind its artifact and account its construction separately.
Do not silently replace only one score while leaving the other stages Euclidean.
