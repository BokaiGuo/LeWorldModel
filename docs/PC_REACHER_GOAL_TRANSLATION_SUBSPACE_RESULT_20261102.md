# Goal-only image translation subspace: fixed-pair result

A predeclared goal-image translation subspace was computed on host CUDA
without model training, simulator state, or future images as inputs.
The four shifted goal encodings had full declared rank on both exposed roots.
The frozen saved predicted and real RGB latent vectors for all 26 action
pairs were then scored in that subspace. This is a post-exposure development
test of a possible task proxy, not a newly executed planner.

| Root | Model vs real latent reversals | Real latent vs physical reversals | Model vs physical reversals | Real latent ties |
| --- | ---: | ---: | ---: | ---: |
| 20263160 | 1/13 | 8/13 | 9/13 | 2/13 |
| 20263161 | 0/13 | 0/13 | 0/13 | 0/13 |

Exact ties are excluded from reversal counts; the two first-root ties have
identical actual terminal RGB with physical distance differences around 5e-9.
For comparison, the same fixed-pair L2 real-latent-versus-physical reversal
counts were 6/13 and 0/13, and goal-Q counts were 7/13 and 0/13. These
dependent pairs on two known roots cannot support statistical generalization.
The fixed full-image translations also move the static background, so their
latent directions should not be described as certified task or joint axes.
There is no reason from this result to replace the existing objective with
this particular subspace or claim an improvement in planning.

An independent saved consumer checked the SVD span, orthogonality, singular
values, source binding and all 52 scores. It did not independently rerun
encoder inference. Four targeted tests passed, including image-edge behavior
and deliberate basis/score tampering. No physical trajectories or planning
decisions were generated in this probe; P1 remains unproven.

- Plan SHA-256:
  `3b97a3d64a94526e2a58709c3292a527dd8b3032cfb4e3ea44d988c11fd7c584`
- CUDA result SHA-256:
  `bc60fa547727a250fd9f1ba3ecb188514058eee7c0bef36bb76ebe61f23319b4`
- Saved consumer SHA-256:
  `8b42830b8288c4735157d87a165c316859b090d7320972a95d7b0115f2b7fcbf`
