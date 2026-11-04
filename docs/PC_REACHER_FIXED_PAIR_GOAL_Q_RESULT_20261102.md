# Fixed-pair goal-Q prediction and real RGB attribution

The 26 fixed L2-versus-direction action pairs were rescored on host CUDA with
the previously authenticated, goal-RGB-derived encoder co-metric. The same
cp5 RGB/action history, saved bounded action packets and 52 real replayed
cp5/cp10/cp15 RGB trajectories were used. Target, predicted and observed
latent coordinates were saved. This is an exploratory analysis on two
previously exposed roots, not an independent test or a planner execution.

| Root | Q model vs real latent reversal | Q real latent vs physical reversal | Q model vs physical reversal | Real latent ties |
| --- | ---: | ---: | ---: | ---: |
| 20263160 | 0/13 | 7/13 | 8/13 | 2/13 |
| 20263161 | 0/13 | 0/13 | 0/13 | 0/13 |

Reversals require strict opposite signs, and exact ties are kept separate.
The two first-root ties have identical final RGB but approximately 5e-9
physical distance differences. Thus the model-versus-physical count includes
comparisons whose actual latent is tied; it is not the sum of the preceding
two columns. The prior L2 analysis found 2/13 and 1/13 model-versus-real
reversals, and 6/13 and 0/13 real-versus-physical reversals, respectively.
Q reduces the measured model-to-latent rank mismatches in these pairs but
does not improve the first-root real-latent-to-physical count. It is therefore
not evidence that replacing the planner objective with Q will improve
control; a previous Q-tail physical comparison also failed to improve
consistently.

An independent saved consumer recomputed the Q quadratic form from stored
vectors with a linear solve, and rechecked replay and prediction bindings.
Maximum Q absolute score difference was 3.61e-15. It does not independently
rerun the neural model, validate Q training-data independence, or certify
total planner computation. Its initial implementation exposed an arithmetic
contract mismatch: the original score subtracts float32 coordinates before
promoting to float64, whereas the first consumer promoted before subtraction.
The consumer was corrected; no experimental score or gate was changed.
Four targeted tests passed, including a Q ABI equivalence test and deliberate
saved-score and action tampering.

P1 and the full Planning Compiler scientific closure remain unproven. Next,
evaluate a deployable task-aligned objective and its complete physical
planner under a frozen development protocol; do not use real future RGB or
simulator state for candidate selection. New independent roots, strong
baselines, physical mechanism evidence, and full compute accounting are
still required.

- Plan SHA-256:
  `7e7de463479f53f0bf564f15350cdfe706266917e3190d73eb1537425c919ce9`
- CUDA result SHA-256:
  `f8d77261c34340af5c9ad4ec3e940351a5b54005bade4a59c8b6ff7d282f435d`
- Saved audit SHA-256:
  `7cc6139381613e8fa4dcdc499540e7413956c0223c1dc545d6bd60fcc6409d68`
