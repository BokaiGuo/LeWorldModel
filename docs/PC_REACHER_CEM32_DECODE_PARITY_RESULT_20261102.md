# CEM32 action decode parity: six development chains

The six recorded CEM suffixes were decoded again on CPU and host CUDA with
the existing bounded-action transform and checkpoint action statistics.
Each has 20 scalar action components. The CUDA result is a reconstruction:
the original selector evaluated bounded normalized action tensors but did not
save decoded raw action tensors.

| Root | Prefix role | CPU raw vs executed | CUDA raw unequal components | Max raw absolute difference |
| --- | --- | --- | ---: | ---: |
| 20263160 | selected | bitwise equal | 6 / 20 | 1.1920929e-7 |
| 20263160 | neutral | bitwise equal | 5 / 20 | 8.9406967e-8 |
| 20263160 | same-bank greedy | bitwise equal | 6 / 20 | 5.9604645e-8 |
| 20263161 | selected | bitwise equal | 6 / 20 | 5.9604645e-8 |
| 20263161 | neutral | bitwise equal | 3 / 20 | 2.9802322e-8 |
| 20263161 | same-bank greedy | bitwise equal | 3 / 20 | 2.9802322e-8 |

CPU and CUDA reconstructed normalized values also differ: 3–6 components
per tail, with maximum absolute difference at most 1.1920929e-7. The six
CPU tails are bitwise equal to the saved physical plan; the corresponding
physical journal was already validated against that plan.

This rules out a large CPU-to-CUDA decode discrepancy for these six saved
parameters, not a sensitivity of the environment to small perturbations,
nor a failure elsewhere in history, normalization, task scoring, or model
prediction. Exact equality to the original CUDA run is not established:
its bounded intermediate tensors were not archived, and the original
selection did not decode raw actions. These are exposed development roots,
not independent P1 confirmation.

No model inference or physical action was performed in this diagnostic.
The result file contains all reconstructed tensors; tests recompute
comparisons from the tensors and bind the CPU arrays to the physical plan.

- Plan SHA-256:
  `b88d9b3f878a9b0bf19e806a49ccb9b551a2839d3c398d9cb3b827dfc5d76aa7`
- Result SHA-256:
  `f287c63f31a3566be6ec7515ca4d80e3772c7052fe13c2debe98fa22060e0921`
- Relevant targeted tests: 10 passed.
