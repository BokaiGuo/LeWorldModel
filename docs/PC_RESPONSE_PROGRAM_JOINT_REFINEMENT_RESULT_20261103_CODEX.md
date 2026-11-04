# Response programs with joint refinement: development result

The predeclared exact-unicycle experiment was run on 16 geometry-defined
goals and all three evaluation caps. This is a development panel, not an
independent confirmation or an RGB experiment. The source, protocol,
dynamics and gradient hashes and every selected action are in the saved JSON.

| Evaluation cap | Response | Same-bank greedy | Random bank | Flat |
| --- | ---: | ---: | ---: | ---: |
| 64 | 2.83e-8 | 8.00e-8 | 1.14e-7 | 1.16e-7 |
| 128 | 2.83e-8 | 4.51e-8 | 1.13e-7 | 1.08e-7 |
| 256 | 2.83e-8 | 4.51e-8 | 1.13e-7 | 1.08e-7 |

Entries are mean final distance; all four methods are near numerical zero.
The small differences do not establish a practically meaningful advantage.
The limited mechanism witness count was **0/16 at each cap**. The same
response-selected suffix and independently optimized one-step suffixes were
evaluated on the actual refined prefixes; this is a privileged exact-model
diagnostic and does not establish the physical RGB mechanism.

Mean charged objective-equivalent calls for response / same-bank greedy /
random / flat were respectively 60.125 / 64 / 44.375 / 37.75 at cap 64,
and 61 / 73.625 / 45.8125 / 38 at both higher caps. Early convergence
left allocations unused, as declared before execution. Finite-difference
calls and gradient calls are separately recorded. The cap is not an
equal-FLOPs or total-hardware-cost guarantee; the post-selection mechanism
diagnostic is outside the planning cap. Same-bank greedy chooses starting
chains by prefix distance and then jointly refines the entire chain; its
final prefix is not necessarily a member of the original candidate bank.

The simple task is saturated by all methods. It gives no evidence that the
response program improves search over full-action optimization, and no
evidence of non-immediate-progress mechanism in the selected chain. It
does not test SVD-specific value. PC2, PC3, historical P1 and the RGB
closed-loop and independent-confirmation gates remain unpassed.

Artifacts:

* `le-wm-main/results/pc_response_program_joint_refinement_20261103_codex.json`
* `docs/PC_RESPONSE_PROGRAM_JOINT_REFINEMENT_PROTOCOL_20261103_CODEX.md`
* `le-wm-main/experiments/spectral_reachability/pc_response_program_joint_refinement_20261103_codex.py`
