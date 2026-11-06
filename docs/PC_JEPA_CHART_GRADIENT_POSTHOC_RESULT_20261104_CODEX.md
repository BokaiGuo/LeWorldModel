# JEPA saved response probes: chart versus full-score gradient

The original v12 contact run saved two five-step nominal probe batches
at each of three feedback decisions. Their action perturbations, response
matrices and source decision hashes were checked. Central finite
differences at the recorded `±0.05` scale give:

| Decision | Nominal | Gradient cosine | Same-sign coordinates |
| --- | --- | ---: | ---: |
| 0 | 0 | 0.998 | 10/10 |
| 0 | 1 | 0.999 | 10/10 |
| 1 | 0 | 0.510 | 7/10 |
| 1 | 1 | 0.455 | 5/10 |
| 2 | 0 | 0.886 | 6/10 |
| 2 | 1 | 0.266 | 3/10 |

The first context's chart and full-MSE finite-difference directions are
strongly aligned despite differing magnitudes; later recorded contexts
show weaker alignment for several nominals. These are different
predicted-objective gradients at one finite-difference scale, not direct
physical gradients. Ten coordinates were above the prescribed `1e-6`
absolute threshold in each row, but magnitudes differ across contexts.

This is one exposed constructed goal with dependent feedback contexts.
It motivates a prospective score-aware proposal ablation but does not
prove that the chart caused the physical failure or that the proposed
alternative will improve planning. No model or environment was rerun,
and no PC gate is established.

Protocol: `PC_JEPA_CHART_GRADIENT_POSTHOC_PROTOCOL_20261104_CODEX.md`.
Raw vectors and hashes:
`le-wm-main/results/pc_jepa_chart_gradient_posthoc_20261104_codex.json`.
