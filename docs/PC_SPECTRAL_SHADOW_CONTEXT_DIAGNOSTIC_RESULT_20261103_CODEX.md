# Frozen-model spectral CEM on saved shadow RGB contexts

The fixed-input diagnostic ran on CUDA using the two decision-0 RGB pairs and
noise tensors saved by the earlier exposed-root shadow development run.
Snapshot hashes, normalizers and the frozen parameter hash were checked.
The no-injection CEM reproduced the saved chosen plan and final scores within
the prescribed tolerances. The model parameter hash was unchanged.

| Development context | Baseline selected model cost | Spectral selected model cost | Raw Jacobian FD relative error |
| --- | ---: | ---: | ---: |
| r8343_a0_p2 | 0.772118 | 0.721373 | 0.000793 |
| r8345_a0_p2 | 1.665219 | 1.639002 | 0.000534 |

Costs are the existing RGB-derived projected model objective; lower is better.
These are two already-exposed contexts, not independent roots. Both arms used
50 CEM model samples in four batches per context. The spectral arm additionally
used one differentiable response forward, two reverse-vector calls and one
SVD. Each Jacobian check used 20 additional finite-difference forwards; these
are diagnostic, not planner costs. The result is therefore not an equal-total-
compute comparison. Timings in the receipt are single observations and include
warmup effects.

This test establishes that the spectral component can differentiate through
the frozen checkpoint and that its local action/normalization chain agrees
with finite differences on these inputs. It does **not** establish correct
physical ranking, real closed-loop improvement, non-greedy mechanism, spectral
advantage over a matched-compute random initializer, or P1 admission. Neither
spectral plan was executed here.

The machine-readable receipt and runner are
`le-wm-main/results/pc_spectral_shadow_context_diagnostic_20261103_CODEX.json`
and `le-wm-main/experiments/pc_spectral_shadow_context_diagnostic_20261103_codex.py`.
The spectral component's three unit tests passed in the project environment.
