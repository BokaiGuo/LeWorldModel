# Selected weak-direction precision and nonlinearity audit

The prior spectral run checked finite differences only for the strongest
singular direction, although the model selector chose the weakest direction on
both development roots. This audit holds the saved selected action-parameter
direction, root, goal and frozen checkpoint fixed and tests four perturbation
scales. It uses host CUDA and does not execute new physical actions.

| Root | Arithmetic | Centered FD vs autodiff at 0.001 | Forward linearization error at 0.5 |
|---|---|---:|---:|
| 20263160 | float32 | 0.4968 | 0.9985 |
| 20263161 | float32 | 0.7986 | 0.9502 |
| 20263160 | float64 diagnostic | 0.0000667 | 0.9773 |
| 20263161 | float64 diagnostic | 0.0000125 | 0.7902 |

FD relative error is the norm of the centered finite-difference derivative
minus the autodiff Jacobian-vector product, divided by the latter's norm.
Forward error is the norm of the actual forward displacement minus the linear
displacement, divided by the actual displacement norm. The 0.5 radius is the
original parameter-space candidate amplitude. Repeated predictions at the
identical input were bit-identical in both arithmetic settings.

This establishes a concrete **numerical precision limitation in the earlier
validation**: its strong-mode check did not validate the selected weak modes,
and float32 finite differences at the chosen small scale are unreliable here.
The float64 diagnostic agrees locally with its own autodiff but still shows
large nonlinear error at the original amplitude. It does not prove that the
original float32 autodiff itself is wrong, nor that this issue alone caused
the physical outcome. The nonlinear CEM selector did evaluate the candidate
through the full model, but that does not turn the zero-point weak singular
vector into a reliable finite-amplitude reachability interpretation.

The original action embedder forces `.float()` on its input. The first float64
launch therefore failed with a dtype mismatch before producing outcomes. A
second launch used an in-process dtype-preserving override of the same
permute/conv/MLP sequence, with the original stored weights converted to
float64 and converted back for the post-run hash check. This is a diagnostic
variant, not a deployment result or a modification of the installed package.
The failed output directory is retained.

Per root each completed audit used 192 backward VJP rows, one Jacobian primal,
four baseline/repeat primals, and eight finite-difference primals, excluding
RGB encoding and wall-time accounting. Tests of linear and nonlinear
known functions passed. The next method experiment should explicitly account
for direction reliability and the cost of measuring it, and must retain
nonlinear rollout and physical verification instead of assuming small local
singular values imply a finite-amplitude weak response.

Artifacts:

- `le-wm-main/results/reacher_pc_selected_direction_fd_repeat_20261102_v2/report.json`
- `le-wm-main/results/reacher_pc_selected_direction_fd_double_20261102_v2/report.json`
- `le-wm-main/experiments/spectral_reachability/run_reacher_pc_selected_direction_fd.py`
