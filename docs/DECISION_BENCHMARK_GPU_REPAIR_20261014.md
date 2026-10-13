# Decision benchmark GPU repair — 2026-10-14

## Change

`le-wm-main/experiments/decision_benchmark/train_candidate_effect_adapter.py`
now exposes `--device {auto,cpu,cuda}`. The default remains `auto`; requesting
CUDA fails closed when `torch.cuda.is_available()` is false. Training and
inference tensors are moved to the declared device, and the report records the
device per seed.

## Run

The history-conditioned adapter was rerun on the existing root-disjoint
`jepa_history_near_goal_v5_source` bank with five seeds, 300 epochs and
`--device cuda`.

Artifact:

`le-wm-main/results/decision_benchmark_gpu_history_adapter_20261014/adapter_cuda.json`

Observed summary:

| quantity | estimate |
|---|---:|
| selected regret | 0.0668 |
| random regret | 0.1911 |
| paired regret gain | +0.1243 |
| paired success gain | +0.1250 |
| held-out effect RMSE | 0.3358 |
| train effect RMSE | 0.0128 |

All five seed reports record `device: cuda`. The train/test RMSE gap remains
large, so this is an intervention-level result with nontrivial generalization
risk, not evidence of a universal causal latent dynamics model.

## Claim boundary

The result supports a GPU-executed, history-conditioned action-effect adapter
on this PointMaze candidate bank. It does not establish D-JEPA preference
distillation, frozen RGB online planning, PushT pose closure, cross-environment
transfer, or a universal world model. Those claims still require realized RGB
successor verification and independent closed-loop gates.
