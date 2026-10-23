# PointMaze V14：RGB 几何读出到闭环的首次证据

## 结论

V14 不证明 frozen latent world model 已经解决视觉规划。它证明了一个更窄但可复现的结果：在固定、支持内的 16 个 20-step action sequence 上，经过校准的 RGB 几何历史能把已授权的单段 action-ranking 收益带到短视、重观测的 PointMaze 闭环终点距离上。

这个结论有严格边界：严格 success 没有改善；fixed candidate bank 的有限可达性仍是主导限制。因此结果不能被表述为通用 MPC、普适 world model、或 frozen latent planning 成功。

## 冻结协议

- 策略：V14 `rgb_geometry` state estimator 加 true-state-supervised task ranker；部署时输入仅为渲染 RGB 中绿色 marker 的八帧位置/速度读出和固定 candidate bank。
- 对照：相同冻结 ranker 的 action-only 分支。
- 运行时：每轮选择一条固定 20-step candidate，执行、重新渲染 RGB、再选择；每个 root 三轮。
- 目标：`[0.8, 0.8]`，与 V14 context 的 `goal_qpos` 一致。
- 物理环境状态只用于 MuJoCo 推进和事后评价。每一条 deployable trace 都记录 `simulator_state_in_policy_input=false`。
- RGB chart 仅由固定 calibration roots `1800–1839` 拟合；所有闭环评测 root 与它们互斥。

实现位于 [run_pointmaze_v14_rgb_geometry_closed_loop.py](/home/gbk/code/LeWorldModel/le-wm-main/experiments/run_pointmaze_v14_rgb_geometry_closed_loop.py)。GPU 运行前经过 `require_cuda_recurrent` 检查，实际设备为 NVIDIA GeForce RTX 5060 Ti（CUDA 13.0/cuDNN 9.20）。

## 结果

| Panel | 新评测 roots | RGB − action-only 终点距离配对增益 | 95% bootstrap CI | strict success（RGB / action-only） | 判定 |
| --- | ---: | ---: | --- | --- | --- |
| 首次闭环 | 32 (`7160–7191`) | +0.03041 | [-0.04090, +0.10409] | 1/32 / 0/32 | 方向性，未确认 |
| 独立复现 | 64 (`7192–7255`) | +0.06564 | [+0.00142, +0.12816] | 2/64 / 2/64 | 短视距离收益确认 |

独立复现的逐 root 方向为 36 正、4 持平、24 负，中位增益 +0.05645。RGB 分支平均 progress 为 +0.21530，对照为 +0.14966。两份 artifact 都通过 root-disjoint 和 runtime-input 审计。

对应 artifact：

- [首次 32-root report](/home/gbk/code/LeWorldModel/le-wm-main/results/pointmaze_v14_rgb_geometry_closed_loop_20261021_cuda/report.json)
- [独立 64-root report](/home/gbk/code/LeWorldModel/le-wm-main/results/pointmaze_v14_rgb_geometry_closed_loop_replication_20261021_cuda/report.json)

## 为什么 strict success 仍未改善

这是 planner/candidate-bank 的独立限制，而不是将 V14 的 RGB 读出失败重新归因于表示。后验特权 greedy oracle 使用相同 16-action fixed bank，在首次 32 roots 上每轮以真实 simulator terminal distance 选择：

| greedy oracle blocks | 平均终点距离 | strict success |
| ---: | ---: | ---: |
| 3 | 1.75140 | 3/32 |
| 8 | 1.46037 | 1/32 |

即使给 selector 特权 rollout，固定 bank 加短视贪心也很难完成任务，且增加 blocks 进入局部最优。这排除了“只要 RGB ranker 更准即可得到 strict success”的解释。

## 工程修复

首次运行发现真实部署边界错误：从 JSON context 读取的 candidate bank 是 `float64`，而冻结 GRU 是 `float32`，在第一次排序前触发 PyTorch dtype exception。闭环入口现将 history/candidate 显式 cast 为 estimator parameter dtype，并新增回归测试，确保 `float64` JSON 输入可传入 `float32` frozen model。

验证：

```text
tests/test_pointmaze_temporal_ood_v12_decoupled_belief.py
tests/test_pointmaze_v14_rgb_geometry_closed_loop.py
4 passed
```

## 后续科学边界

下一项正当的修复不是调整本次 held-out root 上的 threshold，而是单独设计并冻结一个 goal-conditioned、可更新 candidate bank 的 planning protocol，并以特权 oracle 上限预先验证可达性。该 protocol 必须用新的 roots 评估，并报告 action-ranking、progress、terminal distance 和 strict success；不得用本页任何评测 outcome 选择其 architecture 或超参数。
