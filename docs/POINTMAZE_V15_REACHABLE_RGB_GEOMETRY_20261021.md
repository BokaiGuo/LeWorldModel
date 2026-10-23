# PointMaze V15：在可达 planner 下检验 V14 RGB 状态读出

## 问题

V14 的固定 16 条 20-step bank 在闭环 strict success 上没有收益。该结果不能说明 PointMaze 不可规划：它同时改变了候选集、horizon 和重规划语义。V15 因此固定 V14 的 RGB-geometry estimator，只替换为一个已通过可达性检查的 planner：RGB maze waypoint、每轮 32 个 goal-relative action、5-step pulse 和重新观测，最多 24 blocks。

## 部署隔离

- `rgb_greedy`：只用 RGB marker 位置和 RGB-inferred waypoint 选朝向 waypoint 的动作。
- `v14_rgb_geometry_dynamics`：用 frozen V14 estimator 从八帧 RGB 几何历史读出状态，再由独立 calibration roots 拟合的视觉局部动力学预估 5-step endpoint。
- `privileged_one_step_oracle`：唯一允许以 MuJoCo state 对候选作一步反事实排序的诊断 arm。

前两个 arm 的 trace 都记录 `simulator_state_in_policy_input=false`。视觉 calibration roots 是 `1800–1839`，动力学 calibration roots 是 `1840–1879`；正式 evaluation roots 是未用的 `7258–7289`，三者不重叠。

## 正式 32-root 结果

| Arm | Strict success | Mean terminal distance | Mean progress |
| --- | ---: | ---: | ---: |
| RGB geometry greedy | 31/32 | 0.15087 | 2.10770 |
| V14 RGB state + calibrated dynamics | 29/32 | 0.19891 | 2.05966 |
| Privileged one-step oracle | 32/32 | 0.13227 | 2.12630 |

V14 dynamics 相对 RGB greedy 的 paired terminal-distance gain 为 `−0.04803`，95% bootstrap CI `[-0.10008, -0.00537]`。因此它在这个 reachable planner 下有稳定负迁移，不能部署。

Artifact：[V15 formal report](/home/gbk/code/LeWorldModel/le-wm-main/results/pointmaze_v15_reachable_rgb_geometry_formal32_20261021_cuda/report.json)。实现：[V15 runner](/home/gbk/code/LeWorldModel/le-wm-main/experiments/run_pointmaze_v15_reachable_rgb_geometry.py)。

## 机制诊断

在相同 32 roots 的 source-history state readout 上：

| State source | Position RMSE | Velocity RMSE | Full-state RMSE |
| --- | ---: | ---: | ---: |
| 直接 RGB geometry chart | 0.02452 | 1.46268 | 1.03441 |
| Frozen V14 estimator | 0.05030 | 0.56993 | 0.40457 |

V14 显著改善整体/速度 readout，却加倍了位置误差。V15 的 maze waypoint objective 对位置最敏感；同时 RGB greedy 已达到 `31/32`，而特权 oracle 只剩一个 strict-success headroom。这解释了为什么更好的 full-state estimate 没有带来控制收益。

## 结论与正确动作

这个负结果拒绝了“只要 state RMSE 下降，视觉 state adapter 就应改善规划”的假设，也拒绝把 V14 的 offline ranking 增益泛化为 dynamic MPC 增益。当前 PointMaze 的正确部署路由是直接 RGB geometry greedy；V14 只保留为固定 20-step candidate ranking 的有界证据。

若要继续研究 dynamics repair，下一 benchmark 必须预先满足两项条件：

1. 特权 oracle 相对 RGB greedy 有足够 terminal/success headroom；
2. 训练与校准中包含 position-aware、support-matched counterfactual candidate transitions。

否则在近饱和的 navigation panel 上训练更复杂的 state/dynamics readout，只会产生可预期的负迁移。
