# P1 轨迹感知目标：主机 CUDA 与真实物理结果

本轮针对一个具体代码/协议假设新增了诊断：旧 Adam 规划只优化
第 10 步终点 latent MSE，而物理环境允许中间步原生成功。新规划器
在相同已知根、相同冻结 checkpoint、相同四起点和 20 次 Adam 更新下，
改用第 5/10 步预测端点的固定 soft-min 目标：

```text
J = -0.02 * logsumexp([-L5/0.02, -L10/0.02])
```

不读取物理 outcome，不训练模型，不改 P1 gate。

## 工程证据

- 主机 CUDA：RTX 5060 Ti，`cuda:0`，24/24 槽完成；
- 规划结果：`le-wm-main/results/p1_trajectory_objective_adam_20261031_RUN01.json`；
- 规划结果 SHA-256：
  `0b0fe954458445ac93187b76b804ee516add90863b9c513c5957842dc9d46a06`；
- 物理 prepare SHA-256：
  `5be2353396196824141e2e97d56cac90886f060033789f6f45fedcbc2f6fd978`；
- 物理 run report SHA-256：
  `de738a6f0866459c4b94ac20fb6e3368f27e055c6fac515945a7310dfc157905`；
- saved consumer SHA-256：
  `8c1a36d75a4a242bc85e294fe4781d3ccc391bc61ac2884b16335dfac3ee5f12`；
- 24/24 物理行通过保存态消费；22 行走满十步，2 行原生成功提前停止；
- 规划器和物理 wrapper 回归测试 3 项通过。

期间 prepare 首次调用暴露一个真实 wrapper 接线错误：规划模块的
`prior` 位于 `planner.source.adam.prior`，而不是 `planner.prior`；
修复后未执行任何物理动作的 prepare 重新通过。另一次 run 仅因输出
目录未满足 runner 的 `_run_` 命名合同而在动作前拒绝，随后用新目录
重启；两者均不是科学负结果。

## 结果

按三种角色汇总（8 条/角色；已知根且含重复轨迹）：

| 角色 | 原生提前成功 | 平均物体位置误差 | 平均物体角度误差 | 平均 native distance |
|---|---:|---:|---:|---:|
| trajectory-selected | 0/8 | 32.254 | 0.139 | 226.452 |
| zero | 2/8 | 26.677 | 0.172 | 222.232 |
| greedy | 0/8 | 32.594 | 0.254 | 246.314 |

该目标使 24/24 个槽的候选动作不同于既有两包目标规划，但没有使
selected 相对 zero 获得真实任务收益。24 行共享两个已知根，不能
估计独立成功率或泛化。

## 归因

这轮排除了一个具体的代码/协议解释：仅把终点目标改为包含第 5 步
端点，不能解释或修复当前 P1 负结果。模型内所有 selected 候选的
最低端点仍为第 10 步，说明该 soft-min 没有实际把中途目标纳入赢家
偏好。真实结果仍显示 latent/规划目标与原生任务成功不一致。

因此下一步不应继续在同一 known-root 上调 soft-min 温度。当前仍需
优先处理真实任务评价与动作/历史/normalizer 来源合同，并在取得合法
独立 root authority 后冻结方法做确认。`P1_admission=false`、
`independent_confirmation=false`、`equal_budget_certified=false` 保持不变。
