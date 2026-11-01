# 冻结表示与 rollout 时间对齐：主机 CUDA 独立核验

## 结论与适用范围

本轮排除了所测样本上的 checkpoint 参数名映射、ViT 前向、projector、pred_proj
和当前 P1 recurrence 接线失配；没有解决任务度量与真实物体控制之间的失配，
也没有开放 P1 或任何依赖阶段。不能从这项有限样本审计推出“所有代码均正确”。

报告：`le-wm-main/results/pusht_representation_runtime_audit4_v1_20261027_cuda/report.json`。
状态 `audit_completed`；7 项完整性检查、5 项数值检查全部通过。

## 已执行的数值证据

独立参考直接读取原始 checkpoint 键，不调用兼容层重命名后的 ViT 前向。
参考采用原始 tiny ViT 的 12 层、192 维、3 heads、14 patch、LayerNorm eps=1e-12，
以及 frozen BN eps=1e-5 的两层 projector/pred_proj。

| 核验 | 实际范围 | 结果 |
| --- | --- | --- |
| 原始权重参考与当前 encoder CLS | 8 张真实物理诊断图 + 4 张实际目标图 | 最大绝对差 1.6689300537109375e-6 |
| 同一参考与当前 projected embedding | 同上 12 张图 | 最大绝对差 2.1457672119140625e-6 |
| pred_proj 独立函数复算 | 4 个真实上下文的 predictor features | 全部在预定容差内 |
| 当前 P1 与正确对齐的官方 cached rollout | 4 根 × 16 个原程序 | 最大绝对差 0.0 |
| 当前 P1 与已封存预测 | 同一 64 个程序 | 全部在预定容差内 |
| 官方 criterion 与当前 MSE 单位 | 同一 64 个程序 | SSE = 192 × MSE |

实际记录 448 个 predictor samples；参数、buffers、frozen eval 与输入 hash 均保持不变。
这些是诊断成本，不是与强基线匹配的计算预算。官方 rollout 比较使用同一已保存
latent context；dummy pixels 仅供 shape，不代表重新编码链的端到端等价证明。
编码器链由上述 12 张真实图像的独立检查单独覆盖。

## 两种历史上下文约定不能混用

令真实动作 A0,A1,A2 把图像 F0 推进到 F1,F2,F3：

| 上下文约定 | 图像 | 官方 rollout 的 outgoing action 输入 | 两个新动作 B,C 的解释 |
| --- | --- | --- | --- |
| 旧 P0 causal/bootstrap | F0,F1,F2 | A0,A1,A2,B,C | 先预测已真实执行的 A2，再预测 B,C |
| 当前 root-anchored P1 | F1,F2,F3 | A1,A2,B,C | 直接从当前 F3 预测 B,C |

官方 rollout 从 H 帧、T 个 outgoing tokens 生成 T−H+1 个未来帧。
两种路径含历史时均分别返回 6 帧和 5 帧；前者预测 3 帧，后者预测 2 帧。
把 **root-anchored** 图像与全部 3 个 incoming history tokens 拼接会多预测一步，
本轮错误约定与当前 P1 的 64 个终态均不相等。

但实际源码调用核对发现：`preflight_pusht_d2p_p0.main` 调用
`_collect_causal_rgb_context`（`recorder.frames[:-1]`），再调用 `_jacobian_and_fd`
和 `_terminal_prediction`；它使用表中第一行，并非把 root-anchored 图像误送该函数。
旧 P0 的 bootstrap 预测误差可能影响诊断与当前根的对应关系，但不能仅凭调用次数
把历史 P0 的所有结果判为索引 bug。当前 P1 明确使用 `_collect_root_anchored_rgb_context`，
其直接 recurrence 已与表中第二行逐程序精确相等。

本轮未修改这些已封存历史源码或重写旧报告。

## 仍然未解决的主问题

真实物体静止时，agent 运动仍可产生很大的 Q-latent 能力响应；此前完整 sprite 修补
会改变分数，但仍未恢复物理任务排序。该任务度量/模型表示接口问题仍是主线程主攻点。
排除本轮实现假设不是“理论被否定”，更不是 P1 成立。

后续 P2/P3 可并行完善真实执行与成本对照，但其正式科学准入仍需完整 P1 证据；
P4/P5 的独立前置条件须逐项验证，不能从工程测试通过推导跨任务泛化。

## 复现

在 `le-wm-main` 下主机 CUDA 执行，输出目录必须全新：

```bash
.venv/bin/python -m experiments.spectral_reachability.audit_pusht_representation_runtime_cuda \
  --compilation results/pusht_d2p_metric_program_preflight4_v1_20261026_cuda/report.json \
  --physical-fd results/pusht_observed_suffix_physical_fd4_v1_20261026_cuda/report.json \
  --goal-metric results/pusht_goal_object_metric4_v1_20261026_cuda/report.json \
  --policy /home/gbk/.stable_worldmodel/checkpoints/pusht/lewm \
  --output results/pusht_representation_runtime_audit4_v1_20261027_cuda
```

已完成目录不可覆盖。来源 seal 分两组记录 50+18 个文件，执行末尾均 verified；
seal 证明源文件/归档/参数字节未变化，不替代执行或机制证据。

补充逻辑回归：本轮主线程实际运行 10 个相关测试文件 206 项通过，再运行
adaptive budget、P0 resource admission、composition replay 三个文件 80 项通过，
合计 286 项。它们是逻辑/接口测试，不是 CPU 神经实验，也不是 CUDA 效能结果。
