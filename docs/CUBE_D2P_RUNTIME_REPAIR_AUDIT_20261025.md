# Cube D2P：运行时修复与研究闭环审计（2026-10-25）

## 当前结论与主张边界

当前仍是 `recovery / exploratory`，不是冻结 D2P 的有效性闭环。
本分支会更新 predictor/action encoder/prediction projector，故即使通过预测校准，
也不能替代研究方案要求的“无后训练 Planning Compiler”证据。视觉 encoder/projector
在 v6 的主机 CUDA 权重对比中保持逐项相同。

负向证据应分成三类：可复现实现缺陷、训练/部署或评价合同失配、该 checkpoint/协议
下尚未解决的实证限制。任何一类都不等于 D2P、谱分析或世界模型理论被否定；
反过来也不能预设所有失败必然是代码错误。

所有 learned-model 运行均要求真实主机 CUDA 分配，失败立即停止，不允许 CPU fallback。
MuJoCo stepping 使用其主机物理引擎，EGL 渲染；这与将神经模型计算回退到 CPU 不同。

## 本次发现、修复及证据强度

1. 历史动作必须保持固定。旧修复损失交换整个 action context，与部署只改变最后一包
   `a3` 的问题不同；runtime-aligned trainer 现在实现 `[x1,x2,x3]+[a1,a2,a3] -> x4`。
2. 原始物理候选不应再次经过 `tanh`。评分与执行必须使用相同 raw packet。
3. 视觉校准不能用 qpos L2 代替。当前 observed cost 是实际 successor RGB 经相同冻结
   encoder 后与 goal RGB 的 MSE；qpos 只作为独立诊断。
4. 源数据 action 支持不等于 live reset 的联合 state/action 支持。runtime pool 实际执行
   源支持历史，再记录本运行时 RGB successor；这是训练分布修复，不是理论证明。
5. v4 的训练候选来自 source，但 P1 bank 是 raw uniform。v2 pool schema 可声明 uniform
   `a3`，并保留源支持 `a0..a2`。训练器遗漏允许 v2 JSON receipt（但允许 v2 HDF5），
   造成一次训练前报错；已修复并加入 receipt 回归测试，未重新收集或覆盖已完成池。
6. v5 原始 P1 种子 `202614000..202614007` 落在训练/验证池种子区间
   `202613501..202614524` 内。该 panel 隔离门不成立，必须保留为污染诊断，不能称为
   新根泛化证据。不能由“action history 不同”豁免预先约定的 seed-disjoint 合同。
7. 新增的同根池第一版使用 repeated reset(seed)，实际 RGB context 不一致。正式 v1
   同根池已改为捕获一次完整 integration/render snapshot，逐候选恢复它。这个缺陷
   只发生于新 builder 的 smoke，不能回溯宣称它解释了已使用 snapshot 的旧 P1。
8. v6 的批量验证与 singleton P1 latent 最大差异 `5.8162e-4`。独立 CUDA 分解证明
   RGB 和动作数值差异均为零，singleton 预测也是零误差，偏差源于 batch-size 数值路径。
   修复为 checkpoint 选择时使用 singleton rollout；新 CUDA 对齐回执最大误差为零。
   **尚未证明这个微小偏差造成 top-1 失败**，不能把数值修复直接写成科学原因。
9. v6 的 12 epochs 仅约 156 次优化更新，v4 则为 3,072 次。epoch 数不是匹配预算。
   v6 训练池独立验证 top-1 在最后一轮仍上升；v7 因而在打开新根前固定最多 96 轮、
   validation-only patience=12、同一 seed/LR、singleton 验证规则。不能用 P1 结果挑 epoch。

## 已完成的主机 CUDA 预测校准

| 分支与 panel | root 合同 | 正控制 top-1 | mean/min Spearman | 状态 |
|---|---|---:|---:|---|
| v4 runtime source pool | train `202610501..202611524`；P1 `202612000..202612007` | 7/8 | 0.431618 / -0.108824 | rank 均值及不反转门未过 |
| v5 原始 uniform panel | **与训练池种子重叠** | 1/8 | 0.761029 / 0.285294 | 污染诊断，不准入泛化 |
| v5 真正 disjoint 重测 | 与 v6 同一 sealed panel：`202617000..202617007` | 1/8 | 0.708824 / 0.400000 | 仅 top-1 门未过 |
| v6 same-root ranking | train `202616000..202616255`；P1 `202617000..202617007` | 2/8 | 0.674632 / 0.091176 | 仅 top-1 门未过 |
| v7 singleton validation + validation-only 早停 | 新根 `202619000..202619007` | 7/8 | 0.765074 / 0.267647 | 8-root 预测校准通过，非 efficacy |
| 同一 v7 权重，独立 32-root 确认 | 新根 `202620000..202620031` | 22/32 | 0.709007 / -0.117647 | **确认未通过**：top-1 与逐根不反转门失败 |

保持原门槛：正控制 top-1 至少 75%、平均 Spearman 至少 0.60、没有 root correlation
反转；每个候选从相同 root 物理执行，模型参数不变，目标度量一致。不能为过门改变阈值。
这些是每 panel 8 根的诊断，不是统计充分的 efficacy，也不是非贪心重配置结果。
v5 重测和 v6 使用相同根/候选，是可配对诊断；v4 与后续 panel 根不同，不能由其数值差
直接断言因果改善。v5→v6 的 1/8→2/8 也不能称为显著收益。
v7 的首个 8 根与独立确认也不能合并后重新计算门槛来宣称通过。32-root 确认是同一
权重、同一门槛、同一 16-candidate bank 法则的预先固定确认，未根据其结果再挑 epoch。

真实 top-1 失败未集中于零动作：v6 六个失败根选择的是 uniform 候选，非 bank[0]。
因此“只补零动作候选”不能作为当前充分解释。

## 可追溯的本地产物

以下路径均相对 `le-wm-main/`，所有原始文件保持不可覆盖。

- `results/cube_d2p_repair_v4_runtime_pool_fresh_p1_20261025_cuda/p1_visual_rank_audit/report.json`
- `results/cube_d2p_repair_v5_runtime_uniform_pool_fresh_p1_20261025_cuda/p1_visual_rank_audit/report.json`（seed-overlap，仅诊断）
- `results/cube_d2p_repair_v5_matched_disjoint_p1_20261025_cuda/`（真正 disjoint 重测）
- `results/cube_d2p_repair_v6_same_root_fresh_p1_20261025_cuda/`
- `results/cube_d2p_repair_v6_runtime_alignment_components_20261025_cuda.json`（保留的失败对齐回执）
- `results/cube_d2p_repair_v6_singleton_alignment_fixed_20261025_cuda.json`（修复后 max error=0）
- `results/cube_d2p_same_root_ranking_pool_v1_256x9_20261025_cuda.h5.receipt.json`
- `results/cube_d2p_repair_v7_same_root_singleton_p1_20261025_cuda/p1_visual_rank_audit/report.json`
- `results/cube_d2p_repair_v7_same_root_singleton_p1_20261025_cuda/evidence_binding_audit.json`
- `results/cube_d2p_repair_v7_confirmation32_p1_20261025_cuda/p1_visual_rank_audit/report.json`
- `results/cube_d2p_repair_v7_full_score_alignment_20261025_cuda.json`

`cube_d2p_repair_root_admission.py` 在启动 P1 前读取实际 HDF5 root_seed 并沿 base checkpoint
祖先链核验 live pool/hash/receipt，训练与验证种子全部排除。不能只相信 manifest 中的
`overlap=false` 或手算的种子区间。

v7 入口：`experiments/spectral_reachability/run_cube_d2p_same_root_repair_v7_cuda.sh`。
实际训练到第 42 轮早停，按 validation 选择第 30 轮；50 个训练池 validation roots 的
top-1 为 82%，pair accuracy 为 93.5%。这些不是独立 P1 结果。
独立主机 CUDA 对齐审计给出 terminal max error=0；补充完整评分路径审计覆盖全部 50 个
validation roots，目标编码 max error `4.7684e-6`，cost max error `1.0133e-6`，argmin
分歧为 0。因此不能把 v7 新根剩余失败归因于已经修复的批量/单例数值路径。

32 根确认的唯一负相关根为 seed `202620011`，其 selected-minus-optimal observed latent
MSE 约 `1.90616`，不是只在视觉近似平局上错失 top-1。其余 9 个 top-1 失败根也完整保留，
不准按 margin、置信度或观察到的损失事后剔除。这仍是当前 repair 的未解决泛化问题，
不能推演为理论反例。

新增 `cube_d2p_p1_evidence_binding.py` 核验 panel/model-input/model-side-score/weights/
repair-manifest 的真实 SHA，并逐 root 检查 tensor hashes、positive/zero packet、argmin
与实际 selected packet。独立确认的物理和视觉审计均已接入；首个 v7 8-root 结果使用旧
审计程序生成，事后另写不可覆盖的 binding 回执，未改写原始结果。
该完整性修复阻止后续混用旧文件，但目前没有证据表明已有 v7 排序失败来自文件漂移。

运行器现在可显式传入 root 数：

```bash
bash experiments/spectral_reachability/run_cube_d2p_runtime_repair_p1_chain_cuda.sh \
  /absolute/checkpoint /absolute/new-run-root 202620000 32
```

任何一步失败即停止，退出非零不等于基础设施崩溃，必须检查具体 gate。本次 32-root
systemd unit 退出 1 是校准门拒绝，非 CUDA、物理重放或完整性检查失败。

## 离原方案完整闭环还缺什么

| 原方案要求 | 当前证据能否证明 | 必须补齐的决定性产物 |
|---|---|---|
| 冻结模型的 progress compiler 与匹配预算强基线 | 部分环境/协议有证据，Cube 当前只是预测校准修复 | 真实物理 progress outcome、GN 等价与强基线、资源账本 |
| 非贪心 reconfiguration 改变后续可控性（kill experiment） | **Cube 尚未证明；ranking 校准不能替代** | 先封存 controllability-gap root family，再做 calibration/outcome，比较同 bank greedy、无组合 SRA 与强优化器 |
| 可执行中间目标与组合的独立价值 | 尚未由此修复分支证明 | action-witness 与 arbitrary/learned-subgoal 对照；真实执行偏差和闭环成功 |
| 动态 horizon/计算调度 | 此分支未验证 | 等调用/反传/延迟预算的闭环比较，不用 latent rank 代替成功 |
| One frozen model, many tasks / 独立模型复制 | 不能由修复后的 Cube 替代 | 同一冻结 checkpoint 多任务及跨环境/架构的独立确认 |
| 论文与复现闭环 | 未完成 | 对应主张的独立确认、根级不确定性、代码与协议绑定、图表与结论审计 |

因此上游预测校准即使修好，也不能自动把 P2/P3/P5 打开。本轮修复归入诊断附件与
artifact，不应替代主论文的 training-free D2P 主张。完整研究目标保持未完成。
