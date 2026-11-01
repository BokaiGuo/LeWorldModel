# P3 H3 真实上下文后缀物理：混合 RUN 保存消费结果

本轮只消费已有物理产物，没有启动神经模型或物理重跑。由于
`batch00/01_RUN01` 已有历史目录，且新完成产物位于 `RUN02`，
新增独占消费器显式固定以下来源：

- batch 0、1：`RUN02`；
- batch 2–15：`RUN01`，终态保存检查通过；
- batch 16、17：`RUN01`，按 retained-canvas partial 合同检查，失败行保持删失。

18/18 批通过相应的保存态、事件链、cp15 guard 顺序和来源绑定审计。
固定分母为 54 个 pair，其中 2 个 padding 排除，得到 52 个固定
真实 pair；按动作身份去重后为 26 个 unique action pairs，其中
24 个有合法 new/old 终点。batch 16 有 3 行 canvas-guard 删失，
batch 17 有 1 行删失；没有对这些行补终点或改写状态。

主机 CUDA 不是本轮运行内容；本轮只消费此前已经生成的物理结果。
因此这份结果不能证明 P3 调度收益，也不能打开 P1 或独立确认：

- 位置误差：new−old 低 20、高 20、相等 8；
- 角度误差：低 28、高 12、相等 8；
- 推杆位置误差：低 44、高 4；
- native joint-XY 误差：低 44、高 4。

推杆和 joint-XY 的改善不能替代任务成功，也不能把已知根的重复
动作对当作独立任务。当前最稳妥的结论是：代码/协议层的 RUN 混用
和 partial 消费缺口已修复；科学层仍缺等预算调度比较和独立根。

## 产物

- 消费器：
  `le-wm-main/experiments/spectral_reachability/p3_h3_real_context_physical_summary_mixed_20261031.py`
- 汇总：
  `le-wm-main/results/p3_h3_real_context_physical_summary_mixed_20261031_RUN01.json`
- 汇总 SHA-256：
  `e4a75c39eb152643d40f2779245907f59922e7681197c2de0b9ba4cbbdefe477`
- 固定 manifest SHA-256：
  `2912800417ace34400c7f858a7d971d2427b2e094319fefd341d6f9721e1f5ce`

`P1_admission=false`、`P3_admission=false`、`equal_budget_certified=false`、
`independent_confirmation=false` 均由汇总器写入，未因结果修改 gate。
