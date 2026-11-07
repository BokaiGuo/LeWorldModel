# TwoRoom Zero-reference 与 CEM16：同单次查询预算开发对照

按[协议](PC_RGB_HOLD_CEM16_BASELINE_PROTOCOL_20261107_CODEX.md)，在先前已曝光的 40 根任务上新增 RGB-only 闭环 CEM16；与先前四个 zero-reference 保留臂使用同一模型、25 步任务上限、五步反馈和剩余时域末帧目标。每次规划各评分 16 条链。

| 方法 | 曾成功 | 相对 CEM16 的独胜／独负 | 整面板评分 | 模型包转移 |
| --- | ---: | ---: | ---: | ---: |
| CEM16 | 25/40 | — | 2448 | 7968 |
| 普通 lookahead | 27/40 | 6 / 4 | 2496 | 8160 |
| 中心化向量 | 26/40 | 7 / 6 | 2480 | 8144 |
| zero-reference 后果向量 | 27/40 | 7 / 5 | 2512 | 8208 |
| 行内身份打乱 | 27/40 | 6 / 4 | 2544 | 8256 |

本面板上 effect 比这一固定配置 CEM 多成功两根，但打乱和普通 lookahead 也为 27/40；这不能隔离行动后果信息的增量，也不改变原 effect 对照未通过开发出口的结论。每次评分数相同不代表整任务开销相同；本 CEM16 的种子、分布及迭代规则只是一种固定配置，不是充分调优的强基线。

CEM16 的整面板局部同步规划时间约 4.71 秒；先前 effect 臂约 3.32 秒。两者在不同运行中测量，包含各自实现及批处理差异，不含模型加载，不能当作正式效率对比或完整质量—成本曲线。

对 CEM 的 **153 次决策、2448 条候选**重建随机样本及分布，并重新在 GPU 批量评分；最大分数差 `0.000397`，最低分选择均一致。逐步同环境重放 713 个实际执行步，物理距离最大差为 0，同时核对 RGB 输入收据、剩余时钟与旧尾。定向候选和保留测试 **5 项通过**。复核使用同一模型和物理后端，非独立实现；Gym reset 类型和动作转换警告仍存在。

数据与任务已曝光，来源独立性未知；该比较不是独立确认、实际链机制或跨模型结果。PC0–PC5 不改判。[运行轨迹](../le-wm-main/results/pc_tworoom_hold_cem16_baseline_20261107_codex.json) · [候选池](../le-wm-main/results/pc_tworoom_hold_cem16_baseline_20261107_codex_pool.npz) · [复核](../le-wm-main/results/pc_tworoom_hold_cem16_baseline_20261107_codex_audit.json) · [代码](../le-wm-main/experiments/pc_tworoom_hold_cem16_baseline_20261107_codex.py)。
