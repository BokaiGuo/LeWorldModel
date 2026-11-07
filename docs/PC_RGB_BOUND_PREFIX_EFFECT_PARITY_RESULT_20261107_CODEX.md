# RGB 固定前缀 hold-relative 描述：接口与物理核验

已有通用描述函数实现了 `hold 代价 − 探针代价`；本次补齐它所需的批量固定前缀评分适配层，未替换既有规划器或修改历史实验。适配层在查询前检查动作边界、前缀与尾部形状、模型 packet 时钟和完整剩余时域，并记录链评分次数及模型 packet 数。

按[数值协议](PC_RGB_BOUND_PREFIX_EFFECT_PARITY_PROTOCOL_20261107_CODEX.md)，在冻结的 RGB TwoRoom 模型上重算既有 4 根、5 个前缀、5 个尾部的 **100 条**完整链。代价矩阵与 hold-relative 后果向量的最大差均为 **0**；模型与 normalizer 指纹一致。新适配层及既有描述／动作见证测试合计 **17 项通过**。

按[物理协议](PC_TWOROOM_PREFIX_HOLD_PHYSICAL_PROTOCOL_20261107_CODEX.md)，在相同本地环境中重放 20 个固定前缀，然后各执行 5 步零动作：**20/20 个已到达的前缀后状态**零动作最大位置漂移为 0，重放距离与已封存轨迹一致。已知 seed 2717103 的 reset 边界例外仍保留；这个结果只认证上述已到达状态，不能说明其 reset 初态或其他状态的零动作均为保持动作。物理位置仅供评价，不输入 RGB 规划器。

这完成的是接口数值对齐和有限状态的 hold 检查，不是新的未查询预测或规划收益实验；此前物理后果与预测存在不一致，不能因此宣称 PC1 通过。新适配层尚未接入最终候选保留或闭环；PC0–PC5 不改判。

代码与记录：[描述适配层](../le-wm-main/experiments/pc_rgb_bound_prefix_effect_20261107_codex.py)、[GPU 对照记录](../le-wm-main/results/pc_rgb_bound_prefix_effect_parity_20261107_codex.json)、[物理重放记录](../le-wm-main/results/pc_tworoom_prefix_hold_physical_20261107_codex.json)。
