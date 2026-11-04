# V5 响应保留臂：事后同链机制诊断

按照[协议](PC_V5_CANONICAL_RESPONSE_SAME_BANK_MECHANISM_PROTOCOL_20261104_CODEX.md)，对 v5 响应保留臂 24 根的最终选中链，使用同库 zero、greedy 和 lookahead 前缀进行共同尾与各自同预算后缀规划。计划已知后才制定此协议，属于暴露面板上的开发诊断。

- **1/24** 根的最终 selected 前缀非零且满足非即时距离条件。
- 在非即时、同一选中尾部相对各自零尾的收益、同预算 CEM 收益、两尺度响应差与障碍分支稳定性的交集中，selected 对 zero 和 greedy 为 **0/24**；再加入 lookahead 仍为 **0/24**。
- 每个前缀均以零尾初始化、119 条候选完成 CEM。selected 的自行规划终距相对从九个候选中选出的 same-bank lookahead 为 **3 胜、4 平、17 负**（`1e-10` 容差）。lookahead 有九选一选择偏差，不是单个等成本部署对手。
- 每根还记录四种前缀下的六个共同尾部，包括零尾、四个固定常量尾及该根实际 selected 尾；完整绝对终距、相对零尾变化、响应矩阵与任务方向梯度均保存在原始数据。局部名义尾的最大裁剪为 0.1，不能把该处响应不加条件外推至实际尾。

[同库重规划](../le-wm-main/results/pc_v5_canonical_response_same_bank_20261104_codex_plans.json)及[审计](../le-wm-main/results/pc_v5_canonical_response_same_bank_20261104_codex_audit.json)重算了 29,040 条完整链；[共同尾与局部响应](../le-wm-main/results/pc_v5_canonical_response_same_bank_20261104_codex_local_data.json)和[重建审计](../le-wm-main/results/pc_v5_canonical_response_same_bank_20261104_codex_local_audit.json)覆盖 576 条共同尾链及 4,032 条局部探针链。审计使用相同精确动力学代码，不是独立后端复现。

本次未构建 random bridge 对照，119 样本单起点 CEM 也不保证充分优化；不能将本次 0/24 直接提升为 PC2 正式失败或理论不可行。它说明当前 v5 响应保留臂在该既有开发条件下没有完成所需机制交集，亦不能把旧 v3 正见证嫁接给 v5。下一步需要对同一方法补随机桥接、充分优化与事前固定的新任务干预，并保留独立来源及计算预算边界。
