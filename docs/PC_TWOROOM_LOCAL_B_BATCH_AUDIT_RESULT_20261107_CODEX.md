# 局部尾部对照：统一批次重评分结果

在同一冻结 RGB LeWM 和原观测下，逐根将 2 条 incumbent、64 条 A、16 条均匀 B 和 16 条局部 B 共 **3,920 条完整链**统一按 32 条批次重评分。与各来源原评分的最大绝对差为 **0.00079346**；两臂 80 次最终选择的动作哈希均与先前封存选择一致。各根各臂重算后前两名分差的最小值为 **0.08170**。

这排除了本次检查范围内“不同评分批次造成最终选链改变”这一解释，支持此前 38/40 对 38/40 结果在当前模型、硬件和候选集合上的数值稳定性；不等于所有候选排序逐位一致，也不能排除模型评分本身与物理成功不一致。保留旧轨迹和结果，不重新定义成功标准。

动作索引、跨臂候选隔离和稳定平局相关测试合计 **4 项通过**。权重指纹前后一致；完整 98 项重算及原始评分逐根记录在机器结果中。该审计未新增物理重放，且任务已经曝光，不能作为独立确认或描述组件贡献；PC0–PC5 不改判。`code-debugging` 的逐项数值及候选身份检查用于约束本次审计。

[协议](PC_TWOROOM_LOCAL_B_BATCH_AUDIT_PROTOCOL_20261107_CODEX.md) · [GPU 审计记录](../le-wm-main/results/pc_tworoom_local_b_batch_audit_20261107_codex.json) · [代码](../le-wm-main/experiments/pc_tworoom_local_b_batch_audit_20261107_codex.py) · [测试](../le-wm-main/tests/test_pc_tworoom_local_b_batch_audit_20261107_codex.py)。
