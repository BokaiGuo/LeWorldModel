# 本对话交付的逐项验收

日期：2026-10-27。上一目标轮属于实质进展：新增独占接口、证据表、测试及实际离线验证结果。本轮重新检查当前文件，而非沿用上轮完成声明。

本次只验收用户要求的“基线证据表与可执行补实验交付”；不是四系统效能复现全部完成，也不是P1或研究科学准入。

| 用户要求 | 当前权威证据及验收边界 |
| --- | --- |
| 先读README及指定六文档、避免重复 | 上轮已读六文档及明确引用报告；本轮重读当前README和完整交接说明。已有238单元格、moving综合、H1核算与失败probe均保留；无重复模型/物理运行 |
| 四方法来源、版本、页/表 | evidence_handoff第2节及v2 JSON代表cell原始source。3个PDF与4份HTML当前字节重新校验；PlanWAM PDF页明确missing，不伪造、不推算 |
| 环境/数据/checkpoint/历史/horizon/预算/指标/样本数 | evidence_handoff第4节四列矩阵；JSON四方法各14维合同。通用论文设定与具体cell绑定分开；未知权重/episode身份不被通用配置代填 |
| 作者值和本地值分栏 | 交接第3节及JSON reported_column/local_column；GRASP模型内SR与本地物理native/Q严格区分 |
| 协议不匹配不算提升率 | 四方法direct_comparison_authorized=false且improvement_percent=null；匹配的open-loop单维不被升级为全协议匹配 |
| 实现身份与缺失 | GRASP bounded和LeWM-backbone CEM均local_variant；Hi-LeWM与PlanWAM可运行完整管线缺失列为null，不填零或冒充官方 |
| 独占adapter/协议/测试及实际独立验证 | bridge.py的build/local_projection/compare_contracts/weighted_rounding_check；36项测试本轮重新通过；Table13行列分母与Table3舍入区间实际联检，无新神经效能声明 |
| 可消费文件与SHA | evidence_handoff、bridge.py、bridge_test.py、delivery_v2 JSON；docs/chat_baselines_20261027_SHA256SUMS的4项本轮全部OK |
| 主线程隔离、无子智能体、无训练/CPU神经降级 | 本轮仅命令行只读检查、既有独占测试与本新文件；没有调用主线程producer、prepare、consume、assemble或代码审计；未加载torch/模型/模拟器。未启动神经运行，因此无需新GPU占用 |
| 只新增前缀文件、保留改动/封存、无提交推送删除 | 本轮唯一新增为本文件，apply_patch创建；旧交接、源码、测试和v1/v2产物保持原样 |

本轮额外执行了独立的交付一致性断言，而非只看绿色测试：

```text
fresh build() == 已交付delivery_v2 JSON：true
四方法 × 14维合同：完整
10个pinned输入：当前SHA全部通过，build结束再次复核
GRASP paper/local open-loop维：match；不授予跨协议性能比较
四方法提升百分比：全部null
v2 JSON SHA256：dc5cf07f74c8e7ca2fcb8b017da5361eac967c7d1f1d1c94f7d5f2c8055157fe
新增neural_calls / physical_steps：0 / 0
pytest：36 passed in 0.22s
```

入口仍为 `/home/gbk/code/LeWorldModel/docs/chat_baselines_20261027_evidence_handoff.md`；完整输入SHA见最终JSON的input_sha256，不另建第二个registry。

公开保留的科学/资产缺口：PlanWAM PDF页与官方驾驶管线；Hi-LeWM完整高低层预训练资产配对；LeWM Table10误差定义/梯度预算；GRASP官方全协议绑定与新独立根确认；各方法完整成本与原始episode/checkpoint映射。
这些是本次已交付的缺失项，不是被测试消除的效能问题。当前可消费接口可以检查未来补齐的合同，但没有隐含训练/下载/执行能力。
