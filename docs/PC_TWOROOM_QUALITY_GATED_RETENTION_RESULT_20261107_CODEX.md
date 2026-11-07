# TwoRoom 机会质量约束的后果保留：开发结果

基于既有结果中“向量规则的第二前缀 19/40 次处于 A 探针机会排名后四”的观察，按事前协议对 40 个已曝光根检验：先保留探针最优前缀，第二个槽位只在前四名中按后果向量距离选择。固定原前缀、A/B 尾部和完整链 incumbent；另比较普通 lookahead、无约束向量以及同样受约束的动作身份打乱。模型选链封存后再真实重放。

| 方法 | 曾成功 | 末步成功 | 相对约束向量的配对独胜／独负 |
|---|---:|---:|---:|
| 普通 lookahead | 38/40 | 37/40 | 0 / 0 |
| 无约束向量 | 38/40 | 37/40 | 0 / 0 |
| 质量约束向量 | 38/40 | 37/40 | — |
| 质量约束身份打乱 | 38/40 | 37/40 | 0 / 0 |

质量约束在 **19/40** 根改变了相对无约束向量的前缀保留；但四臂最终选择的动作在 **40/40** 根完全一致。其中 37 根选原搜索池 incumbent，3 根选 A 候选，**没有选择任何 B 链**。因此这种保留变化在最终实际执行上没有体现。本实验没有执行全部 B 链，不能由此断言 B 中不存在更好的物理链，也不能把模型未选择 B 直接归咎于模型或生成器。

每臂逻辑费用为原搜索池 600、A64、前缀8、incumbent 复评2及该臂 B16，共 **690 次完整链评分**；联合执行的 B 查询平均每根 28.4 条，不能拿联合查询节省冒充单臂完整成本。缓存仅复用同一观测的编码。实际模型评分先于物理结果读取，真实状态仅供评价端复位与评估。相关测试 **10 项通过**；独立选择审计重建了全部 40 根的保留、实际 B 查询、候选分数选择、动作指纹与成功汇总，但未独立重跑模型，也不构成独立环境认证。

**决定：**没有达到事前规定的相对 lookahead、无约束向量和打乱对照分别至少两根独胜的开发出口，停止扩大这一具体规则。这个负结果只限制当前候选分布、评分和保留方法；不能证明所有后果描述无价值，也不能改判 PC0–PC5。后续若继续此方向，必须先解决“额外查询的链没有打败共同 incumbent”的可检验问题，而不是再单纯调整保留排名。

[协议](/home/gbk/code/LeWorldModel/docs/PC_TWOROOM_QUALITY_GATED_RETENTION_PROTOCOL_20261107_CODEX.md) · [模型选择](/home/gbk/code/LeWorldModel/le-wm-main/results/pc_tworoom_quality_gated_retention_20261107_codex_plan.json) · [物理轨迹](/home/gbk/code/LeWorldModel/le-wm-main/results/pc_tworoom_quality_gated_retention_20261107_codex_physical.json) · [选择审计](/home/gbk/code/LeWorldModel/le-wm-main/results/pc_tworoom_quality_gated_retention_audit_20261107_codex.json)。
