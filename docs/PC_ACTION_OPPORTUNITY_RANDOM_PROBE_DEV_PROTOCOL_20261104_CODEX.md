# Action-opportunity random-probe Stage-1b 协议（2026-11-04）

上一轮常值/交替 probe 没有预测价值。本轮只检验一个替代解释：
probe 覆盖不足，而不是后续作用描述本身无信息。

- 候选库、root、评分器和封存 CEM 目标完全不变；
- A 为固定随机后缀 bank，8 个后缀，幅度 `0.5`，seed 由后缀长度和 split 固定；
- B 为另一独立固定随机 bank，8 个后缀，不参与 A 描述；
- 每个候选使用 zero tail、A、B 一次批量评分；
- 主要指标仍是 A/B 描述一致性、A 与封存 `best_score` 的排序关系；
- 不挑选 root，不训练任何模型，不修改 retention 或正式 planner。

如果随机 probe 仍不能预测封存后续质量，则对“仅扩大有限 probe bank 即可修复”
形成有界负结论；不据此否定更广义的行为后果表征假设。
