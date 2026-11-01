# P3 在线接线：资源结算与 strict receipt 审查

本次仅 CPU 审查、修复和保存数据复核，不运行 GPU、模型、物理／仿真，不宣称预注册或科学通过。原 host runner、原 strict consumer、预算合同、旧结果与旧计划均未修改。

## 具体问题与修复

1. 原 binding 的 `inspect` 接受调用者指定 SHA 的 receipt，再交给预算汇总器；SHA 仅绑定字节，不证明 receipt 确由 strict consumer 生成。它允许跳过 `consume` 阶段，直接放置声称 full strict 的文件，存在将未独立认证内容作为已认证输入的风险。现在 `inspect` 重新运行原 saved strict consumer，要求其重建结果与 pinned receipt 全字段相等，且 strict 标志必须是 bool；不相等时拒绝，分析文件不生成。该验证仍继承原 consumer 的来源信任，并非硬件远程证明。
2. 原预算汇总器检查 complete 槽 attempted/completed 与预定向量相等，但不检查每槽真实 ledger `caps` 就等于预定向量；较高 cap 下碰巧花出相同数值不能证明共同上限。现在 strict 重建后重新核 ledger 账目并逐槽检查 caps、attempted、completed 全部等于合同向量；任一缺失或不符拒绝完整结算。
3. 既有结果未在在线 binding 的结算中明确展示物理 meter。现在逐槽记录真实 journal 中的 physical meter intent 向量、4 个观察编码边界，并逐根分别报告 meter 向量是否相同；数值 ledger、物理 meter 与编码不加权合并。**这是已记录仪表的对照，不证明等总算力**。partial 的该新增结算保留 null，不将失败或未运行补零；原 partial receipt 和其固定槽状态仍保存。

原严格消费器自身核 journal、资源 attempt/return/failure census、hook IO、观测及 action；本次复用而不改写。预算输出中的 shape 匹配仍单独报告；meter 不等不会被改写成相等，单项计数相等也不授予科学或等 FLOPs 结论。单槽完整 wall、CUDA ns 和 FLOPs 继续为 null；数值 ledger wall 包含可能交错的仿真/编码工作，不视为纯 GPU 计算。

## CPU 验证

- 新旧 binding、预算、host、联合语义故障测试：**29 passed**。新增覆盖 receipt SHA 正确但内容不等于 strict 重建时拒绝、实际向量相同但 caps 偏大时拒绝、完成计数不足拒绝；现有 synthetic 测试覆盖配额、动作、根、终止、帧与显存限制。单元测试中的 strict 和 ledger 部分使用替身，不构成物理证据。
- 对已消费的 RUN_MAIN01 仅做 CPU saved 重建：原 strict consumer 输出与已存 receipt 全字段相等；四槽 caps 均与固定向量相等，两个根各自的 physical meter intent 向量相同。此为**回顾性工程复核**，不能倒签为未来注册，也不能证明泛化。
- 因 binding 源码哈希改变，原 `ONLINE_BIND01` 计划已被实际验证拒绝；原文件保留，不得使用。完成修复后只读重新核原 prepare/合同并创建新独占计划 `le-wm-main/results/p3_feedback_online_binding_20261030_v1_plan_ONLINE_AUDIT02.json`，SHA256 `d43c5f20290114bb14c37e3c42a16f2b36e69eb5aceea764ded5750075d3b3d1`。该计划不是资源预留或授权，尚无独立可信时间戳；未来如要运行，仍须外部审读和可信时间戳、资源协调及所有原 runner 运行门，且不能把本计划用于已发生的运行。

此次新结算保留每槽已计量字段，未来运行要通过原 strict、共同 caps、完整实际向量、形状和物理 meter 各自的检查后才能分别描述相应结果；任何未知项仍为未知。没有新在线结果。
