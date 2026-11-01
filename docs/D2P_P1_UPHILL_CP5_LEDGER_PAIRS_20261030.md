# P1 cp5 初始化对照的保存态记账

对零、暖初始化各 24 槽，按物理准备文件及其 SHA 绑定的规划
结果、规划日志 SHA、合同 SHA 和根槽位/任务/前缀/RGB 身份重建
协作式记账链。新结果：
`le-wm-main/results/p1_uphill_cp5_ledger_pairs_20261030_RUN01.json`，
SHA-256 `80eda047018075d281d35ff9e5ac5abaa5f3eaca5e1e3181dd6159e11052e0c7`。

24 个身份配对的两组记录完全相同，每槽记录到：

| 阶段 | 返回的协作式调用 |
| --- | --- |
| 后缀优化 | 动作编码 40、预测 40、反向 20 |
| 最终验证 | 动作编码 2、预测 2、replay 1、selection 1 |

视觉编码的独立账本以原 `attempted` 字段保存，不将不同调用
类型相加。这里的 return 是受控 wrapper 返回事件；历史记录没有
证明所有模块调用均经 wrapper、没有认证总墙钟或 setup/候选生成
成本，也没有认证共享 selected 暖初始化的生产成本。因此不能把
这一结果称为实际全流程等预算、强基线公平比较或 P1 通过。

独立工具
`experiments/spectral_reachability/p1_uphill_cp5_ledger_pairs_20261030.py`
不调用模型和环境；测试覆盖真实账本、配对身份篡改、缺失或孤立
return 拒绝（2 passed）。它不重做原 CUDA 推理或物理实验。
