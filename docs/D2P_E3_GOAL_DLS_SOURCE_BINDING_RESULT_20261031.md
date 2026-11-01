# 目标 DLS 初始化器来源绑定修复：重验结果

根 0 主机 CUDA / 物理重验已完成；与前一目标 DLS 运行比较，
四臂 `executed_raw` 和 `final_state` 均逐项相同，成功状态分别
为 structured 失败、native 失败、random 成功、immediate 成功。
这是一根既有开发任务的复验，不是额外独立样本。来源错误不构成
上述失败的已证实原因。

新保存态检查核验了 15 个节点的合同 SHA 字段及账本、动作绑定，
并复算 4 个 structured DLS 初始化；明确不声称独立重演完整物理
事件链。以旧结果作为负例时，检查器按预期拒绝其旧来源字段。
来源字段和数值测试共 4 项通过。合同和错误说明见
`D2P_E3_GOAL_DLS_SOURCE_BINDING_20261031.md`。

- 修复后报告 SHA256：`62dc2d6e8a5bd3265b0b92a9c01c87625c43cdcba243ae5fd5d24aa09e28def8`
- DLS 初始化器 SHA256：`9726f2b06866b946da57294905a9068c7fad268ec7aba49b26b33b1af185c26f`
- 核验产物：`le-wm-main/results/e3_cp10_dls_sourcefix_saved_20261031_root0_CHECK01.json`
