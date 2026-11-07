# 双轨规划执行记录接收入口

新增 [执行记录接收入口](../le-wm-main/experiments/pc_execution_evidence_intake_20261107_codex.py)。
它复用既有 `strict_eligible`，要求读取并哈希协议与任务文件；依轨道
要求实际记录 `score`（RGB）或 `predict`（state-assisted）；检验
输入收据与轨道、算法和执行状态。接收结果固定为
`scientific_status: not_evaluated`，不自动通过任何 PC gate。
输出文件使用排他创建，不覆盖既有记录。

相关输入权限、旧登记器、严格登记器和新入口测试共 **44 passed**。
使用已有真实双轨 GPU smoke 结果演示入口，两个本轨记录均可接收，
输出分别见 [RGB](../le-wm-main/results/pc_dual_track_smoke_20261106_codex/primary_rgb/intake_20261107_codex.json)
和 [state-assisted](../le-wm-main/results/pc_dual_track_smoke_20261106_codex/official_state_assisted/intake_20261107_codex.json)；
本轮没有重新运行模型。smoke 接收只说明文件绑定和记录格式合格，
不是 PC0 的科学通过，更不是 RGB-only 闭环或机制证据。

本轮检查中，底层 `validate_evidence` 确实不检查输入字段；但现有
`pc_track_registry.eligible` 已检查输入通道，已有 `strict_eligible`
也检查张量收据格式。本入口是将已有严格检查用于明确的规划记录
接收流程，不是首次发现或修复所有登记问题。

**限制：**这是一个新的显式入口，旧 `eligible` 仍可直接调用，尚未
证明所有 runner 或所有人工登记流程都经过新入口。收据和哈希不证明
原始张量、模型、执行或调用过程未被伪造；也不能证明训练外独立性。
本轮未改写旧实验或历史登记器，PC0–PC5 不改判。
