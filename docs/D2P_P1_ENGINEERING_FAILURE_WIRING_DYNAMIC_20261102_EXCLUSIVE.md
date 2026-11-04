# P1 saved consumer：停止、失败与终态的接线复核

本次对 `p1_dev305_contact_equal_scale_saved_20261102.py` 和
`p1_dev305_contact_rgb_feedback_saved_20261102.py` 做 saved-only CPU 复核。
不运行模型、物理或 GPU，不更改 gate、旧 consumer 或旧结果。

## 对前次结论的纠正

两个 consumer **均要求 `report["errors"]` 为空**，但不检查顶层
`report["status"] == "recorded"`；前次笼统表述“没有 `recorded + errors=[]`
guard”容易误读为连 errors 都不检查。实际缺口是**未检查顶层 status 与
errors 的一致性**。两个 consumer 输出的行中都保留 `status`，并未直接抹掉停止类型。

`visibility_guard_stop` 与 `conservative_margin_stop` 是各 runner 显式记录的
守卫/策略停止，不应一概称为工程异常；它们也不是完成五步时域的负结果。
真正的 `failed` 行及非空 errors 已被这些 consumer 拒绝。

## 已验证的行为

1. 实际 equal-scale 保存结果 12 行：7 行 `fixed_horizon_completed`、
   2 行 `visibility_guard_stop`、3 行零步 `conservative_margin_stop`。
   实际 RGB-feedback 9 行：5 行固定时域完成、4 行可见性停止。
2. 两个 consumer 保留行状态，但也为停止行输出 `native_success=false`；
   equal-scale 零步停止没有 reward，却仍输出 false。这个 false 最多表示该
   记录的终止标记不为真或无 transition，不等于五步后未成功。若下游将
   所有 false 无条件计入完整 horizon 失败，便会把删失误读为负结果；本次
   未证明任何特定下游已经这样做。
3. 在内存测试中先由现有 `base.read` 完成真实报告的 SHA 验证，再仅将返回
   对象顶层状态从 `recorded` 改为 `failed`，保留 `errors=[]`。两个现有
   consumer 均继续返回分析。这是有意破坏 SHA 后内存对象的一致性反例，
   **不是**说真实报告已损坏，亦不是声称可绕过正常文件哈希验证。它证明
   consumer 缺少该语义一致性检查；自洽哈希本身不验证 producer 状态语义。
4. 对照 consumer `p1_dev304_spectral_cp10_saved_20261102.py` 要求顶层
   `recorded`、`errors=[]`、每行 `fixed_horizon_completed`、固定长度及无终止/
   截断；这只说明这些特定状态检查存在，不是完整科学认证。

## 工程接线建议

任何新的严格消费层应单独核对顶层状态、errors、固定 roster、每行状态及
实际转移次数的一致性；分别输出 completed、native stop、guard stop、
engineering failure、not started，保留 attempts/returns 和删失时点。停止行的
当前观测可单独报告，但完整 horizon 成败应为 `null`/未观测，不能补成 false。
不要用单一“检查通过”取代来源、盲性或科学 gate。

## 验证

独占测试 `tests/test_p1_engineering_failure_dynamic_audit_20261102_exclusive.py`
与独占静态审计测试及两个既有 consumer 测试组合：
`10 passed, 1 warning in 149.73s`。警告来自 pygame 的 `pkg_resources` 弃用。
测试反例使用 mock，仅验证接线行为，不是新实验或科学证据。前次相关
preflight/runner 组合为 24 passed。旧静态报告和真实结果均保留未改。
