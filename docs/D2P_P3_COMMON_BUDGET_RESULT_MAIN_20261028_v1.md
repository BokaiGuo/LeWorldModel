# P3 预测调用预算对照：实际结果

本轮主机 CUDA 和 saved-only 严格验收均已完成，不是物理闭环优势证明。

报告：`le-wm-main/results/agent_p3_common_budget_20261028_v1_cuda_MAIN01/report.json`
SHA256 `3f7004aeb92d09a8f6010cd5a5c078a0660b19574203c38d5ae3fbe72dfab67e`。

MAIN 严格持久回执与32候选独立终态Q复算：
`le-wm-main/results/consume_P3_common_MAIN_20261028_v1_MAIN01.json`
SHA256 `bea94dc21c849b5d4db86ace36f26628c61c09693e30fb63b0af42cc2cfab02b`。
新增独立算术测试4 passed；验收不新增神经或物理调用。

| 已知根 | 策略 | 候选 | 规划/执行packet长度 | 总predict | 数学VJP方向 | 最低预测Q |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | fixed | 12 | 1 | 32 | 384 | 0.9454683415079467 |
| 0 | adaptive | 4 | 3 | 32 | 384 | 0.9086686598914356 |
| 1 | fixed | 12 | 1 | 32 | 384 | 1.2824975011868995 |
| 1 | adaptive | 4 | 3 | 32 | 384 | 1.2831926496736858 |

每臂独立支付H1/H3信号采集8predict、有效搜索24predict；全四臂128predict，其中新搜索96。
候选均为实际不同动作且逐条整体与分段重放，没有padding或借用另一臂半价J。
数学VJP方向不是已知完整engine backward成本；嵌套计时不得相加当FLOPs或整机耗时。

限制：两个已知根；fixed也支付双horizon信号，是共同pipeline消融，不是最省成本H1基线。
执行动作分别5与15raw步，故虽然预测调用数相等，终态并非同一真实时间。
模型Q一好一坏不是等执行长度科学对照，不证明C3物理收益、全树调度、P1或独立泛化。
已派发固定执行长度的新合同以补此缺口；不事后修改本轮结果或选择门槛。
