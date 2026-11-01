# P3 Reacher 单 chunk 主机 CUDA 接线结果

这是零动作探索性接线探针，不是 P3 机制、性能或等预算评价。
固定阈值 `min_improvement=0.01`、`max_disagreement=1.0`，seed
`20261101`，使用已有的 Reacher 训练语义动作统计 receipt。首次固定臂
因示例指定了不合合同的统计文件，在模型加载和物理动作前失败；失败产物
保留于 `results/p3_live_probe_fixed_20261101_MAIN01/receipt.json`。

修正输入后两臂的主机 CUDA 运行均正常结束，保存 receipt：

| 臂 | 候选执行步 | 扩展预测次数 | 总环境步（含 warmup） | receipt SHA-256 |
|---|---:|---:|---:|---|
| fixed | 5 | 0 | 20 | `6d93799020a06d02c618b52fed868f2ba6ba9ba9bab1d6279416b585a689bb75` |
| dynamic | 10 | 8 | 25 | `f8f2100c12da7beeccc849220b3691883e72f42de6fafca2e71e65695c7bd876` |

动态臂第一次预测扩展改善约 0.01260，超过固定阈值，第二次改善约
0.00541，低于阈值；因此选择 10 步。两臂记录的目标 RGB、根 RGB 及
warmup 动作摘要相同，但这不是底层完整物理状态独立认证。两臂候选
均为零物理动作，且执行时域不同，因此没有任务收益比较。

动态臂共记录 9 次 action encoder 入口（含上下文准备）对固定臂 1 次；
两臂各记录 2 次视觉 encoder 入口。计量不包括完整 FLOPs、能耗和模型
内部算子；相同上限也不构成相同计算量。一次节点未测量闭环反馈、
成功率或有效计划质量，不能用于开放 P3 gate。

固定臂：
`le-wm-main/results/p3_live_probe_fixed_20261101_MAIN02/receipt.json`

动态臂：
`le-wm-main/results/p3_live_probe_dynamic_20261101_MAIN01/receipt.json`
