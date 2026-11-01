# P2 同反馈候选物理回放结果

探索性固定四个 base × selected/F 候选 1，共八槽；均从对应的真实 cp15 反馈上下文所产生的同一批已付费候选导出。该对照在既有结果已知后固定，不是预注册。四个 base 只有两个已知根；base2/3 使用相同反馈上下文，不能当独立样本。

| base | 根 / 旧规划臂 | selected | 固定候选 1 | 验收 |
|---|---|---|---|---|
| 0 | 0 / transfer | 30 步完成 | 第 29 步返回后 RGB/canvas 守卫拒绝 | 部分日志检查；原 strict 拒绝 |
| 1 | 0 / joint | 30 步完成 | 第 21 步返回后 RGB/canvas 守卫拒绝 | 部分日志检查；原 strict 拒绝 |
| 2 | 1 / transfer | 30 步完成 | 30 步完成 | 保存 strict 通过 |
| 3 | 1 / joint | 30 步完成 | 30 步完成 | 保存 strict 通过 |

八槽 cp15 原生和 RGB 精确前缀守卫均在第 16 步 dispatch 前记录。两失败槽不是合法早停；保留错误、已付费调用、图片和不可用后续 checkpoint，没有填补或重试。完整 strict 的拒绝发生于部分三图捕获：日志已记录首个 off 图，但捕获函数只有整个三图组成功返回后才更新聚合 trajectory；原 consumer 不接受这种不一致。未修改已绑定运行源码或证据。

新独立文件的窄范围检查对两个失败目录核验外部 report/completion SHA、输入绑定、事件 hash 链、公共 action 的调用次序与位串、前缀 guard、所有 journal 引用图片字节及文件清单。该检查复用已封存 runner 的事件链和 guard 算术，并不独立重算所有 RGB 像素和原生标签，也不能代替 full strict，状态为 `partial_journal_checked_not_full_strict`。两个完整目录通过该 runner 的完整保存消费，但这不是重新执行神经模型或独立物理复制。

完整两 base 的物块位置误差均为 selected 42.144748、候选 1 135.315419；wrapped 角度分别 0.470906、0.371725 弧度，原生成功均 false。两 base 的结果重复，不能按两根计数。其余两 base 候选 1 未到第 30 步，不以失败前终点代替完整结果。

合计公共 step 尝试/返回 310/310（含 80 warmup 与 230 plan），底层 physics step 3132/3132；资源按各 job 账本计，内部其他开销及总 FLOPs 未测。没有新神经调用或 GPU。共享已付费候选过程不等于两次独立等预算实验；不声称物理公平、机制、P2 或科学通过。

各实际文件在 `results/codex_p2_feedback_candidate_physical_20261030_v1_RUN_JOB{0,1,2,3}/`；每个目录的 `report.json` 和 `completion.json` 原样保留。job2/3 完整 receipt 为 `results/codex_p2_feedback_candidate_physical_20261030_v1_RECEIPT_JOB{2,3}.json`；job0/1 窄范围失败 receipt 为 `results/codex_p2_feedback_candidate_partial_20261030_v1_RECEIPT_JOB{0,1}.json`。

完整入口及旧守卫联合回归 47 passed；部分检查附加 1 passed。运行命令模板和资源边界在 `codex_p2_feedback_candidate_physical_20261030_v1.md`。
