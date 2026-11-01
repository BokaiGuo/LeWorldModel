# E3 cell10 主机反馈探索运行

2026-10-30，已知根 cell10 的主机服务
`e3-cell10-cooperative-20261030-main01` 退出码 0；父日志记录 CPU
前置检查返回 0、随后 native host 返回 0。冻结模型神经计算在主机 CUDA；
单机 claim 为合作式，不阻止旧入口或其他主机。

输出 `le-wm-main/results/e3_cell10_cooperative_native_codex_20261030_v1_MAIN01`
的 report SHA-256 为
`7d4d663b0f2fb22812299029fe65990de5bdf217ab8b5f9e60872ec35fc61419`，
completion SHA-256 为
`d60017da5216e616a38c5393fdfde4070f65f55e4ebacf727b452b8af7996ebe`。
报告记录 transport completed，不自行认定科学通过。

主线程使用既有 continuous 和 integrated 保存消费者，绑定上述哈希及
authority SHA `33064f0c5bae40aa06d662acd0673bf0127eb7faad1ff98e4247977ef84cf484`
检查：状态 `completed20`，10 warm-up、20 control steps、32 帧且没有早停；
边界 10 和 15 各有一次 encoder/projector 完整调用，并接入后续控制节点；
帧匹配无遗漏且 terminal row 通过该消费者检查。这是已有 control5 后
继续真实反馈的单次执行，不能说所有节点均由本次重新规划。

`full_saved_strict=false`：消费者仍指出模型和编码器实例身份未封存、
完整 native snapshot 与失败路径语义未完全验证；物理重放未执行。
成本不包含先前 control5，GPU 时间和 FLOPs 未知。单根、无公平对照，
所以不构成 E3 多任务泛化、P1 通过或算法优势证明。

另用现有 native snapshot 语义检查器重算：37 处保存快照、32 次观测、
20 次动作派发和 20 次返回均完成，cleanup 返回，终态为 `completed20`。
检查器仍明确不认证独立物理／神经重放、未记录的 goal-bind 渲染及 RNG
状态、或完整失败路径；其结果不升级 `full_saved_strict`。
