# P3 四槽真实反馈工程运行

封存协议 `le-wm-main/results/p3_matched_order_feedback_host_v2_codex_20261029_protocol_MAIN01.md` 的固定四槽已于主机 RTX 5060 Ti 上串行运行，冻结模型，不训练。事前查看 GPU 仅显示占用、空闲存储约 22 GB；物理指本机仿真环境而非外部实体设备。四槽顺序和失败处理不变。

报告 `le-wm-main/results/p3_matched_order_feedback_host_v2_codex_20261029_RUN_MAIN01/report.json` SHA256 `02b5f2094de56faf807bbb552b43492b05021dc7caac79f7966ce4734da0c8cd`；completion SHA `fc3ab4a3aa21c28f28f81d3fa97f14f9f4038a41636da0bb71c13c01f0327a30`。报告 `four_complete_candidate`，耗时约 51.02 秒，errors=[]，模型散列一致、hooks 恢复。该 wall 是报告范围内的计时，不代表完整准备和消费总成本。

独立 saved consumer receipt `le-wm-main/results/p3_matched_order_feedback_host_v2_codex_20261029_receipt_MAIN01.json` SHA256 `0820addb1025ae79b37172767ce34e43ca76350825b90903752d2a6f710d6b23`，状态 `four_saved_feedback_engineering_verified_known_roots`、`full_saved_strict=true`。四槽各记录 20 个物理控制步、4 个编码边界和 22 次 native 调用，均通过观察上下文重建。该验证依赖封存来源和保存 hook，不是硬件证明或独立物理重演。

两个根均已知、fresh=0；本轮没有完成等总算力或 FLOPs 比较，`scientific_pass=false`。四槽工程闭环完成不意味着 P1/P3 或动态计算收益成立；原生任务结果和同时间成本分析须绑定本 receipt 分别核查，不能从 `four_complete_candidate` 推断成功率。
