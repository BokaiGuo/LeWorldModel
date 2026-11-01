# P1 双包上坡生成器：主机 CUDA 模型内结果

固定两个已知根、四槽、每槽 stage descent/ascent 两臂，各保留 9 个迭代，共 72 个候选位置。原赢家来源、两段 float32 动作 chart 与模型任务均由真实保存消费重建。完整主机 CUDA 运行及事后全账本消费均通过：288 次预测及 128 次 scalar reverse 实际完成；八臂的 128 个 AD 标量均通过从目标和端点独立复算的检查。模型状态、精度和来源前后闭包在该消费合同内通过。各轴嵌套计数不得相加，也不声称等 FLOPs。

**八臂均无模型内合取见证**：严格要求同一迭代的 stage Q 高于初始 Q 且 terminal Q 低于原赢家 terminal Q。stage descent 四槽分别有 4、5、8、8 个迭代终态改善，但没有任何 stage 上坡。stage ascent 四槽分别有 4、0、7、6 个迭代上坡；对应终态改善迭代数为 3、0、0、0，且没有与上坡重合。第一槽 ascent 上坡与终态改善发生在不同迭代，不得把两者拼成见证。

该生成器与固定目标上的有限模型结果，不是全局不可行证明。已知根、模型预测和 identity-Q 都不能替代独立根的真实 RGB、控制能力或物理任务评估。无新见证因而不触发物理执行；P1 门不变，不能由这些结果单独归因理论或模型失效。优化及模型—物理排序断点仍应继续审计。

源报告 `le-wm-main/results/agent_p1_two_packet_host_cuda_20261029_v1_cuda_MAIN01/report.json` SHA-256 `73e0a684f3641c343d02d2537344479e47854ae4dbdc55cf181cad14f7042a3a`；完整消费 `le-wm-main/results/agent_p1_two_packet_host_cuda_20261029_v1_consume_MAIN01.json` SHA-256 `a5fd056c99860b5b43425cd68447bab52aaff77eeb3bafa3313e5c13fbb0ebc2`。该结果不训练模型，不声称 freshness 或物理释放。
