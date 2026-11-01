# 四臂单根主机 CUDA 工程运行

2026-10-30，固定已知根的工程入口经主机 systemd 服务
`four-arm-engineering-20261030-main01` 运行完毕，服务退出状态为 0。
报告 `le-wm-main/results/codex_four_arm_host_cuda_20261030_HOST01/report.json`
的 SHA-256 为
`77d0ecaac24c9be7e1635dc4c1f35a610381dc0177a592ab3948f0f3b81d23a3`。
报告的四臂均为 complete，设备为 CUDA RTX 5060 Ti，模型权重和 buffer
运行前后哈希一致；峰值 allocated/reserved 分别为 142633472/146800640 字节。

持久账本消费者报告 structured 的 predict/action-encode 各 30、VJP
192；native 和 random 各 27 次 predict/action-encode；immediate-progress
各 30 次。各账本自身状态均为 `closed_unverified`，明确标记
`actual_usage_verified=false`、`coverage_verified=false` 和
`fairness_proven=false`。上述为账本已记录上界，不是完整总算力证明。
报告中的 elapsed 从 `run_cuda` 内部开始，不能等同服务总耗时；
systemd 记录服务耗时 2 分 55.568 秒。

该运行没有物理执行，根也不独立。structured 是 ridge proxy，不是正式
D2P 谱程序；native 是 zero-center CEM 变体，不是认证的强基线。
checkpoint 训练 normalizer 来源尚未认证，输入采集、模型加载、内部操作、
物理成本也没有完整计入。故此结果只证明该工程接线完成了本次冻结模型
CUDA 运行，不证明 P1 机制、真实收益、等预算公平或任何基线优势。
