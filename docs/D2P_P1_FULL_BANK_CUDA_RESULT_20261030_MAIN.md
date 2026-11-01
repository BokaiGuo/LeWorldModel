# 全候选选择主机 CUDA 重放

固定清单
`le-wm-main/results/p1_full_bank_cuda_replay_20261030_main_prepare_RUN01.json`
SHA-256 `37488b452ba5e3e0c64428c9d30dabde3ed975803191b6ac43ed1bfd2b65c462`。
8 臂 incumbent 与全候选选择各一行，全部在主机 CUDA 用冻结模型
重放，结果
`le-wm-main/results/p1_full_bank_cuda_replay_20261030_main_RUN01.json`
SHA-256 `a86fc9b1a283e1001a173ef3bc0ca6f3249b219a1152dc762441b8abb930b2b6`。
执行器报告 16/16 连续和分段 rollout、源前缀、源端点及前缀动作
历史完全一致，模型前后哈希一致。独立保存态消费者复算动作
归一化、完整数组和上下文后确认 16/16 一致；消费者没有重新
计算神经网络，也不授权物理执行。

本次没有训练或新物理执行，使用已知根和事后规则；候选库 20
项的搜索成本已经发生。它是接线证据，不是 P1、泛化或公平
基线优势。
