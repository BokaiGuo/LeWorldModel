# 等预算尾部实验：GPU 分数重算结果

按照 `PC_EQUAL_TAIL_GPU_SCORE_REPLAY_PROTOCOL_20261107_CODEX.md`，从三个
开发 root 的重置渲染图及原目标图重建输入，使用相同冻结模型、
normalizer、动作缩放及原批大小 16，逐批重算了全部 720 个候选。

最大绝对差为 **0**；超出预定 `1e-5` 数值容差的候选为 **0/720**；
15 个前缀臂的最终最低分选择改变 **0/15**。模型指纹在重算前后
一致，封存计划、候选池、原协议及原 runner 的哈希通过检查。

原始分数、每臂选择和权限收据保存在
`le-wm-main/results/pc_equal_tail_gpu_score_replay_20261107_codex.json`；
脚本为 `le-wm-main/experiments/pc_equal_tail_gpu_score_replay_20261107_codex.py`。
环境仍出现 reset 返回 Tensor 与声明空间不一致、动作类型转换警告。

本结果说明原等预算实验的模型评分及最终最低分选择在同一环境、
同一实现和 checkpoint 下可复现；不证明评分与真实控制一致，也
不认证动作时钟、物理重放或独立模型。实验使用事后提供的共同
尾部及已曝光任务，不改变 PC0–PC5 状态。
