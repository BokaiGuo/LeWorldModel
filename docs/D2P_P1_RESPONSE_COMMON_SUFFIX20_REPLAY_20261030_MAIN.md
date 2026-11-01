# P1 结构化候选：20 步后缀模型路径重放

已对保存的 54 个候选在主机 CUDA 上重新执行相同模型路径：根上下文和前缀生成预测中间 latent，再从保存的中间 latent 与后缀计算终态目标分数。入口同时检查源面板、候选集合、动作字节哈希及赢家绑定。

- 54/54 候选的前缀 latent 最大逐元素绝对误差为 0；终态分数最大绝对误差为 0。
- 显式计数为 action encoder 108 次、predict 108 次；模型参数和缓冲区前后哈希一致。
- 结果状态为 `complete_cuda_replay`，设备为 `cuda:0`。
- 输入结果 SHA-256：`9b29bfd041bd0ed73ba03c3fe2add8bc22522de6a319326a02ea68b9c0f6a053`。
- 重放结果 SHA-256：`4f0c5fd74436c74001fc3fa800b1ae523cc43a47ad38576125ae9b6c2a9310dc`。

结果文件：`le-wm-main/results/p1_response_common_suffix20_replay_20261030_main_RUN01.json`。
实现和篡改拒绝测试分别位于 `le-wm-main/experiments/spectral_reachability/p1_response_common_suffix20_replay_20261030_main.py` 和 `le-wm-main/tests/test_p1_response_common_suffix20_replay_20261030_main.py`。

这是同一模型实现的保存路径重放，不是独立模型 ABI 验证、真实 RGB 对照或物理执行。它不能排除归一化来源、共享实现错误、预测失真、任务评价失配，也不能认证全链路等预算。前述两种候选筛选在八臂中选到相同动作，仅说明这批模型内探索没有区分出不同赢家；不能由此宣称理论失败或 P1 通过。新重选前缀的共同十步物理面板须完整封存后才能消费，且独立后缀规划收益必须另行评价。
