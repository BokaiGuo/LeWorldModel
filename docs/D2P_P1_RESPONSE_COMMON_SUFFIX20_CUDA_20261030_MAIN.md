# P1 共同后缀二十次更新及上下文检查

在观察五次更新的结果之后，对相同八臂、相同四候选集合、相同
模型预测后继、零 raw 初始化及 0.05 步长，新开二十次更新的
模型内开发实验。主机 RTX 5060 Ti 完成 54 个不重复的臂内候选；
checkpoint 参数和 buffer 散列前后相同。显式代码路径共 1134 次
action encoder、1134 次 predictor、1080 次 backward；这些不是
独立 hook 认证的全流程账，含重叠候选复用，不含先前候选生成。

在本次条件下，54/54 个候选的模型终态 MSE 相对五次更新更低；
但八臂中多样性和即时筛选仍均选中各自共享的 index 0，分数逐臂
相同。两次实验不同计算预算，且根、臂和候选并不独立；这不能
证明真实控制改善、动态多样性无效或理论失败。这里只测试零
初始化和固定后缀时域，也未执行物理。

新增检查将候选 prefix 数值与保存 raw float32 位逐项比较，验证
预测后继中的前两帧 latent 历史、上一动作包及归一化 prefix 包
与根上下文相容；54 个候选均通过。它没有重新预测前缀、没有
认证 checkpoint 训练 normalizer 来源，也不证明模型输出与真实
物理一致。测试覆盖位串和历史篡改拒绝，1 passed。

结果
`le-wm-main/results/p1_response_common_suffix20_cuda_20261030_main_RUN01.json`
SHA-256 `9b29bfd041bd0ed73ba03c3fe2add8bc22522de6a319326a02ea68b9c0f6a053`；
代码
`le-wm-main/experiments/spectral_reachability/p1_response_common_suffix20_cuda_20261030_main.py`。
P1 admission、独立根、真实 RGB 后缀重规划和强基线公平比较均未
因此完成。
