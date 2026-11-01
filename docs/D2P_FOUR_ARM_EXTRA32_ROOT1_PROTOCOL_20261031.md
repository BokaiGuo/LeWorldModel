# 第二已知根：对照赢家追加 32 次更新

探索性开发合同，固定在运行前。沿用第二已知 PushT 根、冻结
checkpoint、四臂、各自真实 RGB、动作历史及 0/5/10/15 反馈时刻。
structured 与 immediate-progress 不变；native-style 与 random 各自
经原有共同优化器和模型评分选出赢家后，再固定追加 16 次两步
优化，重新计算赢家分数，只在分数严格降低时接受。不能使用
物理 outcome 选择候选。随后各臂最多执行 20 步；保留停止与错误。

逐节点对 predictor 和 action encoder 前向 hook 计数；独立拦截
Python `torch.autograd.grad` 与 `torch.autograd.backward` 调用并与
账本的反向次数比较。该计数不覆盖其他直接调用 autograd 引擎
的路径，不是 FLOPs、完整编码成本或全部物理执行成本。
所有神经计算使用主机 CUDA，模型不训练。

四臂的统一逐轴上限取四臂成本的逐轴最大值；由于调用类型不同、
各臂实际使用不等，以及前缀之外的开销未认证，这不是全流程
等成本或公平强基线结论。native-style 不是官方 LeWM/GRASP。
已知根重复用于开发，不能声称独立确认、P1 通过或显著优势。
