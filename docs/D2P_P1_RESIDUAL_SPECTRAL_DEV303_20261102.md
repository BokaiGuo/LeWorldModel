# P1：残差条件多谱方向开发实验

现有候选器只取任务加权 Jacobian 的最大奇异方向；最大动作响应并不必然
最对准当前目标残差。本实验为每个既有 RGB 候选锚点按阻尼残差投影的
绝对值选择至多两个已解析奇异方向，各保留正负扰动，再沿用原 RGB
几何 guard 和有界非线性后缀试算。局部谱只负责产生候选，不代表真实
控制能力，也没有强制选择非贪心动作。

代码是新的开发分支，未修改历史候选器和封存结果：
`p1_residual_spectral_candidates.py`、
`p1_dev303_residual_spectral_candidates_20261102_main.py`。
相关新旧内核测试共 13 项通过。

已在已知 dev303 根完成主机 CUDA 模型内运行：

- 21 个候选完成非线性后缀评价，另外两个谱扰动被既定 RGB guard 拒绝。
- latent 评分选择 `spectral_0_mode2_+1`；goal-tangent 评分选择
  `task_greedy_direct`。二者不是相同任务评价，不能混作共同胜出。
- 记录到 768 次 VJP、46 次 SVD、3091 个 predictor 样本，
  以及 11 次 encoder 调用；`full_cost_certified=false`。
- 未训练模型；报告标记模型前后状态一致。该次没有进行真实物理执行，
  不证明接触、非即时进展、响应改善、等预算优势或 P1 通过。

结果目录为 `le-wm-main/results/p1_dev303_residual_spectral_20261102_main_run01/`。
协议 SHA256 为 `d0623327d4ba6a280dc742e6c627399477e03daa2fda0b7bad533acaff1e6104`，
报告 SHA256 为 `555eb599aa26de8d524849405e059a2366f10b40a5300acf14c57f5ffaeb3443`。
这不是公平预算下对旧方法的性能比较，候选数量和结构计算成本均有变化。

下一步应事先固定开发合同，在同根上执行被模型选中的候选及对照、
共同后缀和真实 RGB 同预算独立后缀规划；结果不得用于回写已有合同。
旧 ATM/MSE decision-transfer P1 与此 D2P 机制 P1 是不同科学命题，
该结果不改变任何旧 gate。
