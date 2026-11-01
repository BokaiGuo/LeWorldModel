# P1 模型内重选动作的 CUDA 重放交接

固定 8 臂各两个动作：原联合终态 winner 和模型内 non-improvement
重选 winner。`p1_uphill_cuda_replay_20261030.py` 复用现有完整
CUDA 双路径重放与独立保存态消费者；两条动作来自同一动作库，
但都按事先固定 roster 保留，即使动作字节重复。

准备文件
`le-wm-main/results/p1_uphill_cuda_replay_20261030_prepare_RUN01.json`
SHA-256 `94a4faf335f8411a5d0af05a563e77d52e3627a44d190a8d148ce524b6da4ab8`。
主机 CUDA 运行结果
`le-wm-main/results/p1_uphill_cuda_replay_20261030_RUN01.json`
SHA-256 `67fb9fe7be0acce5392685be7edba52b170bbd7fb46a73ad87f905343dd3843f`，
生产报告 `complete=true`，保存了逐包预测、归一化动作和上下文。
主机 CUDA 以冻结 checkpoint 执行，未训练、未做物理实验。

独立保存态消费者已通过 16/16，输出
`le-wm-main/results/p1_uphill_cuda_replay_20261030_consume_RUN01.json`
SHA-256 `26a07821cb9cf9b9ef9c89d0fe2e4b4f2f14c0bafd310bd3070b89fe628d12f4`。
消费者重核前缀及终态预测、两条路径、动作标准化、逐包历史和
输入身份；消费者本身不重新调用网络。消费通过也不自动授权或证明真实物理机制；还须单独
准备物理合同，绑定两种角色与全部动作，执行预定 roster，
并分别分析前缀、共同后缀和自主优化后缀。

另一个 24 anchor H10 共同后缀作业仍在运行；不修改其源文件、
协议或目录。这里只是已知根探索，既无独立新根也无完整等预算
强基线，不能开放 P1。
