# 独立对话：P3 计算调度

解决当前配额取整导致各策略相同的问题，预先固定有效调度对照，再验证等预算收益，不得事后调参挑结果。

主线程负责 P1 和代码审计，不重复其当前任务。不使用子智能体；仅写独占新文件，保留已有改动和封存结果。神经实验只用主机 CUDA，不降级 CPU；不训练冻结模型、不放宽科学门槛。

工作目录 `/home/gbk/code/LeWorldModel`；先读本目录 README.md。个人goal为“消除调度碰撞并交付预声明有效对照”，不是靠改配额得到有利结果。独占新前缀 `chat_p3_20261027_`，仅创建 docs/ 和 le-wm-main/{experiments/spectral_reachability,tests,innerdocs,results}/ 下对应新文件；apply_patch编辑，不改任何旧scheduler/结果。

先读并核实际产物：

- `/home/gbk/code/LeWorldModel/le-wm-main/innerdocs/moving_budget_scheduler_discretization_design_20261027_v1.md`
- `/home/gbk/code/LeWorldModel/le-wm-main/innerdocs/moving_budget_scheduler_prospective_separation_20261027_v1.md`
- `/home/gbk/code/LeWorldModel/le-wm-main/innerdocs/moving_history_budget_scheduler_cuda_20261027_v1.md`
- `/home/gbk/code/LeWorldModel/le-wm-main/innerdocs/moving_budget_scheduler_comparison_rgb_20261027_v1.md`
- `/home/gbk/code/LeWorldModel/le-wm-main/innerdocs/p3_joint_horizon_goal_metric_scheduler_20261027_v1.md`

历史观察是权重接近、整数配额均[10,10]导致策略退化成相同执行，不是调度理论被否定。验证碰撞链，设计固定total budget/minimum allocation/tie-break的离散分配规则；只用冻结动力学可见信息决定计算资源，不能看真实outcome后改温度、权重、root集合或预算。新run前封存规则与候选顺序，验证至少能形成真正不同的调度，若仍碰撞必须如实记录。分别报告predict API、predictor samples、编码/AD/SVD/wall与共享成本，不把等预测次数冒充等FLOPs。实现和测试实际可执行调度/队列/消费者，准备同root同目标同预算的固定与自适应对照；P1与freshness未通过时科学准入仍关闭。交付代码/测试/预声明/实际saved验证/SHA，缺失依赖精确列出，不自动删除/推送。
