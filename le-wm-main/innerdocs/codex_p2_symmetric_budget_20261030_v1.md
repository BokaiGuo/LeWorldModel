# P2 对称操作预算探索内核

本次仅新增独占内核和离线测试，不改旧实验、结果或科学门。既有 wallcapped CUDA 已运行：其共同上限不是等实际用量，自然完成 joint/composition 分别为 1902/1062 predict，且各次 Jacobian 的展开形状不同。最近四根调用匹配方案也有 Jacobian 1/41 的差异，不能复用其结果称总预算匹配。

新内核是探索性新算法，不是旧算法的等价实现：相同零初值，固定两次更新、四个 alpha，joint 使用完整 Jacobian，suffix 第一次屏蔽前 10 列、第二次屏蔽前 20 列；两个策略均计算完整 192×30 Jacobian 并对同形状矩阵做 SVD，所得方向用于真正更新，不用空操作填充。每臂自然完成请求 39 次 predict（包括每个 Jacobian 的三个 primal 和两次完整轨迹 replay）、2 Jacobian、384 个数学 VJP 行、11520 Jacobian 元素、2 SVD。450 秒为共同调度上限，不要求墙时相等；单次操作返回超时不能抢占，标 partial。严格更优才替换，平局保留先前解。数值常量和策略为看过既有结果后的探索设计，不是预注册。

这些是接口的**请求计数**，不是底层 hook 证明；两个完整 replay 当前只检查 terminal 位相等，尚未验证全部 context。实际运算耗时、FLOPs、内核反传、模型加载和设备上下文成本均未知，SVD 数量形状一致也不保证实耗相同。并且本内核的 logits clip 和 ridge 常量未与原生产动作域/求解合同绑定，不得将它描述为原 P2 物理计划。异常只保存在返回对象，尚未耐久化。时钟需由真实后端同步 CUDA；当前默认单调时钟仅适用于离线测试。

离线命令（工作目录 `/home/gbk/code/LeWorldModel/le-wm-main`）：

```
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_codex_p2_symmetric_budget_20261030_v1.py
```

测试使用 NumPy 线性假后端，非真实根、非神经推理；验证完整计划、计数、策略差异、失败扣账及截止拒绝。接下来仍需：绑定已消费四根的完整 task/context/goal 和真实模型权重及规范化；实现实际 forward/J/VJP/SVD hook 与同步时钟、不可覆盖的持久 partial journal；严格核对每次 primal、完整 replay context、来源和结果；在此基础上才能提供生产 CLI。由于这些前置缺失，本轮未启动 CUDA，不提供臆造的生产命令，也没有物理运行、训练、公平通过或新科学结论。
