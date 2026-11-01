# P1 中间状态＋终点目标：CUDA 和物理开发结果

按 `D2P_P1_PACKET_OBJECTIVE_PROTOCOL_20261031.md`，对 24 个已知根
前缀执行固定四起点 Adam，目标是第 5、10 步预测 latent 误差的
等权平均。结果保存全部分数、动作、调用计数及输入来源；主机
CUDA 设备 `cuda:0`，冻结模型前后哈希一致。新目标的 24 个选定
动作哈希均不同于旧终点-only 方案；每角色各 8 个中，selected、
zero、greedy 分别有 6、5、4 个改变了所选起点。

新动作按已封存前缀、真实 cp5 图像与固定身份绑定，24 槽均
完成物理执行并通过保存态消费；没有提前停止，全部有十步终点。
原生成功 selected、zero、greedy 分别为 **0/8、0/8、0/8**。
物体终点位置误差 selected 低于 zero 为 6/8、低于 greedy 为
1/8；物体终点角度误差 selected 低于 zero 为 3/8。上述只是
同根上的逻辑槽描述，部分动作和根重复，不能视为独立统计，也
不证明非贪心控制能力或对强基线的优势。

历史相同四起点与更新次数、终点-only Adam 的记录中，原生成功
为 selected 0/8、zero 5/8、greedy 0/8；部分成功轨迹提前停止，
不能将其当有固定第十步终点。新旧是同一批已知开发根上的探索性
目标对照，后缀调用计数相同不代表完整流程等预算；新实验并未
改善原生成功。不能仅凭这些结果判定轨迹目标理论无效，也不能
声称已经排除输入归一化、表示与物理评价失配等问题。

结果和来源：

- CUDA 计划 `le-wm-main/results/p1_packet_objective_adam_20261031_RUN01.json`，
  SHA-256 `3614dad72f2a1ca38662c390e326a543bec927d512742ce8adf8aae422f987e1`。
- 固定物理输入
  `le-wm-main/results/p1_packet_objective_physical_20261031_prepare_RUN01.json`，
  SHA-256 `47c46d54bf2f1f511ab60488fc3eb3d9c0034b7b481e8e29471e48c3895c5079`。
- 执行报告 SHA-256
  `c4ab3d99255d40c16759793867333f98f55da1f7741fc051e749bfb566f62d4c`；
  保存态消费 `le-wm-main/results/p1_packet_objective_physical_20261031_consume_RUN01.json`，
  SHA-256 `7fdc29a1c22179c2123bfa97ba392e9ad62c1657888715f2e395d6a0ab3aaed1`。
- 两项测试覆盖两包评分的梯度回传、固定 24 行绑定以及分数和
  动作篡改拒绝。生成的是规划动作，不是训练；P1 gate 未变。
