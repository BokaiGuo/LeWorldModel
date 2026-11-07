# TwoRoom 共同起点 Adam 与随机局部搜索：开发结果

按事前[协议](PC_TWOROOM_JOINT_RANDOM_DEV_PROTOCOL_20261107_CODEX.md)，对已曝光的 40 个 root，从相同 lookahead 链出发，模型选链封存后再分别重放。规划使用冻结 RGB LeWM；物理状态只用于重置与评价。

| 方法 | 25 步内曾成功 | 末步成功 | 平均末距 |
| --- | ---: | ---: | ---: |
| 原链 | 38/40 | 37/40 | 5.323 |
| Adam | 39/40 | 38/40 | 3.926 |
| 随机局部搜索 | 39/40 | 38/40 | 3.963 |

Adam 相对随机的配对独胜和独负均为 **0**，未达到至少两胜且至多一负的事前开发出口；停止扩大这一具体配置。上次出现新增成功的 root 39，此次两种精修均成功，物理末距分别 1.392 和 3.179；不能据此将成功归因于梯度。两种方法相对原链的共同新增成功仅为开发案例，不构成独立确认。

每个精修臂包括原链在内各评分 9 条；Adam 额外进行 8 次反向。同步计时的每根平均扩展时间为 Adam 0.155 秒、随机 0.066 秒；随机实现逐条串行评分而非批处理，二者不是等总计算预算的最优实现比较。时间均不包括共享的前序 lookahead、图像编码、模型加载、物理执行或重规划，也不是系统级成本曲线。

已保存所有 9 轮动作和评分、8 轮梯度以及完整物理轨迹；测试从种子重建随机链、用保存的梯度复算 Adam 更新，并检查最低分与最终动作绑定。相关测试共 **10 项通过**。模型计划与输出 SHA-256 绑定，执行前校验源文件、模型、输入及协议；模型阶段前后权重指纹一致。物理数据文件未在本次单独散列，评价复用同一环境实现；Gym 的 reset 类型及动作转换警告仍存在。未做独立实现复算模型分数或独立任务验证。

`experiment-design` 用于提前固定共同起点、对照及停止规则。该实验是基础优化比较，不包含行动后果描述消融，不证明闭环收益或任何 PC gate 已通过。

[模型计划](../le-wm-main/results/pc_tworoom_joint_random_dev_20261107_codex_plan.json) ·
[全部候选](../le-wm-main/results/pc_tworoom_joint_random_dev_20261107_codex_pool.npz) ·
[物理轨迹](../le-wm-main/results/pc_tworoom_joint_random_dev_20261107_codex_physical.json) ·
[运行代码](../le-wm-main/experiments/pc_tworoom_joint_random_dev_20261107_codex.py) ·
[审计测试](../le-wm-main/tests/test_pc_tworoom_joint_random_dev_20261107_codex.py)
