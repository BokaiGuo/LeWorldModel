# TwoRoom 生成式动作见证任务：开发结果

按[事前协议](PC_TWOROOM_GENERATED_WITNESS_DEV_PROTOCOL_20261107_CODEX.md)，预定 40 个环境 reset 与随机动作序列，以每条 25 步动作的实际后继作为目标，未按初始距离、是否越墙或控制结果筛选。

- **40/40** 条见证从相同 reset 重放后到达生成的目标并满足本地成功规则；其中 **2/40** 个初态已满足成功距离。
- 仅 **2/40** 对初态—目标位于墙的不同侧。这是按端点位置描述，不代表其余任务必然没有碰撞，也不认证非贪心机制。
- RGB CEM300 真实反馈为 **40/40** 成功；排除初态成功的两个任务为 **38/38**。共 86 次规划、25,800 条候选评分、344 个实际环境步。规划从当前及目标 RGB、合法旧尾和固定动作 normalizer 出发，评价端状态只用于目标构造、reset 与记录。
- 对本地 HDF5 每个长度至少 26 的 episode，比较其初态及第 25 步 agent 位置与本组状态对，未发现逐值相同的对。该比较不覆盖其他时间偏移、近邻、全部图像或外部训练集，不能证明训练外或完整独立性。

**决策：此无筛选随机见证分布在本面板被 CEM300 全部解决，且跨墙比例很低，不适合用于检验方法的绕障优势或 PC2 机制。** 它保留为可解性和数据管线开发资产，不替代标准性能集，不用这 40 个结果挑选有利子集进入确认。

逐条重放了 **1,000 个见证步和 344 个 CEM 实际步**，核对 reset、图像、目标图、动作、距离及成功；该审计使用相同环境，没有重新计算 GPU 候选分数或独立认证模拟器。种子和动作块测试 **5 项通过**。Gymnasium 观测和动作转换警告仍存在；模型训练成员、normalizer 与 checkpoint 训练合同以及历史开发独立性仍未知。PC0–PC5 未改判。

[公开 RGB 任务](../le-wm-main/results/pc_tworoom_generated_witness_public_20261107_codex.npz) · [见证及状态（仅评估）](../le-wm-main/results/pc_tworoom_generated_witness_truth_20261107_codex.npz) · [真实参考轨迹](../le-wm-main/results/pc_tworoom_generated_witness_reference_20261107_codex.json) · [CEM 反馈](../le-wm-main/results/pc_tworoom_generated_witness_cem300_20261107_codex.json) · [重放审计](../le-wm-main/results/pc_tworoom_generated_witness_audit_20261107_codex.json)
