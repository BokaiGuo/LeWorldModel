# PushT 条件 rollout 实现等价检查

为检查上轮负结果是否由手工状态／动作窗口实现造成，在四个已暴露开发任务上比较冻结 PushT LeWM 的原生 `rollout` 与诊断用 `conditional_rollout`。两者使用相同模型、初始编码和归一化动作；原生路径仅用已有编码替代图像编码器输出，其他预测代码保持原样。没有读取物理成功标签。

首轮每任务捕获 30、200、200、200、2 条候选，比较各链全部五个预测时刻；共同前缀重评分每任务比较 301 条链、各三个剩余时刻。

| 任务 | 首轮与捕获预测最大表示差 | 尾部与原生预测最大表示差 | 尾部目标代价最大差 | 封存选择 / 原生选择 |
|---|---:|---:|---:|---:|
| 0 | 1.67e-6 | 1.67e-6 | 3.05e-5 | 300 / 300 |
| 1 | 4.53e-6 | 1.79e-6 | 1.22e-4 | 255 / 255 |
| 2 | 1.67e-6 | 1.19e-6 | 3.05e-5 | 250 / 250 |
| 3 | 1.85e-6 | 1.43e-6 | 4.58e-5 | 295 / 295 |

本次重算的手工尾部分数与先前封存分数逐值一致；原生 rollout 与之有微小浮点差，但四任务选择一致。模型权重指纹与封存记录一致、运行前后未变。人工窗口、原生时钟与候选评价相关的测试共 **7 项通过**。

这排除了这些固定候选上“自定义窗口实现相较当前原生 rollout 造成明显差异或改变选择”的解释；不证明官方实现本身、图像预处理、动作物理执行或预测动力学均正确，也不证明其他候选与设备上的等价。初始真实图像编码来自旧回放，本次没有重跑物理过程；不改变任何 gate。

结果和运行器：[JSON](/home/gbk/code/LeWorldModel/le-wm-main/results/pc_pusht_conditional_rollout_parity_20261107_codex.json)、[脚本](/home/gbk/code/LeWorldModel/le-wm-main/experiments/pc_pusht_conditional_rollout_parity_20261107_codex.py)、[协议](/home/gbk/code/LeWorldModel/docs/PC_PUSHT_CONDITIONAL_ROLLOUT_PARITY_PROTOCOL_20261107_CODEX.md)。
