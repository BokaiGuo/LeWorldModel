# PushT 共同前缀物理身份检查

对四个已暴露开发任务的每个任务 301 条共同前缀链，核对封存动作哈希、前两个五步动作包的逐值相同、先前回放的第 5 和第 10 环境步记录状态，以及原选链实际派发轨迹。物理记录哈希与上次封存评估一致。归一化动作按原始数据拟合的相同 scaler 反变换。

四任务结果均相同：

- 301 条链与原选链在第 5、10 步记录的七维状态最大绝对差均为 0；两时刻均无候选超出 1e-4。
- 原选链回放状态与实际派发 trace 对应状态最大绝对差均为 0。
- 原选链前十个反归一化动作与 trace 实际动作最大绝对差为 0，无动作步超出 1e-6。

因此，先前共同前缀重评分所使用的动作时间和已记录状态在这四任务上相互吻合。检查使用已保存回放，不是新的独立物理执行；七维观测不是完整模拟器状态，不足以排除未记录的速度、接触缓存或其他内部状态差异，也不认证像素预处理、动力学预测和评分质量。物理状态仅用于事后核验，不作为规划器输入。本结果不改变 PC0–PC5。

本检查与 rollout 和评估回归测试合计 **7 项通过**。原始记录：[JSON](/home/gbk/code/LeWorldModel/le-wm-main/results/pc_pusht_shared_prefix_physical_identity_20261107_codex.json)；[脚本](/home/gbk/code/LeWorldModel/le-wm-main/experiments/pc_pusht_shared_prefix_physical_identity_20261107_codex.py)；[协议](/home/gbk/code/LeWorldModel/docs/PC_PUSHT_SHARED_PREFIX_PHYSICAL_IDENTITY_PROTOCOL_20261107_CODEX.md)。
