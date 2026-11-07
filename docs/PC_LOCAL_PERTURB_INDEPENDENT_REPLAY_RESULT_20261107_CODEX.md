# 小扰动实验的 GPU 与物理重放核验

按[核验协议](PC_LOCAL_PERTURB_INDEPENDENT_REPLAY_PROTOCOL_20261107_CODEX.md)，
独立从父链和种子重建全部 320 条扰动，与封存动作逐元素一致。
在 GPU 上用统一的 32 条分批重算每根 64 条原链与 32 条扰动链，共
960 条：与原封存评分最大差为 **0**；10 根的父链、直接最低分、
扰动均值和打乱臂选择全部一致。

对直接评分实际选中的全部 8 条新扰动链重新执行 25 个物理步：距离
最大差为 **0**，成功标记无差异。另将 96 候选直接选择与原 64 候选
最低分的已记录成功结果配对，均为 8/10，独胜和独负均为 0；这是
事后比较，不能当作前瞻实验。新增与原实验相关测试 **4 项通过**。

这些检查削弱了“本次负结果由扰动重建、不同批次评分或新链物理重放
错误造成”的解释，但不排除其他实现问题或其他动作支持下的不同结果。
两个运行共用冻结模型、数据和同一物理后端，任务已经曝光；不构成独立
确认、RGB 闭环优势或 gate 通过。Gymnasium reset 类型和动作转换警告
仍存在。`code-debugging` 用于针对跨批评分和动作身份设计复核，并未
预设负结果必然由代码错误造成。

[逐根核验记录](../le-wm-main/results/pc_local_perturb_independent_replay_20261107_codex.json) ·
[核验实现](../le-wm-main/experiments/pc_local_perturb_independent_replay_20261107_codex.py) ·
[测试](../le-wm-main/tests/test_pc_local_perturb_independent_replay_20261107_codex.py) ·
[原实验](PC_TWOROOM_LOCAL_PERTURB_STABILITY_RESULT_20261107_CODEX.md)。
