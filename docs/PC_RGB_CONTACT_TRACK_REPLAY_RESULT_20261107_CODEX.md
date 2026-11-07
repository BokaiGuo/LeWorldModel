# RGB-only 接触面板：权限入口重放

按 `PC_RGB_CONTACT_TRACK_REPLAY_PROTOCOL_20261107_CODEX.md`，从既有的 16 根几何合法 PushT 面板提取当前图像、目标图像和动作，在独立 GPU 评分进程通过 `primary_rgb` 入口重放。评分进程读取的封存输入不含真实后继或物理评价标签；过程隔离不是操作系统级沙箱。

全部 **192/192** 候选的模型分数在预定容差内与旧分数一致，最大绝对差 `6.103515625e-05`；全部 **64/64** 三动作条件的 argmin 动作相同。冻结模型指纹一致；192 条运行收据均只有 `pixels`、`goal`、历史 `action` 和候选 `candidate_actions` 字段。后置审计核对了全部初始图像、目标图像及动作与原始记录一致。

该结果验证了权限封装迁移在该面板上保留旧预测，不是行动后果可信、后果描述优于 ordinary lookahead、实际规划成功或闭环收益。原面板已有物理轨迹并经此前审计，但本次没有重新执行环境；其几何 roots 属开发数据，训练外独立性未知。历史输入沿用显式空历史占位，不能外推多帧历史。收据中的 `visual_closed_loop` 是轨道的预设声明范围，不是本次闭环已经执行；`scientific_status` 为 `not_evaluated`。

运行器为 `le-wm-main/experiments/pc_rgb_contact_track_replay_20261107_codex.py`，封存输入、192 条分数和审计见 `le-wm-main/results/pc_rgb_contact_track_replay_20261107_codex*`。权限和对比相关单元测试 **18 项通过**；其合成测试不是规划性能证据。PC0–PC5 不改判。
