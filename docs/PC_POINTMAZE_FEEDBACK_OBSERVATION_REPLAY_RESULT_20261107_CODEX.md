# PointMaze 反馈观测与评分重放结果

按[重放协议](PC_POINTMAZE_FEEDBACK_OBSERVATION_REPLAY_PROTOCOL_20261107_CODEX.md)，从八根原始初态连续执行已封存的 25 步动作，保存每根四次重规划前的重放 RGB、proprio 和图像哈希。32 次观测的 proprio 均在 `1e-6` 容差内匹配原反馈记录，最终状态也匹配。

使用这些重放观察、原目标、冻结模型和原候选，在 GPU 上重算 **1536 个候选评分**：

- 与原记录的最大绝对分数差为 **0**。
- **32/32** 次最终最优候选一致。
- 三轮共 **96/96** 组 elite 集合一致。
- 相关单元测试 **2 项通过**；模型运行前后指纹一致。

原始观察及哈希位于 `le-wm-main/results/pc_pointmaze_feedback_observation_replay_20261107_codex_observations.npz` 和同名前缀的 `observations.json`；逐候选复算分数与汇总在同名前缀的 `scores.json`。

**证据边界：**原始反馈运行没有保存现场中间帧或其哈希，因而不能直接证明原图与重放图逐像素一致；完整评分复现是决策可复现的有力核对，不是原始图像身份的证明。该实验未改变动作或重新规划，仍是已曝光任务上的 state-assisted 事后核验，不是 RGB-only、等成本收益或独立确认，也不改变 PC0–PC5 的判定。
