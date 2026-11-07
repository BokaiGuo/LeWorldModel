# TwoRoom 时序保留固定池开发结果

按[事前协议](PC_RGB_TEMPORAL_RETENTION_FIXED_POOL_PROTOCOL_20261107_CODEX.md)，在已封存的 119 个共同历史决策池、40 个已曝光或独立性未知的任务上，固定相同三个前缀、描述探针和后续尾部，只将普通 lookahead 保留依据从末帧代价改为时序最低代价。先封存模型分数和保留选择，再读取六条固定尾链的物理结果。

| 指标 | 原末帧保留 | 时序保留 |
| --- | ---: | ---: |
| 119 个决策中，保留的尾链有曾成功链 | 64 | 65 |
| 119 个决策中，决策初态及前缀未成功、尾段才成功 | 35 | 36 |
| 40 个任务中至少一次有上述尾段成功覆盖 | 22 | 24 |
| 排除初态成功后的 37 个任务，上述覆盖 | 22 | 24 |

排除初态成功的任务中，时序保留独有覆盖 2 个、原版独有覆盖 0 个。由于事前出口要求至少 3 个时序独有覆盖，**本配置未达到开发出口，不扩大至端到端确认**。这不证明时序保留在其他候选和任务上无价值。

119 次保留中，按槽位顺序比较有 39 次不同；忽略槽位顺序后，真正保留不同前缀集合的是 **27 次**。独立算术审计逐条重算全部保留选择与物理覆盖，相关测试 **6 项通过**。原结果文件中的 `changed` 记录的是前者，不应误读为 39 次集合变化。

物理结果用于事后覆盖评价，其中第三个前缀的两个尾链原算法未必评分；不能把这个池当成免费部署查询。多个决策共享任务，覆盖也不是实际最终选择、真实闭环成功或后果描述的因果贡献。这是使用已曝光结果提出的开发假设；本实验没有新增模型调用或环境执行，不改变 PC0–PC5。

[模型选择封存](../le-wm-main/results/pc_tworoom_temporal_retention_fixed_pool_plan_20261107_codex.json) · [物理覆盖结果](../le-wm-main/results/pc_tworoom_temporal_retention_fixed_pool_result_20261107_codex.json) · [运行器](../le-wm-main/experiments/pc_tworoom_temporal_retention_fixed_pool_20261107_codex.py) · [独立算术审计](../le-wm-main/experiments/pc_tworoom_temporal_retention_fixed_pool_audit_20261107_codex.py)
