# 分离求解器 seed 与有界 CEM：PushT 开发结果

任务采样 seed 固定为 42，物理 reset seed 固定为 771001，仅改变
求解器 seed；各臂 50 个任务首次动作前的完整物理快照逐项相同。
按 `PC_PUSHT_SPLIT_SEED_ACTION_AUDIT_PROTOCOL_20261105_CODEX.md`
和 `PC_PUSHT_BOUNDED_CEM_DEV_PROTOCOL_20261105_CODEX.md` 执行：

| 求解器 seed | 原生成功 / 50 | 原生实际越界动作 | 有界成功 / 50 | 有界实际越界动作 | 有界候选投影比例 |
|---|---:|---:|---:|---:|---:|
| 42 | 47 | 0 / 1466 步 | 47 | 0 / 1455 步 | 1779 / 189000（0.94%） |
| 1042 | 43 | 1 / 1566 步 | 43 | 0 / 1590 步 | 2176 / 201000（1.08%） |
| 2042 | 43 | 1 / 1608 步 | 42 | 0 / 1605 步 | 1903 / 207000（0.92%） |

原生两条越界动作的最大绝对值分别约 1.7533 和 1.1059，环境
声明动作域为 [-1,1]。原生数据未裁剪、未删例；有界 CEM 将
候选在模型评分之前投影到按相同训练集 StandardScaler 换算的
物理域，模型分数与精英分布均使用投影后的动作。实际执行
trace 中有界三臂共 4650 步未越界；单元测试验证所有被评分
候选及返回均值的逆变换合法。

这是同一开发任务面板的三次求解器随机流，不是三个独立任务集。
计分候选和规划轮次因闭环执行不同而不同，表中不宣称等成本、
优效或等效；有界和原生的搜索分布也不同。固定 reset 与初始
快照配对不证明恢复了数据集原始完整物理状态，更不完成 checkpoint、
上游来源或任务参考动作合法性的认证。

分析命令分别为
`python -m experiments.pc_split_seed_action_audit_analysis_20261105_codex`
和 `python -m experiments.pc_bounded_cem_analysis_20261105_codex`；
原始指标、逐步 trace 与投影计数均在对应 `results/pc_split_seed_action_audit_*`
和 `results/pc_bounded_cem_*` 中。未改变旧结果或 PC gate。
