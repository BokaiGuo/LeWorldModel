# 后继探针保留准则：封存计划开发结果

按 `PC_LOOKAHEAD_VS_RIDGE_PROSPECTIVE_DEV_PROTOCOL_20261104_CODEX.md`，在原窄束面板的 16 个已暴露任务上，用配对种子分别运行局部岭响应值（ridge）与相同探针中最小完整链终距（probe_min）。全部 32 条计划在干预前写盘并审计，全部 32 条最终链干预完成；原面板未改动。

| 指标 | ridge | probe_min |
|---|---:|---:|
| 终距 < 0.05 | 3/16 | 3/16 |
| 平均最终终距 | 0.23820 | 0.21414 |
| 平均规划 prediction-token | 2879.25 | 2879.25 |
| 规划 token 范围 | 2878–2880 | 2878–2880 |

5/16 对任务的层级保留索引不同，2/16 对最终链不同；这两根的 probe_min 终距更低，其余 14 根相同。root 0 为 0.50028 → 0.11532，root 3 为 0.500408 → 0.500375。一个较大的开发样本变化驱动均值差；保留索引跨两臂后续不同候选库时不可简单视为相同动作。两臂在比较中都计算岭值并使用相同探针，不能以此推断跳过响应计算带来的资源收益；相同 token 不是等墙钟、GPU 或全系统成本。平均规划墙钟分别约 0.163、0.155 秒，包含记录开销，样本小且无隔离性能测试。

每臂对 selected、zero、greedy、random 前缀记录六条共同尾部和三次各 481 完整链评分的重规划分布，以及 20 坐标、两个尺度的实际尾部响应。每臂 16/16 干预完成；selected、zero、greedy 三前缀在两个尺度的所有坐标均保持分支稳定的各为 3/16。该全尾部测量和旧面板前五步测量不同，数量不应直接当作同一指标前后对比。平均干预计费分别为 18334.5 和 18329.25 prediction-token，不计另行执行的碰撞分支重放；诊断费用未混入规划成本。

计划账本经同环境另一几何实现复算；独立审计脚本另检查干预账本、测量动作及输出与调用日志绑定、选中链、实际差分及分支重建。`interventions_audit_v2.json` 通过；这不是跨环境独立复现。数据位于 `le-wm-main/results/pc_lookahead_ridge_prospective_dev_20261104_codex/`。

两个都为 3/16 的开发结果、有限探针和已暴露任务不证明 probe_min 是稳定改进，亦不证明响应机制或 PC2/PC3 通过。此开发协议不包含 lookahead 对照前缀，也不定义新的机制通过阈值；不能冒充完整 PC2。未覆盖独立数据、强基线、真实 RGB、完整资源或来源认证。

复现需新输出路径避免覆盖：在 `le-wm-main` 依次运行 `.venv/bin/python -m experiments.pc_lookahead_ridge_prospective_dev plan`、`audit_plans`、`intervene`，再运行 `.venv/bin/python -m experiments.pc_lookahead_ridge_intervention_audit`。入口当前固定输出路径并拒绝覆盖；变更输出路径须另存代码并重新封存依赖。
