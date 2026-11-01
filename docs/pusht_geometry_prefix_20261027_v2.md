# P1 future-response-aware prefix generation：固定协议

此实验针对实现覆盖缺口，不预判旧负结果来自理论：现有10个prefix只来自当前
Jacobian的两条谱方向及DLS；虽然逐候选重算后续K，却没有用K(B)随prefix的变化
来生成新的prefix。旧detach在固定库评分中正确，不得仅删除detach冒充修复。

新两臂geometry / stopK：完全相同原四个已用根、冻结LeWM、goal0/Q、完整5×2
动作包、atanh尺度、damping=.002和半径sqrt(5)*.75。选择仍为stage+bounded
ridge值，beta=1，不调阈值，不用物理结果重选，不训练。

从zero与half rootDLS两起点，各固定两次更新。每次先计算完整中心K，然后在
十个prefix参数的正负.01位置计算stage、residual和K；geometry微分全部变化，
stopK仅在微分评分时固定中心K，仍重算residual/stage并重新解相同约束后缀问题。
下一迭代中心K刷新。两臂调用同样数量的真实模型/Jacobian，每次更新沿负梯度
单位方向走cap/4；起点≤cap/2，第二轮中心≤3cap/4，全部probe在球内，终点≤cap。
stopK也计算probe K但不用于其导数，避免暗中把不同Jacobian预算说成匹配。

原10个候选不变，每臂加入两条优化终点，共12条，统一用真实own-state K评分。
所有96候选独立完整模型重放后封存，再交真实物理consumer；非线性终态仍只是
诊断，不替换原score。原10候选是共享参考，额外搜索算力不能冒称零开销优势。
另在zero处计算.01/.005两组导数用于数值敏感性检查，.005不用于生成/挑选步长。

不声称该优化已发现非贪心、物理能力改变或任务收益。任何旧保护中止仍保留；
科学P1/P2准入不由此脚本开启。四根已用，不能作为独立确认集。

主机CUDA命令（路径必须不存在，禁止CPU神经fallback）：

```bash
.venv/bin/python -m experiments.spectral_reachability.pusht_geometry_prefix_20261027_v2 --compilation results/pusht_d2p_metric_program_preflight4_v1_20261026_cuda/report.json --policy /home/gbk/.stable_worldmodel/checkpoints/pusht/lewm --output results/pusht_geometry_prefix_20261027_v2_cuda_v1
```

单元测试是非神经解析函数核对，包含K变化的envelope项、constant-K控制、保留
残差导数、中心刷新、坐标身份和边界；不是模型预测或物理效能的替代证据。
