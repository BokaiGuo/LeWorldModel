# TwoRoom：保留完整原链的 RGB 反馈搜索冒烟结果

按预定协议，在四个已曝光任务上，用冻结 LeWM、当前 RGB 和目标 RGB 真实反馈执行三个策略：块动作搜索基线、普通 lookahead、中心化向量保留。每次规划均实际评分 680 条合法动作链，旧尾从新观测重新评分，全部已评分完整链保留到最终竞争；每次最多派发 5 步，任务最多 25 步。每个策略独立重置及执行，不使用其他策略的实际后继。

| 策略 | 曾成功 | 规划次数 | 候选评分 | 模型包步 | 环境步 | 同步规划耗时合计 |
|---|---:|---:|---:|---:|---:|---:|
| 块搜索基线 | 4/4 | 19 | 12,920 | 40,120 | 88 | 3.965 s |
| lookahead | 4/4 | 19 | 12,920 | 40,120 | 88 | 3.760 s |
| 向量保留 | 4/4 | 19 | 12,920 | 40,120 | 88 | 3.765 s |

每臂均有 18/19 次从最初 600 条链中选择，1 次从额外 80 条中选择；旧尾分别获选 5、6、6 次。lookahead 和向量臂在这四根的选择与物理结果一致。等候选评分和包步不等价于严格的总计算成本公平：这里的同步耗时仅是规划代码边界，四任务、单次顺序运行，没有独立重复或完整系统成本比较。

新增选择、动作范围、旧尾和计数测试 **16 项通过**。独立轨迹审计逐项核验了 57 次决策的 680 条评分数量、25 次模型前向收据、动作哈希、派发前缀、旧尾获选身份及物理终止时钟。随后在同一硬件、模型和环境中重置并沿实际动作重建 RGB 观测，重新运行全部 57 次规划：候选评分最大绝对差为 **0**，保留索引、最终动作及逐步真实距离均一致。重复运行是实现可复现检查，不是独立模型或任务验证。

物理状态只由评价端用于环境重置和结果记录，不作为评分器输入；规划调用接收 RGB 模型上下文和合法动作。但当前执行器在同一进程中持有评价状态，不能据此声称已通过严格进程级权限隔离。环境仍提示 Gym reset 返回类型和动作转换警告。

这次工作把“将已评分原链留在竞争中”实现到真实重规划路径，补上上一轮离线组合的工程缺口；它没有在本面板证明描述增量，也不证明完整成本优势或 PC0–PC5 通过。四个根已曝光且三臂全成功，无法判别泛化或性能差异。按协议不扩大该配置争取阳性。

[协议](/home/gbk/code/LeWorldModel/docs/PC_TWOROOM_INCUMBENT_RGB_FEEDBACK_PROTOCOL_20261107_CODEX.md) · [搜索实现](/home/gbk/code/LeWorldModel/le-wm-main/experiments/pc_rgb_block_incumbent_search_20261107_codex.py) · [完整轨迹](/home/gbk/code/LeWorldModel/le-wm-main/results/pc_tworoom_incumbent_rgb_feedback_20261107_codex.json) · [轨迹审计](/home/gbk/code/LeWorldModel/le-wm-main/results/pc_tworoom_incumbent_rgb_feedback_audit_20261107_codex.json) · [重算记录](/home/gbk/code/LeWorldModel/le-wm-main/results/pc_tworoom_incumbent_rgb_replay_20261107_codex.json)。
