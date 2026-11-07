# TwoRoom 条件局部尾部：开发结果

按[事前协议](PC_TWOROOM_CONDITIONAL_LOCAL_TAIL_PROTOCOL_20261107_CODEX.md)，
在 40 个已曝光任务上，使用冻结 RGB LeWM 给每个保留前缀相同的局部
尾部搜索机会；封存选择后，分别实际执行三臂各 40 条链。

| 保留规则 | 25 步内曾成功 | 末步成功 | 选中局部链 | 相对原随机 B 臂独胜 / 独负 |
| --- | ---: | ---: | ---: | ---: |
| 普通 lookahead | 38/40 | 38/40 | 6/40 | 0 / 0 |
| 质量约束描述 | 38/40 | 38/40 | 6/40 | 0 / 0 |
| 质量约束身份打乱 | 38/40 | 38/40 | 6/40 | 0 / 0 |

三臂在 40/40 任务最终选中逐字节相同的链；6 条局部链均来自三臂
共同保留的第一前缀。局部搜索确实在这 6 根产生比共同 incumbent
更低的模型代价，但没有带来事前指标上的成功增量，更没有显示描述
相对 lookahead 或身份打乱的额外贡献。描述臂相对两对照独胜／独负
均为 0/0，未达到预定开发出口；**停止扩大这一固定局部构造**。

每臂逻辑成本为 691 条完整链评分，包括本次复评共同最佳链一次；
三臂联合实际新增 GPU 评分为 984 条局部候选及 40 次共同链复评。
该数不含模型加载、编码、执行、反馈的完整计时，不支持成本优势。
本实验复用已有候选及历史原始评分，固定局部尾部标准差为 0.2，
不否定其他动作支持或搜索设计；它不是充分调优的 CEM 比较，也不是
PC2 的同链机制实验或 PC3–PC5 确认。

## 核验与边界

- 三项测试检查条件前缀固定、动作合法与确定性、不同臂候选共享、
  同分时保留共同链以及不可用候选拒绝。
- 审计重建了全部 40 根候选和三臂选择；对预定根 18、38、49 在 GPU
  重算全部实际查询候选，最大代价差为 0。审计复用同一生成函数、
  checkpoint 与模型实现，没有独立重放全部物理轨迹或证明模拟器正确。
- 模型只接收 RGB、目标图及合法动作；物理状态只用于评价端初始化与
  执行。环境发出 reset 观测类型及动作转换警告，保留为接口认证限制。
- `experiment-design` 用于事前固定对照、预算及停止规则。所有任务和
  原结果已曝光；结果不构成独立盲测，不更改任何 gate。

[计划与选择](../le-wm-main/results/pc_tworoom_conditional_local_tail_20261107_codex_plan.json) ·
[候选池](../le-wm-main/results/pc_tworoom_conditional_local_tail_20261107_codex_pool.npz) ·
[真实轨迹](../le-wm-main/results/pc_tworoom_conditional_local_tail_20261107_codex_physical.json) ·
[汇总](../le-wm-main/results/pc_tworoom_conditional_local_tail_20261107_codex_result.json) ·
[GPU 审计](../le-wm-main/results/pc_conditional_local_tail_audit_20261107_codex.json) ·
[实现](../le-wm-main/experiments/pc_tworoom_conditional_local_tail_20261107_codex.py)。
