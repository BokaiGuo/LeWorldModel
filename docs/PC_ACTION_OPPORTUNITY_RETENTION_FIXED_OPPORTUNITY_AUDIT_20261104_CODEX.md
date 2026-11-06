# Action-opportunity fixed-opportunity retention audit

**对照名称修正：** 本文 `probe_min` 实际选择的是最大相对改善，不是绝对探针代价最小，不能作为方案中 E 臂的实现。后续的统一探针、绝对代价对照见 [修正审计](PC_ACTION_OPPORTUNITY_RETENTION_ABSOLUTE_PROBE_AUDIT_20261104_CODEX.md)。本文件和原结果保留，不追溯覆盖。

日期：2026-11-04。性质：对已封存候选后缀质量结果的后验固定机会诊断；不重跑模型、不改变候选生成、不接入正式 planner，不是 PC2、PC3 或独立确认结果。

入口：

`le-wm-main/experiments/pc_action_opportunity_retention_fixed_opportunity_audit_20261104_codex.py`

结果：

`le-wm-main/results/pc_action_opportunity_retention_fixed_opportunity_audit_20261104_codex.json`

## 1. 目的与比较

该审计固定候选池和共同的有限后缀求解机会，只改变第二个保留槽位的后验选择规则。每个 root 在 depth 0、1 各形成一个组；保留槽位数从原候选质量记录读取。即时 greedy 槽始终保留。

比较两类候选来源：

- `random_direction`：非谱方向候选；
- `spectral`：谱方向候选。

两个来源使用的描述库不同：`spectral` 使用固定探针，`random_direction` 使用随机探针 v2。因此只在同一来源内比较两种保留规则；跨来源差异不能归因于生成器或谱方向。这也不是 A–E 的完整矩阵。

比较两种保留规则：

- `probe_min`：第二槽按 A 探针的最大改善选择；
- `effect_complement`：第二槽按标准化完整 A 改善向量的最远点选择。

`effect_complement` 只使用已记录的 A 描述；它不是独立验证，也不代表后果描述已经有物理或视觉可信性。

## 2. 绑定与审计修复

首次运行得到 0 组，不应解释为没有候选。原因是输入 schema 的两个实现细节：

1. 候选质量文件在文件级记录完成状态，单条候选行没有 `status` 字段；
2. 描述结果文件同时包含多个 `arm`，初始索引只按 `(root, depth, index)` 建立，导致不同 arm 的行互相覆盖。

审计代码随后：

- 对无单行 `status` 的候选接受文件级完成状态；
- 按请求的 `arm` 过滤描述行；
- 修正 farthest-point 选择的 pairwise 距离计算；
- 在即时分数不一致时 fail closed。

因此，重跑后的 128 行结果才是有效的绑定结果；首次空结果仅记录为实现诊断，输出文件已由重跑结果覆盖，不作为实验结果。

## 3. 结果

| 候选来源 | 保留规则 | 组数 | 成功 | 平均选中最佳终距 | 平均候选池最佳终距 | 平均距池最优差距 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| random_direction | probe_min | 32 | 0/32 | 0.50240 | 0.50050 | 0.00190 |
| random_direction | effect_complement | 32 | 0/32 | 0.50204 | 0.50050 | 0.00155 |
| spectral | probe_min | 32 | 0/32 | 0.49895 | 0.47490 | 0.02405 |
| spectral | effect_complement | 32 | 0/32 | 0.50191 | 0.47490 | 0.02701 |

组是 root×depth 聚合单位，不是独立 episode。depth 2 不在本审计中，因为 15 步时域下它已经是完整链，没有剩余尾部。

## 4. 结论边界

本面板没有提供继续采用 `effect_complement` 的开发性证据：

- 两种规则在该固定机会上的成功率均为 0/32；
- 候选池在该有限求解机会下也均未检出成功，求解器漏检时不能据此断言前缀不可行；
- `random_direction` 上两者接近，`effect_complement` 的平均距池最优差距略小；
- `spectral` 上 `effect_complement` 明显差于 `probe_min`；
- 结果只说明在当前候选库和共同后缀机会下，向量互补保留没有稳定提升选中候选质量；
- 不能把候选池中存在更好后缀解释为保留规则能够发现它；
- 不能据此否定行动后果描述在其他生成器、后缀库或完整成本下的价值。

因此不把 `effect_complement` 直接接入正式 retention，也不把本结果包装成端到端 A–E 消融。下一步若继续研究，应先补上真正独立的验证后缀，并把固定机会诊断与完整成本比较分开；若没有预先指定的候选质量改善假设，不继续围绕该保留规则扩大实验。

## 5. 可复现信息

审计输入由脚本写入结果 JSON 的 `source` 字段：两个描述文件和每个 arm 的 root 00 候选质量文件带有 SHA-256；其余 30 个候选质量文件尚未逐一哈希，因此目前不是完整冻结依赖清单。运行使用项目环境：

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
.venv/bin/python -m experiments.pc_action_opportunity_retention_fixed_opportunity_audit_20261104_codex
```

该命令只读取已封存 JSON 并生成审计 JSON，不重跑动力学或模型预测。
