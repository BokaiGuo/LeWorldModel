# Action-opportunity random-probe Stage-1b 结果（2026-11-04）

结果文件：

[result.json](/home/gbk/code/LeWorldModel/le-wm-main/results/pc_action_opportunity_random_probe_dev_20261104_codex/result.json)

## 结果

固定原候选库和封存 CEM 目标，对 2476 个候选使用独立的随机 probe A/B：

- A/B 最大作用改善的 Spearman 相关：**0.6833**；
- A 与 `-best_score` 的边际相关：**-0.2667**；
- immediate score 与 `-best_score` 的相关：**0.6382**；
- 控制 immediate score 后，A 描述与 `-best_score` 的 rank-residual partial correlation：
  **0.2994**；
- A top-10% 候选平均 `best_score`：0.6446；全体平均：0.5898；
- A top-10% success rate：0/248；全体 success rate：1/2476。

分层 partial correlation：

| arm | partial rank correlation |
|---|---:|
| spectral | 0.3025 |
| random_direction | 0.2730 |
| diverse | 0.3035 |

## 解释

随机 probe 的 A/B 一致性明显高于上一轮常值/交替 probe，说明它们测到了可重复的
动作敏感性。其边际相关为负，主要因为“相对 zero tail 的最大改善”也会随当前候选
较差而增大；不能直接把这个边际结果当作无效。

控制当前终距后出现一致的正向残余关系，说明该描述可能包含当前终距之外的后续质量
信息。这只是开发线索，仍不能证明因果机制或自动搜索增量，原因包括：

- 封存 `best_score` 来自既有开发 CEM，而非新预注册的独立重规划目标；
- partial rank correlation 是诊断统计，不是冻结后的确认统计合同；
- 作用描述和目标都来自同一 exact predictor，不能认证视觉或真实物理响应；
- 尚未做等槽位、等总成本的保留对照。

## 决策

本结果足以支持一个新的、明确的 Stage-2 开发假设：

> 在控制当前终距后，随机后缀作用描述能否提高固定槽位候选的后续可实现质量。

但暂不接入正式 planner。下一步若继续，必须预注册 residual-based retention、使用
独立重规划随机种子和独立后缀目标，并将 probe 计算计费；同时保留 response、diverse、
greedy 和 global CEM 对照。若该等成本比较不复现正向残余关系，则交付有界负结论。
