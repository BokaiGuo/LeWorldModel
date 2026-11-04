# v3 选中尾部附近事前机制开发结果

按[事前协议](PC_RESPONSE_TEMPORAL_V3_TAIL_LOCAL_MECHANISM_PROTOCOL_20261104_CODEX.md)，
24 个新网格根上 72 条规划在干预前封存并逐批重算；对全部根进行
相同共同尾部、自行规划与双尺度响应干预。干预通过同代码确定性
重建。14/24 根满足非即时条件，**1/24 根满足预定 selected 对全部
四种对照的收益、双尺度响应差异和阻挡分支稳定性交集**。

该开发见证为根 20，起点 `[-0.5,0.5,-0.05]`，目标 `[0.5,-0.25]`。
它是 v3 事前自动选择并精修后的同一开放环链；链的 selected
前缀使距离从 1.25 增至 1.62244。选中尾部无需裁剪：

| 前缀 | 前缀后零尾终距 | 同一选中尾部终距 | 120 次评价自行规划终距 |
| --- | ---: | ---: | ---: |
| selected | 1.62244 | 0.10883 | 0.37898 |
| zero | 1.25000 | 2.76302 | 0.54956 |
| same-bank greedy | 0.90365 | 1.27767 | 0.50277 |
| matched-random bridge | 1.35864 | 2.19372 | 0.56921 |
| ordinary lookahead | 0.86854 | 2.15386 | 0.50362 |

selected 对 zero 的终点响应差在 `.01/.05` 两尺度分别为
6.77983/6.77339，对 greedy 为 2.68349/2.68094；五类前缀的
中心与全部扰动在各自尺度内保持同一阻挡序列。实际选中尾部
相对各前缀自身零尾的变化，以及各自同预算规划，也都满足协议。
独立审计从封存选中链重新计算全部共同尾部、分支和响应，且用
另一段 CEM 实现重建全部五个自行规划结果；它仍共享同一精确
动力学实现与 numpy，不是独立模型或数据复现。

这是一个有界自动程序与非即时响应机制开发见证，**不是 PC2 全面
通过或 PC3 优势确认**：新网格只排除了与两个已引用面板的完全相同
任务，历史独立性未知；自行规划只有一套 120 次评价配置，不能称
对照搜索充分；响应差异与收益的关联也不单独证明中介因果。
物理总终距 0.10883 不满足此前开发使用的 0.05 成功阈值。没有把
这条开放环链与后续重规划或 RGB 轨迹拼接。此前零名义网格 0/24
的负结果保持不变。

- [规划](../le-wm-main/results/pc_response_temporal_v3_tail_local_mechanism_20261104_codex_plans.json)，SHA-256 `6d54c346dd34b2e3d9ec79e022ce44c1291ed76df6bf1160676c41edbafb2d7a`
- [规划逐批审计](../le-wm-main/results/pc_response_temporal_v3_tail_local_mechanism_20261104_codex_audit.json)，SHA-256 `c82c7b3f2e217317f3f741273d711f3894603b738ad794fe29a7a032f2a622d8`
- [完整干预](../le-wm-main/results/pc_response_temporal_v3_tail_local_mechanism_20261104_codex_intervention.json)，SHA-256 `32e3149b3e750f29b9f7d051dc56858554d5c1430a7ba8d3cc275fa613839501`
- [确定性重建](../le-wm-main/results/pc_response_temporal_v3_tail_local_mechanism_20261104_codex_intervention_rebuild.json)，SHA-256 `9270907871905398bc48d6994a430310c1e9b57becba4cdfdba7136f21749df7`
- [见证独立实现审计](../le-wm-main/results/pc_response_temporal_v3_tail_local_mechanism_20261104_codex_witness_independent_audit.json)，SHA-256 `4ef251296b6145117ba925ce7569759757a7fd4f65ead1e7bbd88722b72d0b3e`
