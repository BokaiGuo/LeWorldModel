# v3 同预算多起点 CEM 开发对照结果

按[事前协议](PC_RESPONSE_TEMPORAL_V3_MULTISTART_CEM_E0_PROTOCOL_20261104_CODEX.md)，
在 24 个已暴露根上比较已封存 v3 首次开放环 response 与三个全链
CEM 配置。每个 CEM 配置以同一零均值做四次独立随机搜索，四次共
639 个完整链样本，另共享一次零链评分；每根实际 1920 token。
72 个 CEM 规划的所有模型输入、精确端点、评分、计量和最终
incumbent 经逐批重算。

| 方法 | 平均终距 | 终距 < 0.05 | response 逐根终距更低 |
| --- | ---: | ---: | ---: |
| v3 response | 0.414354 | 3/24 | — |
| 4 结点、四起点 CEM | 0.501557 | 0/24 | 15/24 |
| 8 结点、四起点 CEM | 0.501476 | 0/24 | 16/24 |
| 15 结点、四起点 CEM | 0.502344 | 0/24 | 18/24 |

这是一个有利于 response 的开发信号，不是 PC3 通过：根已暴露；
每个起点仍以零均值初始化，有限配置没有穷尽可用基线和调优；
这里只统一预测 token，没有使 SVD、其他计算和墙钟成本等价。
没有将三个基线的结果事后挑选为确认配置，也没有用这份开放环
结果替代闭环成绩或 PC2 同链机制证据。

- [72 条计划](../le-wm-main/results/pc_response_temporal_v3_multistart_cem_e0_20261104_codex_plans.json)，SHA-256 `e372cbe755ed8cf940a89307cb1b459cbf622b8d973f46b60dbe1af61832d251`
- [逐批审计](../le-wm-main/results/pc_response_temporal_v3_multistart_cem_e0_20261104_codex_audit.json)，SHA-256 `0f190dea60c8db1403f298982a0fdb5eeff92693f7988be83fb7a818a075b782`
