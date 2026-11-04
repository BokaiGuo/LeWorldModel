# v3 新网格事前同链机制开发结果

按[事前协议](PC_RESPONSE_TEMPORAL_V3_PROSPECTIVE_MECHANISM_PROTOCOL_20261104_CODEX.md)，
24 个网格根的 72 次规划先封存并逐批重算，然后对每根 response
最终选中链进行共同后缀、等预算自行重规划和后继响应干预。
24 根干预通过同代码确定性重建；这不是独立实现复现。

预定非即时条件满足 6/24；同时满足零前缀与 greedy 两个对照的
收益及零名义响应差异阈值的交集为 **0/24**，对全部四类对照的
交集也是 0/24。单项计数不能相加或替代同一链交集：选中尾部的
相对零尾收益分别优于 zero 17/24、greedy 14/24；自行规划终距
分别优于 zero 19/24、greedy 8/24。

## 负结果与归因

根 0 和 6 满足非即时、共同选中尾部的收益比较以及各自重规划
比较，但零名义响应矩阵对 zero 的差分别为 0 和约 `8.35e-15`，
小于预定 `1e-6`；因此不能算作预定见证。根 0 在选中前缀后的
共同选中尾部终距为 0.03380，而 zero 前缀为 1.15733；各自
重规划终距分别为 0.03812 和 0.50064。单项有效收益不等于
已证明响应改变或强基线优势。

按照[事后归因协议](PC_RESPONSE_TEMPORAL_V3_MECHANISM_RESPONSE_ATTRIBUTION_PROTOCOL_20261104_CODEX.md)，
对六个非即时根核对多个差分尺度，并重现原零名义、0.05 的矩阵。
零名义下根 0 的 selected 与 zero 差异在三个尺度均接近零；改用
裁剪的选中尾部首五步为共同名义动作后，0.05 尺度差异为 2.485，
但 0.01 尺度仍接近零。根 6 在该事后名义下 0.01、0.05、0.1
尺度的差异分别约为 0.671、0.670、0.668。

这提示零名义局部响应不是全部动作区域的充分刻画，且根 0 存在
明显尺度敏感性；不预设其原因，也不把事后选取的名义动作追认为
原协议见证。原事前交集仍为零；PC2 尚不能宣告通过。新网格只
核对了与所引用 24 根网格的参数不重合，未完成全仓历史排重或
来源独立认证，不能将它视为独立确认。

- [72 条规划](../le-wm-main/results/pc_response_temporal_v3_prospective_mechanism_20261104_codex_plans.json)，SHA-256 `b445dfd0e0e1a24b07aa8085d2d9fb26f1106186e9563a1abeecc6085329afcd`
- [规划审计](../le-wm-main/results/pc_response_temporal_v3_prospective_mechanism_20261104_codex_audit.json)，SHA-256 `9f078ffc8a1aad42a64615739b7ea3bca5ba2806243ff96bdf64d8bd98ccb1f2`
- [干预数据](../le-wm-main/results/pc_response_temporal_v3_prospective_mechanism_20261104_codex_intervention.json)，SHA-256 `c3fe04f30f1e21ca4c29b28ae0398d2d145598c976628b56437e06390db8dc7b`
- [确定性重建](../le-wm-main/results/pc_response_temporal_v3_prospective_mechanism_20261104_codex_intervention_rebuild.json)，SHA-256 `b00aa26a0ec8f5ebeca3bd5c041011b3135404921ad1460e44d08f5c65c13cd8`
- [事后多尺度归因](../le-wm-main/results/pc_response_temporal_v3_prospective_mechanism_20261104_codex_response_attribution.json)，SHA-256 `eaffeeaeb2330c34727829919d2a2d2036cb0dc5de4c1cc1a49f699c6fe7b272`
