# v3 固定终点精确动力学闭环开发结果

按[事前协议](PC_RESPONSE_TEMPORAL_V3_FIXED_TERMINAL_E0_PROTOCOL_20261104_CODEX.md)，
四臂在 24 个已暴露 Wall 根上各自执行 15 步，在 0、5、10 步重规划。
288 次决策的所有批次、float32 输入、精确预测、评分、预算、carry、
选择、弃权和物理派发经重算审计；另外根据 CEM 批次数独立定位联合
精修边界，144/144 次结构化臂精修均从此前完整链的已评分最优开始。

| 臂 | 平均实际终距 | 终距 < 0.05 | response 逐根终距更低 |
| --- | ---: | ---: | ---: |
| response | 0.396386 | 5/24 | — |
| matched random | 0.458673 | 2/24 | 12/24 |
| smooth CEM | 0.458722 | 2/24 | 12/24 |
| action CEM | 0.500105 | 0/24 | 15/24 |

结构化臂每次实际使用 1919–1920 token，两 CEM 臂均用 1920；该
token 轴不含全部编码、分解和墙钟成本。response 有 2 次弃权，
matched random 有 3 次。以上是描述性开发结果，不能宣称独立强
基线优势；任务已暴露，基线配置不足，且未证明来源与统计合同。

v3 同时覆盖 15/10/5 步，保留全局最佳完整链用于联合精修及最终
弃权；受控探针反例和相关版本共 24 项测试通过。该修复解决了
先前明确的 incumbent 遗漏与精修起点问题，不表示候选搜索充分、
物理响应正确或 PC2/PC3 已通过。v3 尚无同链机制干预和真实 RGB
闭环结果，不能继承 v1 对应证据。

- [方法](../le-wm-main/experiments/spectral_reachability/pc_response_temporal_feedback_v3.py)，SHA-256 `8d84769cc70e993d1e5baf9f835c042a167da41fae4d4e50e28938872896e722`
- [原始轨迹](../le-wm-main/results/pc_response_temporal_v3_fixed_terminal_e0_20261104_codex_plans.json)，SHA-256 `122de4d2e455e0078be6f0b5e6f39753c0fe8eb9b370e26e07b7b6c6a65e75ec`
- [逐批审计](../le-wm-main/results/pc_response_temporal_v3_fixed_terminal_e0_20261104_codex_audit.json)，SHA-256 `db21a77dfa24a53bc55c4705f26cc97311fa8bae0d6759c25d75b8bf127ad421`
- [独立精修边界审计](../le-wm-main/results/pc_response_temporal_v3_fixed_terminal_e0_20261104_codex_refinement_boundary_audit.json)，SHA-256 `e44f9853369b6464686aa2375dedd8c9defd5e159b6b39ae3e5d4ebee2dd2103`

experiment-code 技能用于规范版本化、预先协议和如实记录；合成反例、
测试和开发面板均不替代科学确认。
