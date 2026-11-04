# V5 响应臂：固定随机桥接补充结果

按[协议](PC_V5_CANONICAL_RANDOM_BRIDGE_PROTOCOL_20261104_CODEX.md)，24 根各生成一条固定种子的随机前缀，并与封存的 v5 最终 selected 前缀在相同零尾初始化、119 样本后缀 CEM 及共同尾部下比较。

| selected 相对随机桥接 | 胜 | 平 | 负 |
|---|---:|---:|---:|
| 各自同预算 CEM 终距 | 19 | 0 | 5 |
| 同一 selected 尾相对各自零尾的变化 | 20 | — | — |

自行规划终距以 `1e-10` 为平局容差；共同尾只列改善更大者数量，不把平局并入失败。随机桥接及 selected 在 `.01/.05` 的后继响应矩阵差、任务梯度差、阻挡序列和六个共同尾结果均已保存。前述 selected 对 zero 与 greedy 的完整机制交集已经为 0/24，所以补入 random bridge 后不能将交集变成正见证。

[封存轨迹](../le-wm-main/results/pc_v5_canonical_random_bridge_20261104_codex_data.json)及[重建审计](../le-wm-main/results/pc_v5_canonical_random_bridge_20261104_codex_audit.json)覆盖 24 条随机前缀、3,024 条后缀/共同尾评价以及 4,032 条局部响应探针；审计与原运行共享精确动力学代码。各根仅一个随机桥、一个有限 CEM 配置，不构成强搜索上界；任务和 v5 结果此前已暴露，不能认证 PC2 或 PC3，也不能将此胜负比解释为泛化优势。
