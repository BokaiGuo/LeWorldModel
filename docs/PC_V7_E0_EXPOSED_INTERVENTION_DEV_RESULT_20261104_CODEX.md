# v7 已暴露精确动力学面板：同链机制重算

按[开发协议](PC_V7_E0_EXPOSED_INTERVENTION_DEV_PROTOCOL_20261104_CODEX.md)，对 v7 最终选中的 18 根动作链重新运行已有的共同后缀、相同预算重规划及任务相关后续响应计算。[计算记录](../le-wm-main/results/pc_v7_e0_exposed_intervention_dev_20261104_codex.json)显示，两个原定机制交集均为 **0/18**：selected 相比 zero 与 same-bank greedy 的基础交集为零，进一步加入 lookahead 和随机桥接的交集也为零。

18 根重新生成的干预结果和动作、预测数组与已有 v5 干预 v2 产物逐项一致；单独的[绑定审计](../le-wm-main/results/pc_v7_e0_exposed_intervention_dev_20261104_codex_binding_audit.json)核验了 v7 最终链、历史结果索引与每根元数据和数组的哈希。计算仍复用相同动力学和干预实现，且根已暴露，不是独立复现，也不表明机制在其他任务上不存在。

首次启动误指向旧干预的未完成审计文件，在计算前停止；随后绑定实际已完成并审计的干预 v2，复用其候选索引规范化，未修改根、动作、阈值或历史产物。该结果未建立 PC2，也不能代替 PC3 强基线比较。
