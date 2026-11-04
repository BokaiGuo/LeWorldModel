# v8 搜索诱导名义点：E0 开发结果

按[预定协议](PC_V8_SEARCH_INDUCED_NOMINAL_E0_DEV_PROTOCOL_20261104_CODEX.md)，
v8 用统一预算内的 48 样本全链 CEM 提出两个响应名义点，再执行 v7
响应搜索，并将短搜索、v7 和清零候选同批比较。新入口为
[pc_search_nominal_v8.py](../le-wm-main/experiments/spectral_reachability/pc_search_nominal_v8.py)。
不使用解析见证或特权最优动作；名义点受普通全链 CEM 影响，
因此结果不能单独归因于谱结构。

| 方法 | 平均终距 | 中位终距 | `<0.05` | v8 对其胜／平／负 |
|---|---:|---:|---:|---:|
| v8 | 0.501738 | 0.501494 | 0/18 | — |
| 已封存 v7 | 0.501111 | 0.500535 | 0/18 | 3／0／15 |
| 三重启 smooth CEM | 0.501265 | — | 0/18 | 6／0／12 |
| 三重启 action CEM | 0.501926 | — | 0/18 | 8／0／10 |

胜平负容差为 `1e-10`；这些是已暴露开发根上的配对结果，不是充分
调优的强基线确认。每根 v8 使用 1920 token，17 根最终选择内部 v7 链、
1 根选择短搜索链。规划、候选生成与最终比较均计费，但 token 不是完整
硬件成本。该配置不支持“仅用短搜索诱导名义点即可解决覆盖”的解释；
也不能排除其他名义点、优化预算、组合策略或模型接口的影响。

[记录](../le-wm-main/results/pc_v8_search_induced_nominal_e0_dev_20261104_codex_result.json)
保存所有请求、实际 float32 动作、得分、响应和精修；
[审计](../le-wm-main/results/pc_v8_search_induced_nominal_e0_dev_20261104_codex_audit.json)
用同一精确后端逐批重算 13,266 条轨迹并检查最终链与预算。该审计不构成
独立物理后端验证。二维和五维、三个剩余时域的合成接口检查为 7 项通过，
不证明 RGB 任务性能。

## 最终链干预

计划封存之后另按[事后协议](PC_V8_E0_EXPOSED_INTERVENTION_DEV_PROTOCOL_20261104_CODEX.md)
对 18 根 v8 最终选中链做共同尾、三流各 480 样本重规划和任务相关响应检查。
5/18 前缀不改善初始距离，其中仅 2/18 同时为非零前缀；相对 zero 与
same-bank greedy 的基础机制交集 **0/18**，加入 lookahead 和随机桥接后
的交集同样 **0/18**。各前缀三流重规划终距的跨根均值分别为 selected
0.500595、zero 0.500900、greedy 0.500703、lookahead 0.483314、
random 0.501618。lookahead 包含额外候选筛选，不能视为等成本部署对照。

[干预索引](../le-wm-main/results/pc_v8_e0_exposed_intervention_dev_20261104_codex_result.json)
及各根数组保存实际干预动作和结果；
[重算审计](../le-wm-main/results/pc_v8_e0_exposed_intervention_dev_20261104_codex_audit.json)
使用同一实现、同一后端重新生成全部结果，并非独立实现复现。
这批干预在看到 v8 开发成绩后才固定，不能称完全事前注册；
它未建立 PC2，也不证明其他任务和算法不存在机制。
