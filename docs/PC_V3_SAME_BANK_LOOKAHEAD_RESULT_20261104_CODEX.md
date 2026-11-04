# v3 同库普通 lookahead：事后机制对照

按[事先写入的诊断协议](PC_V3_SAME_BANK_LOOKAHEAD_PROTOCOL_20261104_CODEX.md)，
从 24 个已暴露根各自封存的九个自动生成前缀中，以每前缀相同的
120 次完整链评价选择普通 lookahead；另以相同预算评价精修后
selected 前缀。候选使用共同随机数种子，每前缀另计一次共同
selected 尾部评价。[审计](../le-wm-main/results/pc_v3_same_bank_lookahead_20261104_codex_audit.json)
重算了全部 **29,040** 条完整链。

原见证根 20 上，同库 lookahead 与即时 greedy 选择同一个前缀：

| 前缀 | 零尾终距 | 共同 selected 尾部终距 | 相对零尾变化 | 各自 CEM 终距 |
| --- | ---: | ---: | ---: | ---: |
| 精修后 selected | 1.622436 | 0.108833 | -1.513603 | 0.378979 |
| 同库 lookahead／greedy | 0.903645 | 1.277667 | +0.374021 | 0.502767 |

该根 selected 前缀不在原九项库内；全 24 根中有 17 根精修后
selected 前缀与库内某项完全相同。选中链共同尾部在 selected
前缀下的绝对终距优于同库 lookahead 为 20/24，相对各自零尾的
变化更好为 18/24；但同预算各自后缀 CEM 的终距严格更好仅
3/24。后者中的 lookahead 是从九次有限随机搜索中选择最优，
存在多重选择优势；此外各项差距大小不同，不能当作独立统计确认。

这些结果补充了原 `lookahead` 实际是整链 CEM 前缀的限制，但
**不改变原事前机制见证或使 PC2 通过**。同库选择及 selected
精修后库外均是诊断而非完整方法公平比较；充分多起点、其他后缀
预算、响应消融和开发外确认仍缺。原始记录见
[面板](../le-wm-main/results/pc_v3_same_bank_lookahead_20261104_codex_plans.json)
和[manifest](../le-wm-main/results/pc_v3_same_bank_lookahead_20261104_codex_manifest.json)。
首次入口编译因括号错误失败，未创建 manifest 或运行任务；修复后
按原协议执行，不涉及世界模型或 v3 算法修改。
