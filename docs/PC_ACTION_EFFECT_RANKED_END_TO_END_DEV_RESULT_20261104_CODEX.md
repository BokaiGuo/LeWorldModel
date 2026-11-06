# 任务得分预筛：端到端开发结果

按[运行前协议](PC_ACTION_EFFECT_RANKED_END_TO_END_DEV_PROTOCOL_20261104_CODEX.md)，在同族的新参数组合上比较即时分数排序、动作前缀去重后的同探针两槽保留。任务不与所检查的旧面板逐项重复，但并非来源独立。80/80 次运行完成；递归臂耗费 2878–2880 prediction tokens，global CEM 为 2880。

| 方法 | 成功 | 平均最终终距 |
| --- | ---: | ---: |
| random_min | 5/16 | 0.32913 |
| random_effect | 4/16 | 0.35809 |
| spectral_min | 4/16 | 0.35843 |
| spectral_effect | 4/16 | 0.35843 |
| global CEM | 6/16 | 0.32470 |

相对 min，random_effect 的 root 配对终距为 3 根更好、2 根更差、11 根相同，平均差 **+0.02896**（越低越好）；spectral 两臂的 16 根最终终距在 `1e-12` 容差内相同。面板较小，平均差受各根效应大小影响；不能将胜负数量或结果当作独立统计确认。

## 干预是否真的改变选择

该版本解决了上一轮“质量窗口内从无两个非 greedy 候选”的设计问题：

- random：32 个非末层决策中，17 个有至少两个合格非 greedy 候选；min/effect 两臂跨全部三层共 18 层的实际保留动作集合不同；
- spectral：32 个非末层决策中，21 个有至少两个合格非 greedy 候选；两臂跨全部三层共 13 层的实际保留动作集合不同。

层间存在递归依赖；某一层不同可能只是更早保留差异的下游结果，因此这些层数不是独立干预次数。虽然两臂实际动作集合发生差异，本面板仍没有显示后果互补在最终链上优于 min；尤其谱方向尽管保留发生变化，最终终距没有变化。不能据此断言所有后果描述无效。

## 审计与限制

[审计器](../le-wm-main/experiments/audit_pc_action_effect_ranked_end_to_end_dev_20261104_codex.py)在同一环境检查了 80 次运行的日志哈希、模型调用 token、共享初始全局库、精确动力学预测与返回完整链，并按实际动作而非候选索引核对保留差异。运行前的任务查重只针对列出的旧面板；新 seed 和近邻参数变化不证明来源独立。

即时得分预筛可能丢弃有延迟收益的候选，不能把本面板解释为生成器、预筛与互补保留各自的因果贡献。global CEM 优于各递归臂只是该开发面板的观察；prediction-token 上限不是全部系统成本，也未完成 RGB、同链机制或独立确认。原 PC0–PC5 判定不变。

入口：[运行器](../le-wm-main/experiments/pc_action_effect_ranked_end_to_end_dev_20261104_codex.py)；原始动作、调用和审计回执位于 `le-wm-main/results/pc_action_effect_ranked_end_to_end_dev_20261104_codex/`。
