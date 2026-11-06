# 行动后果端到端开发比较：未触发的保留干预

按[运行前协议](PC_ACTION_EFFECT_END_TO_END_DEV_PROTOCOL_20261104_CODEX.md)，在已有 16 个暴露 E0 任务上运行随机/谱方向 × 最低代价/任务约束后果互补四个递归臂，以及 global CEM。80/80 次运行完成；每次上限 2880 prediction tokens，递归臂实际消耗 2878–2880，global CEM 消耗 2880。

| 方法 | 成功 | 平均最终终距 |
| --- | ---: | ---: |
| random_min | 3/16 | 0.38666 |
| random_effect | 3/16 | 0.38666 |
| spectral_min | 2/16 | 0.41467 |
| spectral_effect | 2/16 | 0.41467 |
| global CEM | 6/16 | 0.32637 |

## 关键结论：本面板不识别互补保留的效果

同方向的 min/effect 两臂在所有 root 的全部层选择了相同保留索引，最终分数也全部相同。每一方向各有 32 个非末层决策，每层均测了 4 个候选；但是按预定 0.05 绝对代价窗口，非 greedy 候选的合格数如下：

| 方向 | 0 个合格 | 1 个合格 | 2 个或更多 |
| --- | ---: | ---: | ---: |
| random | 23 | 9 | 0 |
| spectral | 28 | 4 | 0 |

没有两个合格候选时，互补规则无法在合格候选间改变选择；无合格者时按协议回退到 min。因此两臂相同不是“互补机制已证明无效”，而是**此面板的干预没有改变实际决策**。不能根据这些相同分数推断机制效果。下一步若要测试互补性，必须在新协议中事前定义如何获得至少两个有任务意义的可比较候选，并同时计入获得这些候选的成本；不能在本面板结果出现后调大窗口并当成原实验。

## 审计和限制

新增的[审计器](../le-wm-main/experiments/audit_pc_action_effect_end_to_end_dev_20261104_codex.py)验证 80 个调用日志的哈希与 token、共享初始全局候选、当前精确动力学的输出、最终选中链确实已评分且不劣于已评分完整链，并重算干预激活数量。该审计是同环境复核，不是独立复现；该次审计未做完整算法重放或全系统成本认证。对选择函数的测试另验证存在两个合格候选时互补规则可以与 min 选择不同候选。

本比较使用已暴露任务；token 上限不是墙钟/GPU/内存完整成本。global CEM 在此面板更好仅是开发观察，不是独立统计确认。旧 PC0–PC5、同链机制和此前行动后果诊断均不改判。

实现入口：[运行器](../le-wm-main/experiments/pc_action_effect_end_to_end_dev_20261104_codex.py)，[独立开发核心](../le-wm-main/experiments/spectral_reachability/pc_recursive_action_effect_dev.py)。原始结果与调用日志位于 `le-wm-main/results/pc_action_effect_end_to_end_dev_20261104_codex/`。
