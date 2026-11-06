# Action-opportunity descriptor Stage-1 结果（2026-11-04）

## 执行与审计

运行脚本：

```text
PYTHONPATH=. .venv/bin/python \
  experiments/pc_action_opportunity_descriptor_dev_20261104_codex.py
```

结果：[result.json](/home/gbk/code/LeWorldModel/le-wm-main/results/pc_action_opportunity_descriptor_dev_20261104_codex/result.json)

- 16 个开发 root；
- spectral、random_direction、diverse 三个候选库；
- 2476/2476 候选完成；
- 48 个 prediction journal 分组全部通过调用数和 token 数核验；
- A probe 与 B probe 分离，B 未参与 A 描述构造；
- 没有新增物理执行、视觉模型训练或正式 planner 接线。

## 预定指标结果

| 范围 | 候选数 | Spearman(A,B) | Spearman(A,-best_score) | Spearman(-immediate,-best_score) | A top-10% 平均 best_score |
|---|---:|---:|---:|---:|---:|
| all | 2476 | 0.1288 | -0.2049 | 0.6382 | 0.7464 |
| spectral | 816 | 0.1600 | -0.2645 | 0.7198 | 0.8431 |
| random_direction | 844 | 0.1059 | -0.0722 | 0.3987 | 0.6050 |
| diverse | 816 | 0.1249 | -0.2495 | 0.7182 | 0.8311 |

全体封存 CEM 目标的 success 只有 1/2476，因此 success rate 不是有辨识力的主要
统计量；结果保留该分母，不对单个成功做解释。

## 结论

在当前 probe、exact predictor 和候选库下：

1. A 作用描述对未使用的 B 后缀只有弱一致性；
2. A 描述没有显示出对封存后续 CEM 质量的预测价值；
3. immediate score 的排序关系明显更强；
4. A top-10% 候选的后续质量反而更差，不能把“后果差异更大”当作有用性；
5. 因此不启动 Stage-2 retention，不修改正式搜索，不把该描述接入 PC2/PC3。

这是对本组有限后缀描述设计的有界负结果，不是否定“行为后果定义状态”的一般研究
方向。若继续，下一版必须先针对 probe 覆盖、动作尺度和验证目标另立协议，并继续
使用未参与描述构造的后缀；不能在本结果上事后调符号、挑 probe 或选择 root。

本结果仍不是视觉/物理响应认证、来源独立确认或 PC2/PC3 证据。
