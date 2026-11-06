# PC 来源合格确认预检（2026-11-04）

## 目的

本记录把 PC2/PC3/PC4/PC5 确认前的三条来源边界分开审计：

1. 确认任务不属于方法开发历史；
2. normalizer 的拟合来源与 checkpoint 绑定可证明，且确认任务未参与拟合；
3. checkpoint 的训练数据成员、训练 split 和训练运行可追溯，并与确认任务不相交。

三者不是同一个命题。新 seed、同一 archive 中未使用的根、CPU checkpoint 重放或已有
source seal 均不能单独证明三条边界。

## 本轮执行

运行：

```text
.venv/bin/python \
  experiments/spectral_reachability/codex_p1_checkpoint_state_source_join_20261030.py \
  --output results/codex_p1_checkpoint_state_source_join_20261030_CONTINUE01.json
```

该程序是 CPU/source metadata-only 审计；本轮没有模型推理、GPU、物理 reset、候选
选择或 outcome 读取。输出 SHA-256：

```text
af9867acc0ad4125e56991355f1f081ae8309e884563f866140e3081e759e91c
```

## 结果

| 边界 | 当前状态 | 解释 |
|---|---|---|
| checkpoint bytes/config identity | 可复核 | 固定 weights/config hash 与保存的 CPU state projection 一致 |
| 开发历史不相交 | unknown | 没有完整、权威的开发历史覆盖与排除 receipt |
| normalizer 拟合来源与 checkpoint 绑定 | unknown | 当前本地统计可复算，但统计成员、拟合运行与 checkpoint 训练绑定未证明 |
| checkpoint training dataset/member identity | unknown | 没有权威训练数据成员清单和训练运行 receipt |
| checkpoint training split 与确认任务不相交 | unknown | 不能从 checkpoint state/name 或新 seed 推断 |
| historical reset ledger complete | false | 历史 reset/load/source 范围仍不完整 |
| P1 admission | false | fail-closed |

正式确认因此不应启动。当前结果是来源资格阻断，不是理论失败，也不是物理任务负结果。

## 需要补齐的权威证据

只有在以下材料同时存在并通过哈希及时间顺序核验后，才能打开独立确认运行：

- checkpoint training run receipt：训练代码、解析后配置、数据版本、完整成员清单、
  split manifest、训练开始前的 checkpoint hash 链；
- normalizer fitting receipt：拟合输入成员、统计量、拟合时间、输出 hash，以及与
  checkpoint 的明确绑定；
- development exclusion receipt：开发任务、调参任务、历史测试和预留/失败来源的
  完整排除范围；
- confirmation source receipt：确认任务在方法冻结后产生，且 root/reset/action/RGB
  identity 完整；
- 历史去重 receipt：精确 hash 与语义/近邻去重范围，失败和未选中来源也在分母内。

在这些证据到位前，允许做的工作仅限于来源材料补证、静态审计、代码修复、测试和
明确标记为 development 的诊断；不得把当前开发面板升级为 PC2/PC3/PC4/PC5 确认。

## 与当前研究状态的关系

本预检不改写既有结果：等槽位 retention 面板、同版本 PC2 机制干预和失败链仍在
原始分母内；root 8 的 1/16 smooth-mechanism 见证仍是开发证据，不能升级为 gate
通过。来源阻断解除后，必须在同一冻结核心上重新开展独立配对的 PC2/PC3，并同步
审计完整墙钟/GPU/编码/同步/内存成本和真实 RGB 闭环。
