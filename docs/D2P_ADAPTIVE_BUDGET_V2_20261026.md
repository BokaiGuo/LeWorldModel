# P3 动态时域与预算合同 v2（2026-10-26）

状态：仅前瞻性基础设施与合同测试；**P3 未准入，无 compute advantage 结论**。
P1 未通过；最新封存 PushT 4×16 物理执行虽通过完整性门，selected 均劣于
zero-prefix，并有 raw/projected 指标失配。不得据本文件开启依赖科学阶段、
放宽门限或把数组/元数据测试当物理收益。指标合同审计由父任务处理。

## 缺口定位与复用

已阅读原提案附件 `861f6764-1b25-4227-bf15-f1a237ba7d8f/pasted-text-1.txt`
及 `D2P_RESEARCH_REQUIREMENT_EVIDENCE_MATRIX_20261026.md`、
`D2P_RESEARCH_CLOSURE_AUDIT_20261025.md`。提案要求由局部动力学决定
想象长度/搜索分配，并以真实同预算任务结果检验，而非阈值即贡献。

在当前 `le-wm-main` Python 源码检索未发现独立 D2P 自适应调度类；现有
原型能力分散在采样桥、调用记录和资源比较合同中，不应假称已有完整调度器：

| 已有对象 | 已具备 | 本次具体补缺 |
| --- | --- | --- |
| `d2p_planning_compute_ledger.py` | 完整资源向量、绑定、事后等预算/上限比较 | 本身不拦截模型调用 |
| `d2p_model_call_instrumentation.py` | 实际 batch 逐次计数、失败保留、CUDA 元数据检查、phase | 复用它；新增调用前硬预算门 |
| `d2p_cuda_sampling_cost.py` | CUDA 候选成本、提交/完成/失败分别记录 | 候选数不等于逐 token predict 样本，不能拿 completed 充总账 |

仅新增 `experiments/spectral_reachability/d2p_adaptive_budget_v2.py`、对应测试
与本说明；不修改已有封存源码，不训练、不装 adapter、不启动神经计算。

## 可执行语义

`FrozenSchedule` 固定短/长 horizon、base_candidates、expansion_factor、χ/gain
阈值；边界相等时取长时域/扩展。`FrozenSignals` 仅接受同 source_seal 的
有限 [0,1] unreachable_fraction 与 horizon_gain。不读取 reward、success、
物理终态或 evaluator 输入；同一冻结输入给出确定选择。来源 seal 是外部
验证责任，类型无法证明调用方未把 outcome 伪装成谱信号，也不能证明前瞻冻结。

调度输出的 candidates×horizon **仅单次完整 rollout 的预测样本下界估计**，
不是实际收据，也不自动授权运行。信号计算本身、重复搜索、额外验证均另计。

`PredictionBudget` 绑定 protocol/task/root/frozen-model 和 method，单次使用。
必须在任何预测前进入 guard，并用显式 phase 覆盖谱信号构建、候选预测、
verification 与 evaluation；所有 phase 共用一个不可自动重置的上限。
每次实际 `model.predict` 从输入读真实 batch B，在委托前检查并扣 B；
每个未来 token 再调用就再扣，扩展/重排/重试不靠预期候选数估账。
模型抛异常也不退款；整批超限则调用前拒绝，不部分执行、不截断 batch。
错误设备/非法形状/无 phase 在模型执行前拒绝，不产生预测工作。

收据包含绑定、attempted_predict_samples、blocked_calls 及原始 phase 记录，
不以 successful-only 子集报告总成本。phase 的 completed 仍沿用旧计数器：
表示该 scope 正常退出且无失败委托，不表示没有捕获过超限拒绝；必须同时
检查 blocked_calls。失败后已消耗预算不能通过下一 phase 清零。

接入时：`with PredictionBudget(model, binding=..., method=..., sample_limit=...)`
内再使用 `budget.phase(...)`，实际 runner 沿原 `model.predict` 路径调用。
本次未把 guard 接入封存 runner，因此不存在新实测预测资源或闭环结果。

## 边界与剩余责任

- 仅约束该实例 predict 属性路径；预缓存 callable、内部其他模型、编译/捕获
  执行、并发 phase 或调用期间替换 predict 不在支持范围，不得据此宣称覆盖。
- CPU 输入拒绝；生产神经工作只允许 HOST CUDA，无 CPU fallback。本次测试
  是无 torch/无神经执行的 CUDA 元数据假体，不能证明 GPU 实际可用。
- 样本预算不约束 encoder、VJP/backward、FLOPs、GPU 峰值、墙钟；继续保留
  原资源向量并独立测量，不能由样本少推出算力优势。
- 外层须验证模型真正冻结/eval、host CUDA 设备、root/goal/protocol 绑定、
  source seal、所有预测路径覆盖与外部前瞻性封存；guard 不替代这些门。
- 没有 P1/P2 后的独立真实固定调度/几何调度/打乱信号同预算比较，不得声称
  P3 通过；还需完整失败轨迹、谱开销、噪声消融和质量—成本曲线。

## 验证

在 `le-wm-main` 使用 `.venv/bin/python -m pytest -q
tests/test_d2p_adaptive_budget_v2.py tests/test_d2p_model_call_instrumentation.py
tests/test_d2p_planning_compute_ledger.py`：**132 passed**。
覆盖扩展 batch×token、精确上限/超限、失败不退款、重试、跨 phase 总预算、
CPU/非法输入拒绝、方法绑定、恢复原 predict、收据隔离、单次使用、冻结且
确定的信号选择、非法数值与 outcome 字段拒绝。均为合同验证，非科学效能。
