# CEM 候选与 winner replay 的共同成本轴接线

开发性、独占接口；不修改旧合同、旧控制器、结果或 gate。采用既有
variable CEM 四轮候选和 winner 重算，每个候选及 winner 重算：

1. 在真实 durable driver 上写入并计费 `replay` 标记；
2. 经同一 LeafAdapter 调用 action encoder 与 predictor；
3. 用 durable `selection` dispatch 包围实际有限性检查与目标分数计算。

driver 禁止嵌套 dispatch，因此 replay 标记在 rollout 前单独完成；
它是工作开始的预留，不是围绕整个模型 rollout 的时延测量。selection
确实包围计算，失败与已预留开销保留。winner 同样计一次 replay 和
一次分数计算，避免把复验当成零成本。它们是事件数，而非可与前向、
反向调用相加的 FLOPs；既有 CEM 日志的选择和计费轴意义仍需独立复核。

| boundary | horizon | 每轮候选 | 前向调用 | replay | selection |
|---:|---:|---:|---:|---:|---:|
| 0 | 3 | 16 | 195 | 65 | 65 |
| 5 | 2 | 24 | 194 | 97 | 97 |
| 10 | 1 | 48 | 193 | 193 | 193 |
| 15 | 1 | 48 | 193 | 193 | 193 |

先前共同接口 replay/selection cap=64，连 boundary 0 都不够。
本独占新合同为**两臂**逐节点将这两个 cap 固定为 193，前向仍为
195，其他轴沿用原接口。旧接口和旧合同不变。预算来源于已见开发
结果，只有在未来独立任务结果产生之前冻结，才是该新实验的事前合同；
相同上限不表示相同实际费用、总算力、FLOPs 或时延。

新协议固定旧共同协议、上限、语义及控制器源码哈希。`bind` 检查两臂
合同身份后创建新节点合同；`Controller` 使用既有 variable CEM 的
step 源码生成独立函数，仅替换局部 score，不能影响封存模块全局。
每次 step 检查新鲜 driver、node 合同、节点优化器、来源和摘要。
来源校验要求包含 adapter、旧共同接口、variable/base CEM 和 `_normal`
所在模块；其传递闭包仍由宿主负责，不能将其称为完整来源认证。

测试使用原有 controller、真实 durable Driver 和 LeafAdapter，模型和
物理回执是非神经 CPU fixture。四个边界分别验证各轴精确计数、逐项
intent 和提前停止；另检查旧 cap 不足、错节点、协议篡改、模型回调
异常时已记账且不派发动作。CUDA、神经推理和真实物理均未执行。

尚未把新合同接入 structured 规划器，也未提供真实闭环宿主、实际 RGB
历史认证、全来源闭包、模块覆盖或实际 GPU 成本认证。score 改写依赖
既有控制器函数的源码结构，未来来源变化须重新审查并生成新协议；
它不是可直接执行的主机运行命令，也不主张等成本或 P1 admission。

```sh
cd /home/gbk/code/LeWorldModel/le-wm-main
env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_common_budget_cem_events_20261101_leibniz.py tests/test_p1_common_budget_node_20261101_leibniz.py
```
