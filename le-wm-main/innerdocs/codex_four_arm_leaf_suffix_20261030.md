# 四臂叶子计费与共同后缀：离线接口

仅新增本说明、`codex_four_arm_leaf_suffix_20261030.py` 和对应测试；不修改现有
runner、预算合同或 P1 门。未运行 GPU、物理、真实模型或训练。

## 现有模型接口核查

已安装的 `stable_worldmodel/wm/lewm/lewm.py` 中，`LeWM.encode(info)` 接收
`pixels`，且在 info 含 `action` 时内部也运行 `action_encoder`；
`LeWM.predict(emb, act_emb)` 内部运行 `predictor` 和 `pred_proj`。
已有 `p2_grasp_reference_core_20261027_v1.LeWMH1` 在 edge/advance 分别调用
action encoder 和 predict；其 H1 初始化与真实 H3 不等价。已有 H15 neural
adapter 接的是另一套 native ledger ABI，不能直接视为四臂账本适配。

## 新接口和计数

`LeafAdapter(model, driver, contract, contract_sha, analytic_fixture=False)` 验证
四臂合同及 durable driver header 绑定，公开 `encode(info, phase=...)`、
`action_encode(actions, phase=...)`、`predict(states, action_embeddings, phase=...)`、
`gradient(loss, actions, phase=...)`。每次调用分别通过原四臂 `Driver.dispatch`，
不嵌套；encode 明确仅允许 pixels，防止隐藏的 action-encoder 调用。
每次 encode 按输入 batch 计 encode_calls/encode_samples；action encoder
计 action_encode_calls；predict 按 batch 计 predict_calls/predict_samples；
标量 loss 的 `autograd.grad` 计 backward_calls=1、vjp_rows=1。每个预留在
callback 前持久化，异常保留失败，超额阻断实际调用。

`optimize(leaf, states, history_actions, initial_actions, goal, config)` 是
一个固定步数、固定学习率、逐坐标 box 投影的共同 suffix projected gradient
descent，目标为最终 latent 与 goal 的均方差；每一步序列 rollout 的每个
action encoder、predict、以及一次总 loss 反向均单独入账。输入形状依次为
`[B,H,D]`、`[B,H-1,A]`、`[B,T,A]`、`[B,D]`；每个预测使用对齐的历史和当前
action。返回更新后的动作，**没有末次评估、选择或物理验证**。config 精确字段
为 `steps`, `learning_rate`, `history_size`, `lower`, `upper`，其 canonical
SHA 须等于四臂合同的 `identity.optimizer_config`。

这只是可运行、明确的共同优化器接口，不是 GRASP/Adam/CEM，不保证足够强、
不复现论文实现，也不自动绑定 checkpoint 或证明合同中 optimizer hash 就是本代码。
四臂同 config 不保证它们在 model/task/action domain 上真实一致。不得把仅有该接口
说成已经完成强优化器公平比较。

## 主线程接线

先按前瞻协议固定共同任务、模型/源码、四臂初始化及预算曲线，并完成四臂前置费用
和其他阶段的同一 durable driver 接线。每臂每预算点使用不同的 Driver，随后：

```python
from experiments.spectral_reachability import codex_four_arm_leaf_suffix_20261030 as suffix

leaf = suffix.LeafAdapter(model, driver, contract, contract_sha256)
actions = suffix.optimize(leaf, encoded_history, past_actions, initial_actions, goal, config)
receipt = driver.finish()
```

只接收显式 frozen/eval CUDA `torch.nn.Module`；CPU 路径仅用于明确声明的非 Module
解析替身。实例化不装载模型，不启动设备。真实模型的 `encode` 应经过 leaf 的
pixels-only 接口；实际初始化、筛选、Jacobian/VJP、trial、独立验证及物理入口
仍须另外接入同一 driver。不要在一个 driver callback 内嵌套 leaf 调用；也不要将
既有 runner 包在外层 callback 后又加内部 hook。任何失效后应保留失败 journal，
由已有独立 consumer 和外部可信摘要验证，且不能把异常后的动作当完成端点。
适配器发现内部验证失败时通过既有 driver 的 invalid 事件 fail-stop；如日志 I/O
本身失败，可能没有可消费的终态，不保证失败报告落盘。

## 仍缺

本接口计的是 LeWM **公开外层调用**，而非其内部 predictor、projection、
autograd kernel 的独立 hook；action context token 数、内部编码、优化器张量运算、
同步、FLOPs、CPU、峰值显存、wall 和物理成本均未完整测量。encode_calls 的
sample 是 batch 而非帧数；predict_samples 是 batch 而非 context token 数。
只计入这些轴不等于所有成本或等 FLOPs；同步可能在有限性检查时发生。
不能证明其他代码没有绕过接口、模型/参数在运行中未被修改、异步 CUDA 任务
已完成、合同被事前独立承诺或真实资源公平。此前置还包括预先绑定并验证代码闭包、
真实模型 ABI/precision/action normalization 与 H3 历史、初始化来源及分母、
候选产生的 Jacobian/筛选/试算费用、强优化器选型与比较设计、真实 hook 覆盖、
各 budget point 的四臂完整结果，以及独立 P1 审核。均未在此认证。

## 测试

先写测试，确认模块不存在时收集失败；随后实现并补充回归。最终在 `le-wm-main`
运行下列命令为 `63 passed in 7.73s`：

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -m pytest -q -p no:cacheprovider \
 tests/test_codex_four_arm_leaf_suffix_20261030.py \
 tests/test_codex_four_arm_durable_20261030.py \
 tests/test_codex_four_arm_full_cost_20261030.py \
 tests/test_p2_grasp_reference_core_20261027_v1.py \
 tests/test_p2_h15_neural_ledger_adapter_20261030_codex.py
```

测试包括解析 CPU 替身的可求导 H1/H3 rollout、按事件消费计数、超额前阻断
predict/backward、模型异常、非有限输出、配置哈希、禁 CPU 神经 module、
嵌套调用 fail-stop；不是实际模型或预算公平证据。
