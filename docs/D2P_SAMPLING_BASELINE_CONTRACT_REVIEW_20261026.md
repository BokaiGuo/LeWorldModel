# D2P sampling kernel / CUDA cost bridge 合同复核

日期：2026-10-26。结论：**有条件可集成，尚不宜将现有收据当作完整、严格失败关闭的采样协议证明。**发现一项可复现的成本域验证缺陷，以及坐标、协议绑定和失败计数风险。不修改内核、桥或 runner。

## 范围与方法

仅阅读当前 `le-wm-main/experiments/spectral_reachability/d2p_sampling_baselines.py`（下称 K）与 `d2p_cuda_sampling_cost.py`（下称 B）。行号对应此次快照。按 code-debugging 的逻辑错误定位流程进行静态复核，并运行一次纯 NumPy 最小反例；未加载/执行 CUDA 桥、未运行神经 CPU/GPU，也未审阅 Hilbert runner 或导入的 P0 `_cost_fn` 实现。

这是针对两个模块的单独 review 交付，但 reviewer 也是 K 的实现者，不宣称作者独立或第三方盲审。原 57 项测试通过不能排除下述未覆盖输入域。

## 发现 1：实数成本域并未失败关闭（已复现，P2）

K:90 先执行 `np.asarray(cost_fn(argument), dtype=np.float64)`，之后 K:91 才检查形状和有限值。complex 成本可丢弃虚部，仅发出 `ComplexWarning`，随后作为有效实数进入排序。K:42 的 softmax 输入也采用相同先强转方式。B:51 仅检查 CUDA device、shape、`isfinite`，并不排除 complex 成本；B:53 最终 `astype(np.float64)` 同样可能静默丢弃虚部。因此默认真实 `_cost_fn` 是否会产生 complex 与接口本身是否严格验证必须分开：本轮证明的是接口缺陷，不声称当前模型已经输出过 complex。

最小反例（仅 NumPy 内核，现场执行）：

```python
import numpy as np
from experiments.spectral_reachability.d2p_sampling_baselines import random_shooting
params, receipt = random_shooting(
    lambda x: np.full(len(x), 1 + 2j), initial=np.zeros(1),
    lower=-1, upper=1, samples=2, seed=0)
# 实际：ComplexWarning；仍成功 best_cost=1.0, cost_evaluations=3
# 严格实数合同预期：ValueError，而非成功结果。
```

最小建议：在强转前取得原始 array/tensor，明确只允许 real numeric dtype（拒绝 complex/object/string/bool，或显式公布允许集合），再转换并检查转换后有限值。B 的自定义 cost_fn 同样检查浮点实数 dtype。添加 complex、数字字符串、bool 及转换后 overflow 的回归；不能用“warning 默认可见”代替失败关闭。其它输入如 complex initial/noise 的转换策略也应统一，但本轮未逐一复现，不作为额外已证实运行缺陷。

## 发现 2：params box 与物理动作 box 不能混用（条件性集成风险，P1）

K:3–5 泛称输入为 bounded actions；K:51–68 只保证变量位于调用方 `lower/upper`。B:40–45 接收 parameter proposals；B:58 则明确输出合同为 `unconstrained_parameters_tanh_mapped_to_normalized_physical_box`。B:33 的 `action_lower/action_upper` 是传给模型目标的 normalized physical box，并不是 K 的采样参数 bounds。

依 B 声明，设物理 normalized box 为 [-1,1]，对应动作 `a=tanh(p)`。若调用方将物理 bounds [-1,1] 直接传给 K，则动作只覆盖约 [-0.761594,0.761594]，不是完整 [-1,1]。例如目标动作 .9 所需参数 `arctanh(.9)≈1.472219` 已被排除。该数值关系用 NumPy 验证了 `tanh(1)=0.7615941559557649`；**并未读取 runner，故没有证据证明主线程已犯此误接错误。**

最小建议：接线显式区分 `parameter_lower/upper` 与 `normalized_action_lower/upper`，在 receipt 记录 tanh 及 affine mapping 合同。若保持 params-space 采样，公开额外参数截断范围，并保证 CEM/GN 等对照享有相同有效支持或披露差异；若期望 raw/normalized action uniform sampling，则需显式逆映射及端点策略，不能只更名 bounds。两种分布不能称为相同 random shooting 支持。

## 发现 3：当前结果收据不足以重建成本与采样域（确定的绑定缺口，P2）

K:98–104 返回 configuration，但 random 仅在 K:120 记录 samples/seed；MPPI K:156–157 记录 samples/iterations/temperature/noise_scale。两者都不记录 initial、lower、upper。现场输出 random configuration 为 `{'samples': 2, 'seed': 0}`。B:55–59 仅记录设备、horizon、坐标文字及评分数，不含具体 normalized bounds、dtype、目标/根/模型身份、cost 实现或其 metric。

因此同一 seed/配置摘要可对应不同搜索空间或不同目标；“收据相等”不能证明合同相等。K 的 callable 也可包含任意控制代价，单凭 `control_cost='caller_only'` 无法核实实际项。此处不是要求纯数值内核负责 checkpoint hashing，而是不能让当前最小 receipt 冒充完整实验绑定。

最小建议：K 或外层不可变协议保存 initial 与完整 broadcast bounds（或带形状/dtype的规范 hash）；B/外层绑定输入身份、normalized bounds、float dtype、目标/metric/cost 实现和是否包含控制项。若外层已有这些字段，引用其 hash 即可，不重复实现。当前审查范围不允许确认外层是否已经补齐。

## 发现 4：成功计数正确；失败计数需区分 submitted / converted / completed（确定语义差异，P2）

正常路径：random 为 2 次 callable、1+N 个候选；MPPI 为 1+2I 次 callable、1+I(N+1) 个候选。初始 incumbent、全部提案及每轮更新 nominal 都计数；best-ever 输出不重复计费。证据 K:88–95,114–119,140–155；B:46–50 每个通过转换的 batch 计一次 calls 和 batch length，正常情况下两层候选计数应相等。

失败路径不完全一致：

- K:88–90 在调用前计数，因此 dtype/device 转换失败也已提交；异常后内核不返回 receipt。
- B:43–47 在候选转换及有限检查后计数，因此转换失败未计入 bridge.calls。
- B:46–52 在成本验证前计数，因此成本 callable 抛错/输出 NaN 时，bridge.receipt 仍可显示新增 calls/samples，不能理解为成功评分。

最小反例（逻辑路径，不执行 CUDA）：float64 有限候选 `1e100` 通过 K 的有限检查，若 history 为 float32，B 转换后为 inf，在 B:45 抛错。K 已尝试一次，B 未计一次；无“成功成本”可用。另一例：自定义 cost_fn 返回 CUDA NaN 向量，B 已计 calls/samples 但拒绝输出。

最小建议：字段明确命名为 attempted/submitted/completed/failed，并在异常路径提供失败收据或由外层日志捕获。失败必须中断正式结果，不能返回 best-ever 冒充成功。现有行为确实中断而不回退，因此不是已发现的错误放行；主要风险是异常后桥的数字被外层误标为完成评分。

这些计数仅为候选成本，不是 predict、encoder、backward、token 或总算力。B:59 已正确限制声明。主线程必须另做模型调用计数，不能以 `candidate_cost_samples` 替代同预算证明。

## 其它有限性与冻结条件（静态风险，未证明当前运行违约）

- B:17–29 检查设备与动作 bounds，但不检查 pixels/goal/history 的完整形状与输入有限值；最终有限成本不保证输入合法，因为自定义 cost_fn 可忽略非法输入。B:30 检查顶层 `model.training` 与 parameter requires_grad，不检查所有子模块 `.training`；手动设子模块 train 的模型可能穿过这层检查。若外层已做完整冻结/输入验证，这可以作为显式前置条件；否则建议桥入口补检。未读取外层，不能断言模型实际非冻结。
- B:43 按 history.dtype 降精度：不同 float64 params 可映射为同一 float32（例如 1 与 1+1e-9）。这是正常 adapter 精度合同，不是必然 bug；best-ever 对应实际转换后成本，不保证重新批量/单例评分逐 bit 一致。若 GPU kernel 的 batch rounding 改变排序，需部署 singleton 检查并额外计数，不能无成本假定一致。
- K:46–48 最小值平移 softmax 避免普通指数 overflow；极大正差值变为 inf 时对应零权重，最小元素仍权重1，分母不会为零。K:148–149 对采样 overflow 明确失败，不返回部分最优。以上逻辑在声明的有限实数域内合理。

## Importance / control-cost 判定

K:7–14、135–139 明确：clipped Gaussian heuristic、无 density importance correction、control cost 仅由 caller 提供，best-ever 可选提案或 nominal。B 只是透传成本（B:49–50），没有新增 importance/control cost。**未发现两模块之间擅自添加或重复该项的代码证据。**

这不是严格路径积分控制、不是官方 MPPI 复现，也不是以最终 nominal 作为唯一动作的标准输出合同。Gaussian 经 clipping 后的边界质量、再经 tanh 的物理分布，均不能忽略后宣称满足标准 Gaussian importance 理论。建议报告名保持 `clipped_gaussian_mppi_heuristic`，记录 cost_fn 是否含控制惩罚，不把该实现视为已满足 faithful MPPI 强基线要求。真正 importance-corrected 变体应独立版本，不应悄悄改变历史内核合同。

## 最小接线准入清单

1. 修复/阻断非实数成本，并验证 cast 后有限值；加入上述反例。
2. 明确 params 与物理动作 bounds，封存 initial/采样域/成本定义及 adapter dtype。
3. 成功时核对两层候选计数；失败时拒绝有效实验结果并保留失败状态，不冒充 completed。
4. 外层验证完整模型冻结、根/目标身份、同反馈 schedule 和真实模型资源向量。
5. 保留 heuristic 与非官方复现标签；数值内核通过不意味着强基线、公平预算或 P1 通过。

本轮只新增本 review 文档，未修补被审模块，未运行 GPU。
