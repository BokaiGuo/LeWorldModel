# 真实 RGB 反馈梯度 MPC20：独占实现交付

## 固定协议与算法

入口为 `experiments/spectral_reachability/agent_p3_feedback_gradient_mpc20_20261028_v1.py`，测试为 `tests/test_agent_p3_feedback_gradient_mpc20_20261028_v1.py`。不修改原 P3 调度器、冻结模型或原 receipt。

复用原 P3 known2 roots × fixed/adaptive 标签共4协议槽；标签只标识原参考槽，本实现四槽均执行相同梯度算法，不是4个独立根。每槽最多20步，在实际0/5/10/15边界编码自身真实 RGB，滚动真实 latent3 与实际 raw10，重新优化 full10，只执行前5步，丢弃后5步。边界15的模型终点是25，独立记录的真实终点是20，两者不能直接冒作同时间预测误差。早停、失败及未到达节点均保留4槽/16节点分母，不补根、不补步、不复用过时动作。

原 checkpoint、identity192 mean squared Q、原 action mean/std、native whole-agent+object+angle goal/success 合同不变。没有 object-only coverage 代理成功。每节点零 logits 初始化，固定32次动作标量梯度更新、eta=1、每packet logit半径 `.75*sqrt(5)`。动作 chart 明确为 NumPy `tanh(float32 logits)` 到 literal rawfloat32；新鲜 raw leaf `[10,2]` 的 scalar meanQ 梯度乘解析 `1-raw²` 链一次，再执行原投影递推。选择 zero+32候选的稳定模型Q最小值，随后独立三次 fullH2 replay。没有 outcome 输入、训练、拟合、线搜索、warm start 或按效果选预算。

复用原 `gradient.optimize/choice/journal_check` 的数学与完整 recurrence 验证，但不安装原全局 GradGate。原 optimizer task 的 `open_loop_model_only` 只表示单个优化节点内部的想象 rollout，不代表外部控制器无真实反馈。新 NativeGradient 使用实例级原生模块观察器、实际 `torch.autograd.grad` 与 raw leaf callback，保存模块输入/输出、packet/context、反向及完整资源事件。

## 标量 codec 接缝修复

封存 `signal.blob` 的 `np.ascontiguousarray` 会将运行时0维 scalar 编码为 shape `(1,)`。本 driver 在运行处硬核验 `loss.ndim==0`、fresh raw leaf、`scalar_outputs=1`，保留原 AD entry ABI；saved checker 仅接受 loss blob `(1,)` 并用 `[0]` 取值，raw leaf 和梯度仍必须 `(10,2)`。codec 自身无法区分 scalar 与 singleton vector，标量语义来自 source 绑定的运行守卫，不能声称 blob 单独证明。回归覆盖 codec 实际行为、长向量 loss 拒绝以及 raw/梯度展平拒绝；封存 codec 无改动。

## 双 ledger、严格消费与边界

每满节点136 predict API、32 scalar reverse sweeps；16节点上限2176 predict、512 reverse。action encoder/predictor/pred_proj/native IO、scalar grad/VJP/raw backward 逐操作 intent/return 对账；scalar grad 与 VJP 是同一次 reverse 的不同资源单位，不相加冒充独立计算。未知 FLOPs/backward engine calls 为 null，无等 FLOPs/等 wall 声明。physical/visual ledger 沿用原 P3 明确计量，unknown 不填零。

严格消费重建原 prepare/source/model/history/goal、完整 native journal、优化递推、候选与独立 replay、资源与实际 tensor IO、真实 RGB 预处理及实际 raw/action/history 因果滚动、native reward/success 和 dispatch seal。所有未知 native event/字段拒绝。失败保留部分账与分母，不生成完成或科学通过。

`physical_census` 先验证真实新 report/事件 hashchain、schema 与 contract，之后仅在不变的本地副本中投影旧 transport checker 的 contract/status 常量以复用同一物理资源 ABI；不写旧 report、不生成旧策略 receipt、不宣称采用原调度算法。原物理 caps 与本实现必须精确相同，否则拒绝。

全阶段 cap：1800s，512MiB 输出，64MiB final reserve，200MiB disk floor；CUDA allocated 3GiB/reserved 4GiB。仅 MAIN 显式授权主机 CUDA+native physics，冻结参数，TF32/autocast off、MATH SDP。worker 未执行 neural、physics、生产 prepare 或网络。测试只用 NumPy/RAM接口 doubles 与只读 saved template；不是模型/物理/科学证据。P1/P3/freshness/scientific admission 始终 false，已知根校准不冒独立确认。

## MAIN 命令

在 `/home/gbk/code/LeWorldModel/le-wm-main` 执行；prepare 后不得改本三文件。下面 `PREPARE_SHA`、`REPORT_SHA` 仅替换为实际输出，不预填未来 SHA。

```bash
.venv/bin/python -m experiments.spectral_reachability.agent_p3_feedback_gradient_mpc20_20261028_v1 prepare --output results/agent_p3_feedback_gradient_mpc20_20261028_v1_prepare_MAIN01.json
.venv/bin/python -m experiments.spectral_reachability.agent_p3_feedback_gradient_mpc20_20261028_v1 verify --prepared results/agent_p3_feedback_gradient_mpc20_20261028_v1_prepare_MAIN01.json --prepare-sha PREPARE_SHA
.venv/bin/python -m experiments.spectral_reachability.agent_p3_feedback_gradient_mpc20_20261028_v1 run --prepared results/agent_p3_feedback_gradient_mpc20_20261028_v1_prepare_MAIN01.json --prepare-sha PREPARE_SHA --output results/agent_p3_feedback_gradient_mpc20_20261028_v1_MAIN01 --authorize-known-cuda-and-physics
.venv/bin/python -m experiments.spectral_reachability.agent_p3_feedback_gradient_mpc20_20261028_v1 consume --prepared results/agent_p3_feedback_gradient_mpc20_20261028_v1_prepare_MAIN01.json --prepare-sha PREPARE_SHA --output results/agent_p3_feedback_gradient_mpc20_20261028_v1_MAIN01 --report-sha REPORT_SHA --receipt results/agent_p3_feedback_gradient_mpc20_20261028_v1_consume_MAIN01.json
```

轻量回归：

```bash
.venv/bin/python -m pytest -q tests/test_agent_p3_feedback_gradient_mpc20_20261028_v1.py tests/test_p1_constructive_gradient_mpc_cuda_20261027_v1.py tests/agent_p3_same_feedback_physical20_20261028_v1_tests.py
```

未运行项：真实 prepare、CUDA、physics、生产严格 consume 均留 MAIN。实际 runtime 接线与效能未由 RAM 测试认证；生产错误必须保留，不改阈值、不按结果补跑。
