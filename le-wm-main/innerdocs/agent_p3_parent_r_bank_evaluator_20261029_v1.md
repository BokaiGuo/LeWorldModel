# Parent R8 explicit-bank evaluator v1

## 边界

纯 NumPy/RAM 依赖注入 kernel，不接生产 orchestrator，不加载权重，不执行神经/物理，不改 policy、阈值、原队列或旧源码。只接受 `callback.kind == 'RAM_FIXTURE'`；即使 bank 提供认证指针也不能开放生产。不是 P1/full-policy/C3 证书。

这是明确的新 variant：调用方提供八条完整 raw edge 和独立 zero control，kernel 不制造 bank，不把 progress bank 冒称 R。生产 bank/chart 仍需另行冻结。eight raw 可以相同，但 kernel 不生成副本、不复制 winner 填槽。

## Python ABI

模块：`experiments.spectral_reachability.agent_p3_parent_r_bank_evaluator_20261029_v1`。

`evaluate(*, task, parent, prefix_actions, prefix_path, bank, zero, horizon, authority, callback, caps, used, verification_reserve, sink) -> evaluation`

- task 是原 `core.validate_task` 合同，goal192、Q 必须逐值 identity192，原 frozen mean/std 和 action-domain binding；parent 是全 latent3/between2 context。完整 prefix actions/path 必须从 root 重建精确 parent，空 prefix 必须原 root，不能只传 terminal192。
- `horizon` 严格 int 1/3；bank 为 float32 可精确表达的 `(8,5*h,2)` 原始 raw，范围 [-1,1]；zero 显式 `(5*h,2)` 全零。长度不截短、不padding。
- authority 精确字段：`source_sha256, bank_sha256, zero_sha256, root_task_sha256, parent_context_sha256, goal_sha256, Q_sha256, normalizer_sha256, action_domain_sha256, bank_authenticated, authentication_sha256`。前九项为 SHA；最后一项未认证时可 null，认证声明则必须有明确 SHA 指针。所有内容 binding 重算；`source_sha256`/认证指针仅由调用方提供并保留，kernel 不读证书、自签或认定其真实性。bank/zero digest 使用原 core canonical JSON，normalizer digest 对 `[task.action_mean,task.action_std]`。
- caps/used/reserve 均必须完整列出 `AXES`，不缺轴、不补0、不接受未知字段，值为非负 int。reserve 是留给外部 verification 的预算，不在本 kernel 执行或虚记为已花。
- `sink(event)` 逐事件接收 deep copy；返回值另保存全部事件。caller 负责保存返回对象；无生产文件写入入口。

callback 接收完整 request，含 operation/index、root-task/outer-parent/input-context SHA、完整 task/context/authority、inputs 和本次 tariff。响应必须为 `{request_sha256, resources, payload}`，完整 resources 必须严格等于本次 tariff，不允许少报或隐含新轴。

| operation | inputs | payload 与校验 |
| --- | --- | --- |
| edge | 原 bank/zero 的全部 raw actions | `{path}`，每5步一个完整 context；用 `core.checked_path` 校验三 latent 移位及精确归一化两 token 链 |
| signal | horizon=1、显式 raw_zero | `{nominal_path,jacobian,input_context_sha256,jacobian_chart}`；nominal path 由 postcontext 推进零动作；J 为 float32 `(192,10)`，输入 SHA 必须该 postcontext |
| suffix | bounded raw5、lookahead_only=true | `{path}`；再次精确校验原 full3/between2 链 |

signal 的 chart 字符串必须精确等于 `contract()['jacobian_chart']`。每个 proposal/control 都有自己的 postcontext request，不借 root J；RAM 例子的 J 确实随 prefix 改变。该身份检查不能单独证明任意外部 callback 的数学正确性，所以不可用于生产认证。

## 数值语义及选择

处理顺序固定 zero、candidate0..7。各自完整 edge replay 后计算原 identity-Q immediate MSE；在该 postcontext 求 H1 nominal/J，调用原 `pusht_full_packet_compiler_20261027.bounded_ridge(J,goal-nominal)`。原源码 SHA 固定 `f359e7b21d1970f95f065dbee206d2dc9359c2eaff7ddbcd7611e07686c6cf2a`，dispatch 前核验；damping .002、radius sqrt(5)*.75 不调。

suffix 为十个独立 logits，`tanh(float64 ridge.params).astype(float32).reshape(5,2)`；零点 logits 导数与 raw 导数一致。真实 replay bounded suffix，同时保留 nonlinear cost，但资格使用原 ridge 的 `own_reach_value`，不以 nonlinear outcome 换 score。该 chart 是本新 variant 的明示定义，不声称复刻旧 sketch/random-bank 生成器。

严格资格：`immediate > zero.immediate && reach < zero.reach`，无 epsilon。保留所有8槽，选择 eligible 中最小 `(reach,index)`。无资格为 `no_witness`、selected=null；不返回 progress/zero fallback。suffix 仅 lookahead，不进入 edge、prefix 链或可执行动作。selected_index 仅诊断，不证明 reconfiguration 物理收益。

## 资源与失败

`minimum(h)` 给完整8+zero最低向量。一次 edge 收 h 次 predict、一次 H1 signal 收2次（nominal+AD primal）、一次 suffix 收1次：H1 合计36predict、H3合计54predict。所有独立 forward 均 sample=1；每次 context_tokens=3，action_encoder/predictor/pred_proj 各轴同步列明。

全组9 J、9 AD primal、17280 J元素、1728 VJP方向/grad API/raw-backward callback、9原 bounded solve/array SVD、9edge compile、18 metric、18 replay（edge和suffix），encoder/projector/model SVD/schedule 为本 RAM 合同不请求。zero 不是免费对照。这些是**规范化 RAM callback tariff**，不是已经测量的 CUDA/native census；实际后端可能采用不同AD实现，不能拿这份计数给 vectorized AD签证书。FLOPs、cuda_ns、backward_engine_calls 保留 null，未宣称等算力。

先检查所有轴 `used+minimum+verification_reserve <= caps`。任一不足即 `resource_cap_before_bank`，8槽和zero均 not_requested，0callback、0新增费用。通过后每个 operation dispatch 前收费，成功校验才计 completed；不退款、不重试。原 bounded solve 为内部 operation，也保存 dispatch/returned/validated 与收费。

输入/身份/预算 schema 错误在派发前抛 ValueError。callback异常或返回校验失败时停止全组，返回 `partial_failure`、已完成槽、当前 failed 槽、余槽 not_requested、原返回事件、错误和 attempted/completed；不在 partial 数据上选 winner。sink 本身必须可靠：其异常上抛，不伪称持久记录成功。没有 fullstrict 或生产消费证书。

## 交付与 RAM 验证命令

从 `le-wm-main`：

```sh
.venv/bin/python -m pytest -q tests/test_agent_p3_parent_r_bank_evaluator_20261029_v1.py
.venv/bin/python -m experiments.spectral_reachability.agent_p3_parent_r_bank_evaluator_20261029_v1 --help
.venv/bin/python -m experiments.spectral_reachability.agent_p3_parent_r_bank_evaluator_20261029_v1 --contract
```

测试覆盖 H1/H3 全8+zero、真实不同 postcontext J、原 solver逐值复算、完整history/raw链、严格无witness/稳定tie、所有资源轴及reserve原子拒绝、未知缺轴拒绝、早/晚callback失败不退款、身份/chart/path/资源突变拒绝、冷子进程无 torch/gymnasium 导入。fixture 为显式二维 RAM 动力系统嵌入192维，不是模型或物理结果。

首轮63pass/1fail为fixture parent/prefix_path共享引用使突变同时改变两者，触发比预期更早的正确链拒绝；测试已改deepcopy隔离，未放宽production校验。最终实际回归 **69 passed in 11.46s**；`--help` 独立执行通过，冷导入及 `--contract` 由回归实际执行。三件套 SHA 在冻结交付消息中给出。

未完成：生产 bank/chart authority、R-aware orchestrator、真实 native meter/IO closure、生产 saved strict、CUDA和物理执行。旧 adapter 仍严格拒绝 R，本 kernel 不改变其行为。
