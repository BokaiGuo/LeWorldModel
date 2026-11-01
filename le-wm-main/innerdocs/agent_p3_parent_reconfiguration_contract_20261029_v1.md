# Parent-native R8：原生成器接线准备（冻结 v1）

## 结论与本轮范围

真实 reconfiguration 数学和生成代码已存在；缺的是**当前 parent-native 合同下的 R8 生成语义与原生计量适配**，不是缺少一次审批或一个新 seed。不能把 progress 标为 R，也不能把旧两阶段编译器的十候选直接说成原 policy 的八候选。

保留 `core.choose`、阈值、宽8、144队列分母和现有硬预算，可以实现一个明确命名的新 R 接线版本，并在预算不够时原子拒绝；但目前不能承诺“不改生成器语义、完整执行原宽深、又保证全部落在 progress 外壳预算”。原 policy 只选择类型/时域/宽深，没有定义 R 的 bank、前后缀切分、评分和无 witness 行为。下述几个选择须先事前固定，不能从已见结果反推。

本轮只新增此文档；不建猜测性的 adapter，不导入旧神经模块，不读取物理 outcome，不运行 NN/physics/fullstrict。现有 refusal 是工程未接通，不是 P1 或 R 的理论负结果；full-policy/C3/P1 均不因此成立。

## 可直接定位的原实现

下列路径相对 `le-wm-main/experiments/spectral_reachability/`；函数锚点对应本轮只读源码。

| 源码/接口 | 已真实实现 | 与当前 R8 的精确差项 |
| --- | --- | --- |
| `preflight_pusht_d2p_p1_reconfiguration.py:170` `compile_reconfiguration(model,frames,history,goal_rgb,*,transform,device,proposals,prefix_horizon,suffix_horizon,damping,sketch,seed)` | 原 RGB context/goal 编码；冻结 uniform bank（index0 为零）；每 prefix 真实推进、post-prefix J、DLS、bounded 非线性 suffix；严格 nongreedy witness | 每 token 仅2自由度、五步重复；tanh/logit 参数；随机 sketch；内部 RGB 重编码；返回一个 prefix 或 None，不是八个 `core.request` 响应 |
| 同文件 `:144` `_reachability` | nominal、vectorized Jacobian、torch SVD、reach value、真实 bounded suffix 费用 | 不是当前 identity192/raw10H 的 native signal ledger；返回 scalar/J 摘要，不保存全部 native IO/context |
| `pusht_full_packet_compiler_20261027.py:60` `spectral_bank(J,nominal,goal,projection)` | constant2/full10；sign-fixed SVD；零、root-DLS、两半径×两模态×正负，共10候选 | 仅支持 J 列数2或10，非 H3/raw30；不是8；不能静默删零与 progress 两项 |
| 同文件 `:104` `compile_arm(*,arm,emb,history,goal,q,jac_full,origin,transition,model,device,counter)` | 真 full10 prefix、post-prefix full J、bounded ridge suffix、独立 whole replay；保存候选 raw/预测/J/评分 | 固定 prefix1+suffix1，共 raw10；选择 `stage+own_reach_value`，不是严格 nongreedy witness；无 R8 request/nativeIO 完整响应 |
| 同文件 `:83` `make_transition(model,device)` | CUDA frozen model；latent `(1,3,192)`、between-action `(1,2,10)`、raw `(5,2)` 的真实滚动 | 内部用 `p0.PUBLIC_ACTION_MEAN/STD`；须与当前 task normalizer 逐字段认证，不能仅凭形状兼容直接复用 |
| 同文件 `:28` `bounded_ridge` | 原有 L2-ball ridge，NumPy SVD 和 multiplier 求解 | 是可复用数值核，不等于已经有当前 R8 的 action/support/objective authority |

严格 nongreedy 原规则：非零候选 `immediate > zero.immediate` 且 `reach < zero.reach`，在 eligible 中按 `(reach,index)` 选一个；无 eligible 则 None。不得新增 epsilon 或 fallback。原默认 proposals16/prefix1/suffix1/damping1e-2/sketch8 与 full-packet 的 damping .002/L2 cap 不是同一个合同。

full-packet 的十候选选优与严格 witness 是两条不同路线。其8个 spectral proposals 确实存在，但“取这8个，另付零/progress控制”是一个可命名的新 bank 合同，不是未改原生成器的等价调用。

TwoRoom/Reacher 也有真 R 生成器，但动作范围、模型、normalizer、projection/阈值不同；不能拿来补 PushT 权威。`reacher_d2p_program_dispatch.py` 的 `model_only_reconfiguration_else_same_bank_progress_v1` 明确带 progress fallback，不能迁入当前拒绝语义。

## 当前接缝：不只替换一个 generate

`agent_p3_runtime_contract_20261028_v1.py:271 choose` 必须原样保留：高 chi 选 R；horizon 来自原 schedule；宽由原 schedule 返回；非 root `program_type=composition`，但 `extension_type` 仍可能 R。不能因 program_type 是 composition 就把 edge 当 progress。

`agent_p3_parent_native_adapter_20261029_v1.py` 的三个显式关口目前均只支持 progress：`:38 validate_parent` 的历史 edge kind、`:63 atomic_queue` 的类型拒绝、`:137 generate` 的类型拒绝。因此新增 backend 不能独自开启 R；需独占新 adapter/orchestrator 和对应 saved consumer，保留旧拒绝版本。不得 monkeypatch 或把 `producer_sha256` 改成旧模块。

可保留的公共接缝是：`core.request` 完整 decision、root authority 与 parent task 双身份映射、`core.checked_path` 全 context/raw 链校验、原 `model_cost`、原宽深/child stop、原 whole/segmented replay。现有 `ParentNativeAdapter.signal` 已提供同 parent H1/H3，不能借 root J 作为 post-prefix J。

旧 RGB R helper 的 incoming3 action ABI 不等同当前 between2；可以在新版本明确转换，但必须证明 outgoing 三 token 的时间身份，不能编造最老 token。更小且不新增 encoder 的路线是从真实 parent latent3/between2 接入 numerical kernel，不调用旧 RGB collector/main。

## 最小可执行下一 slice：先冻结这些值，再写一个 bank evaluator

建议优先接 **full-packet 数值核的新 parent-R variant**，而非调用会重编码 RGB 的 standalone main。以下不是已批准的运行参数，也不是已经实现的 adapter。

1. 固定 R bank authority：选严格 witness 的 uniform bank，还是 full-packet spectral bank；若选 spectral 外8，明确另付内部 zero/root-DLS 两控制及其身份，不把10次计算藏成8次。随机 bank 必须固定 seed、设备 RNG 与全部实际 raw bytes，不凭 hash 摘要重造。
2. 固定 edge 长度和诊断长度：原 request 要求 edge 恰为 `5*h` raw（h=1或3）。必须明确 prefix 是否占整个 edge、suffix 仅用于 reach 诊断，或如何把 prefix+suffix 分配到 h；不能把旧 H2/raw10 硬塞给 H1/H3。H3不能自动按 remaining-fit 缩短；原 full10 核不支持 raw30，H3 泛化是新实现。
3. 固定 parameter chart/support、normalizer 与原目标：原 tanh 与当前 raw/chart 不能混称；task goal192/Q 与旧随机 sketch 不能混称。选择完整 identity192 会形成新 R variant，须逐字段记录，不能宣称复现旧 sketch witness。
4. 固定“八个请求返回什么”：每个 prefix 返回完整可执行 edge/path 与 post-prefix诊断，还是内部选一个 witness 后仅一个有效输出。原 `core.choose` 没有定义这点。若要求8个 executable edges，必须定义不合格槽的显式状态及是否可进入原 modelQ beam；不能复制 winner 填满8或把零控制默认为合格 R。
5. 无 witness 必须保留显式 `no_reconfiguration_witness`；无 budget 为 `resource_cap_before_node_queue`；unsupported 为类型拒绝。都不自动回退 H1/progress，不冒充模型实验失败。

冻结后最小实现可提供 `evaluate_bank(parent_task, requests8, bank_authority, native_factory, ledger, sink)`：

- 输入完整 root binding、parent latent3/between2、原 prefix edges/raw/path、完整原 decision、同 parent H1/H3 receipt，以及以上 bank/长度/评分合同。
- 分组原子预检8个 request 身份和完整资源向量（含内部 controls、后验 J、replay reserve），通过后才派发。cache key 包含完整 root/parent/normalizer/Q/协议/source 身份。
- 每个 prefix 实际推进，保存 full context；从**该 prefix 输出 context**建立新 native task，求 own J/nominal，做原绑定 solver 与 bounded replay。记录 request→parent→prefix→J→suffix→edge 全数组、事件和来源映射。
- 输出完整8槽及内部 controls；core 按已固定资格合同接收真实 edge，原 modelQ 排序不暗改为 reach 排序。所有模型输入仅来自这些上下文/动作，不包含 evaluator state/reward。
- 新 strict 从 saved bank/IO/事件重建全 path、评分/资格、队列和预算；最终 whole/segmented replay 从完整 root 开始。source/model/precision before-after 闭包沿用原生实现，不增加审批 manifest。

## 预算：不能只看剩余 predict

以下为前向调用推导，不是已经测量的 native receipt。predict API、sample rows、action_encoder、AD primal、grad API、scalar VJP、J元素、SVD、solve、compile/metric、replay、wall 必须分轴；未知不能填0。vectorized Jacobian 的一个 API 不等于实测192次 scalar backward。

| 路径 | 可由原源码推导的基本费用（不含外部 root signal） |
| --- | --- |
| 旧严格 witness，B个 prefix、prefix P token、suffix S token | B×(P+3S) predict：prefix、nominal、Jacobian primal、bounded suffix；另 context/goal encoder、每 proposal sketch-J/SVD/solve；无独立 whole replay，不能称已有当前 replay 证书 |
| full-packet 原10候选、P=S=1 | 每候选 prefix1+nominal1+AD primal1+bounded1+独立whole2=6 predict，共60；post-prefix J10；NumPy SVD12（bank1+root ridge1+suffix ridge10），ridge solve11；外部 supplied root J另付 |
| 新8 prefix 后只求 identity192 的单 S suffix J | prefix8P，nominal+AD primal16S，J8/1536方向；若有真实 bounded suffix另8S；实际 scalar callback数须由新 native ledger认证 |
| 新8 prefix 后完整 H1/H3 pair | prefix8P，signal64 predict、16J/3072方向；bounded suffix、内部控制、重放另付；不能把它当 parent 已付一次 pair |

旧 full-packet 输出的 `jacobian_backwards=10/scalar_output_vjp_directions=1920` 是声明型汇总，不足以替代当前 native callback/IO 证据。旧 `.phase` instrumentation 的 planning/evaluation 也不能直接当新完整资源 census。

当前默认 fixed/adaptive predict cap 为114/562；这是 progress 外壳，不是 R 费用认证。原 `default_caps` 的 J/SVD/solve 等轴同样可能先耗尽。原子预检必须加入 R 实际最小向量与最终 replay reserve；不能只检查预测数、借4槽抵8槽、退款或padding。若 caps 不变，允许诚实停止，不保证完整原 policy 树。

诊断 suffix 若不成为 executable edge，也必须付费并标明 lookahead，不得冒充物理动作。最终两个 replay 的基础 reserve 为 `2*(已有prefix token数+新edge token数)` predict，另 replay_calls=2；完整历史和所有失败尝试均保留。

## 新 adapter 必需 RAM/保存文件回归（本轮未运行）

- 原 choose 返回 R8、原阈值边界/child depth 停止不变；非 root composition 标签与 R extension 分开认证。
- root/parent 双身份及所有 prefix edge 链，latent3/between2 与 raw bits 完整；同终态不同 history 不可缓存碰撞。
- child/post-prefix J callback 随真实 context 改变；拒绝 root/stale J；H3/raw30 不被缩成 full10 或重复 token。
- 8槽原子分配，7槽余量拒绝；各资源轴不足分别拒绝，派发前零新增调用；内部 zero/progress 控制也计费。
- 严格 witness 的大于/小于、同分 index、无 witness/None；无 progress fallback、无 winner复制、无静默4候选。
- 真实生成/replay路径一致，whole/segmented全链；突变 normalizer/Q/projection/bank/source/事件/数组逐项拒绝。
- vectorized与scalar AD计量分别保存；失败 partial、未知资源和原144分母保留；源码/模型/精度闭包无改变。

## 本轮源身份与交付边界

只读核对的 SHA256：

- `preflight_pusht_d2p_p1_reconfiguration.py`: `57bfd63b078b8df26fefae75786156ddb611c271b2630cacdfadbf179008e7c7`
- `pusht_full_packet_compiler_20261027.py`: `f359e7b21d1970f95f065dbee206d2dc9359c2eaff7ddbcd7611e07686c6cf2a`
- `agent_p3_runtime_contract_20261028_v1.py`: `72f076106fb8f6c2aa73386c06fecb912f033e81a5108aa1f36885bd0342c2bc`
- `agent_p3_parent_native_adapter_20261029_v1.py`: `150a1c4189a68f9b7c71479e5aa6980cea490fdeed0d5524b7bfad0cc78c2558`

没有新 adapter/tests/运行CLI；原因是缺上述 R bank、edge时域和资格语义，不能自行把两个不同原生成器拼成“原policy”。最小下一实施决定是先固定这三项，然后实现该 evaluator 加新 R-aware orchestrator/strict，而非再次生成 manifest 或重跑现有 progress。
