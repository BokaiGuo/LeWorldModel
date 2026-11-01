# P3：goal-patch Q 全链一致编译器规格

日期：2026-10-27。状态：**供 main review 的新实现规格，不是已实现/已 prepare/已执行协议**。

本次只新增本文，不修改 sealed 源码、输入、结果、门槛或总账；不运行模型、AD 或物理。后续实现必须独立新版本，并在任何新 CUDA 结果前封存源码、测试、参数、预算和输入。P1/P2/P3/P4/P5 admission 均不因本文打开。

## 1. 问题与证据边界

当前有两种不同实验，不能合称“新 Q 已贯穿编译器”：

- v4 用旧 Q 计算有限 H2/H3 residual signal、分配搜索预算，再对随机 logit-ball 候选评分。
- goal-patch v1 构造新 Q，对 sealed v4 uniform/dynamic 候选的完整并集重评分。它不生成候选、不重新求 Jacobian，也不调用自己的 `compile_action_response` 生成并执行 ridge 提案。

因此下一步应检验：**在固定模型、根、动作域和前向预算下，新 Q 是否能通过 action-response/residual 生成更合适的候选，而不只是改变旧候选排序。** 这不是已经确定的根因修复，也不是官方算法复现。

main 已报告两对 same-H2 新减旧 pose 为 −57.8736070612、−3.0984134738，另两对 mixed H2/H3 保留不混比；这些数值本次未独立复算，不作为参数选择依据。独立 FD JVP 相对误差约 3.64e-4、5.20e-4 属于另一路 native C1 局部导数证据，不能认证本编译器 H2/H3 Jacobian 或物理有效性。

此前 saved-only 几何 review 见 [native C1 接触几何](p5_native_c1_contact_geometry_review_20261027_v1.md)：C1 后缀未接触物体是该具体轨迹的证据，不据此添加接触损失、改变目标、动作尺度、根或 horizon。

## 2. 只读源码快照与精确接点

下表位置均相对 outer repository；行号对应本次 SHA 快照。实现应新增函数/模块复用可兼容纯函数，不能在原文件上修改。

| 源码接点 | 当前行为 | 新实现需要的连接 |
|---|---|---|
| `le-wm-main/experiments/spectral_reachability/p3_horizon_residual_budget_20261027_v4.py:29` `contract` | 两臂、共享 10 predict、每臂搜索 60、上限 142/root | 独立新合同声明 metric×schedule 四臂与新 census；不能沿用旧 142 上限 |
| 同文件 `:51` `residual_signal` | `QJ_H` 的 SVD 与 `Q(goal−nominal_H)`，但丢弃 U/V | 同一 active Q 计算响应、残差、chi；纯数组谱分解结果需留存以复核 |
| 同文件 `:63` `allocate` | uniform 5/5；dynamic 用 chi2/chi3 | 每个 metric 独立计算 signals，再把其自己的 signals 传给 scheduler |
| 同文件 `:74` `pool` | zero 后全随机，与 Q 无关 | 新 bank 保留 zero，第二槽改为该 Q 的 bounded-ridge 解；后续随机槽固定复用同一 pool 索引 |
| 同文件 `:84` `NeuralCap` | 仅 shared/uniform/dynamic，预 dispatch 计费 | 新四臂 role、根总上限、attempt/completed 与禁 encoder；不能原样容纳四臂 |
| 同文件 `:138` `compile_cuda` | 在零 joint logits 上 fresh 两个 raw Jacobian，之后用 packet 旧 Q | 两个 raw Jacobian/nominal 共享一次；其后显式分叉 Qold/Qnew 的谱信号与 ridge，不重复模型调用 |
| 同文件 `:164` `search` | 随机 bank、packet Q 评分、稳定 argmin、整段/分段 double replay | 接入 metric-conditioned bank 和显式 metric binding；所有候选、排序、回放都用该臂同一 Q |
| 同文件 `:200` `prepare` / `:225` `validate_prepared` | packet projection 必须逐字继承旧输入 | 原 validator 只认证原始输入，不用于认证被替换 Q 的伪旧 packet；新增 schema 分离 source packet 与 active metric |
| 同文件 `:317` `selected_actions` | 重建旧 pool、旧 Q、旧两臂 ledger | 新 exporter 重建新 bank、四臂、metric id、budget；不能冒充旧 exporter 输出 |
| `le-wm-main/experiments/spectral_reachability/p3p4_goal_patch_metric_20261027_v1.py:50` `catalog` / `:81` `basis` / `:91` `compile_metric` | goal-only RGB 干预、nuisance 除去、native CLS projector Q | 保持原定义及 cutoff/roundoff floor；读取已封存 metric 或单独提前构造，不从 endpoint 重建 mask |
| 同文件 `:107` `native_cost` / `:112` `compile_action_response` | 可复用的 Q 代价与数组 ridge helper；后者目前不在执行链上 | 将 helper 返回的 proposal 实际纳入 bank，在完整 rolling context 上做非线性评价 |
| 同文件 `:128` `select_common_bank` / `:189` `compile_cuda` | 只重评分旧并集；predict 被禁止 | 只能作评分隔离对照/metric 构造来源，不是新的 Q-consistent search runner |
| `le-wm-main/experiments/spectral_reachability/pusht_full_packet_compiler_20261027.py:28` `bounded_ridge` | 精确 L2-ball ridge，内部一次 SVD | 原样复用 `.002`，传入 H 缩放的 joint cap；保留 solve receipt |
| 同文件 `:60` `spectral_bank` | 只收 2/10 维 Jacobian，H1 两半径十候选 | **不能原样传入 20/30 维**；本规格主合同不调用它，不隐式扩展其维度/半径/候选数 |
| 同文件 `:83` `make_transition` / `:104` `compile_arm` | 前者为无 detach rolling transition；后者为 detached-prefix conditional suffix 编译 | 只复用前者；不能混入后者 stage-cost/own-reach objective、每 prefix AD 和额外前向 |

SHA256：

```text
27f8b7f1cabed21d664ded40be0e0640aed9b0367bdfe2c39d377f4c482463c0  p3_horizon_residual_budget_20261027_v4.py
f40fca0ccdf5f8f51d68174a7feca21192031816163ba2cc9bce508febb18957  p3p4_goal_patch_metric_20261027_v1.py
f359e7b21d1970f95f065dbee206d2dc9359c2eaff7ddbcd7611e07686c6cf2a  pusht_full_packet_compiler_20261027.py
```

## 3. 不变输入、坐标与 Q 定义

仍为原四根，通过 `root_sha256` 关联，不按 report 数组序 join。另封存 root→seed index 映射；seed index 是随机数合同，不替代 root 身份。每根保留原 `root_context`：三帧 native predicted CLS history `[3,192]` 和两段已归一化 action history `[2,10]`，原始 goal0 `[192]`、旧 Q `[192,192]`、权重及模型配置完整哈希。

H∈{2,3}；joint time-major logits `u:[10H]`，`raw=tanh(u).reshape(H,5,2)`。每 packet 五个二维 raw controls，不是 H 个动作；H2/H3 分别 10/15 controls。半径 `R_H=.75*sqrt(5H)` 约束整条 logit 向量，**不是**逐 packet ball、raw L2 ball、逐坐标硬裁剪，也不增加 `/100`。每次 transition 走已有 public action mean/std 归一化，三 token full context 滚动；AD 路径不 detach，中途不置换真实图像/潜变量。

新 Q 必须仍是 metric v1 的 goal-only 原定义：固定 RGB catalog，native projector(CLS) 192 维；nuisance 来自四个 legacy goal 差、original−clean、四固定角 sprite−clean；object cell ablation 响应先去 nuisance，再取列空间投影。原 `basis` 相对阈值 1e-6 与 source-scale roundoff floor 不变。task rank 零严格拒绝，不换 mask、阈值或退回旧 Q。不是 patch-token mean，不学习 adapter/读出/权重，不用实际 endpoint/state 建 Q。

代价统一为 `C_Q(z)=||Q@(z−goal0)||²/192`。goal 始终原始 goal0，不能换 clean goal、goal state 或逐候选 goal。Q 为对称幂等 projector；不按 task rank、trace 或旧/新 Q 经验尺度再归一化。旧/新 Q 的数值代价不直接相减冒充同量纲效用改善；交叉评分须分别标注 metric。

新 manifest 至少拆分 `source_input_sha256`（含原 Q）与 `active_metric_id`（root、Q 数组 shape/dtype/bytes、goal0、catalog、metric 源码/协议/报告哈希、权重绑定），再计算 arm input id。不把原 `input_sha256` 当作新 Q 已认证的证明。

## 4. 主提案：两槽 root spectral-ridge + 固定随机尾部

这是明确可实现的 **root SVD-ridge proposal** 合同，供 main 接受后写新版本；不是 iterative gradient MPC，也不声称完整复用旧 H1 spectral bank。选择这一最小改动是为了隔离 Q 的编译作用，同时保留 v4 每臂 60 search predict，而不凭空塞入额外前向/AD。

对每根和 H，fresh 共享计算：

```text
n_H = terminal_H(0)
J_H = d terminal_H(u) / du at u=0       # [192,10H]，native raw terminal，不含 Q
K_Q,H = Q @ J_H
r_Q,H = Q @ (goal0−n_H)
chi_Q,H = ||r−Proj_col(K)r||² / ||r||²
delta_Q,H = argmin_{||d||<=R_H} ||K d−r||² + .002 ||d||²
```

chi 的零 residual/零 rank 规则保持 v4：前者 0，后者在 residual 非零时 1。ridge 原问题**未除以 192**；只在 receipt 的 cost/penalty 展示时除以 192。若改为除以 192 的 Gram/rhs 再保持 damping `.002`，等效正则改变，属于实现错误。ridge 用所有奇异值的软正则解；不要把 chi 的 rank 截断擅自施加到 ridge。

原 `compile_action_response(packet,metric,J,n,H)` 已能给出 signal 和 delta，不调用模型：每 Q/H 两次 SVD（signal 一次、ridge 一次）。新 runner 可直接复用并另外保存 `K/r`、proposal logits/raw 和数组哈希；若需要完整 ridge receipt，可显式调用同一 `bounded_ridge` 一次代替 helper 内部 solve，不能两者重复调用而漏账。若以后改为共享单次 SVD 的 factored solve，需独立实现/测试/版本和新的精确 SVD 账，不能宣称现 helper 只有一次 SVD。

每 Q/H 候选按以下确定顺序：

1. index 0：zero；即使 nominal 已计算，也按候选重新完整执行，计 H predict。
2. index 1：`tanh(delta_Q,H)`，reshape 为 `[5H,2]`，实际执行完整非线性 rollout。
3. index k≥2：**原** `pool(stable_seed_index,H,N)[k]`。PCG64 seed 仍为 `20261027+100*seed_index+H`；原随机 index 1 被替换而不是把随机尾部前移。跨 Q 同索引随机尾部动作完全相同。

无 duplicate 去重、无 rank-zero 时补充搜索、无额外 line-search/FD/梯度更新；ridge 恰好零或与其他候选相同仍保留并付费。两种 schedule 对同一 Q/H 使用同一 bank 前缀；不同 Q 的 index 1 可以不同。保留每候选所有 logits/raw、terminal/full context、active metric cost、attempt/completed 及绑定。

这是用谱分解的响应与残差**生成并优化一个线性 trust-ball 候选**，再经非线性模型筛选；不是仅排序，也不是证明该线性解在非线性模型上最优。显式 ±奇异向量候选、多半径 bank、conditional suffix refinement 或多步优化不在此主合同内；如另提，应另版本预声明模式/重根奇异子空间 tie 规则、槽位与额外预算，不能直接套旧 `spectral_bank`。

## 5. metric×schedule 与 matched 预算

主提案四臂为 `{old_Q,goal_patch_Q}×{uniform,dynamic}`；**旧 Q 也运行同样的新 ridge bank**，而不是将旧随机实验拼作 matched 对照。这是 metric 与 scheduling 的 2×2 描述性消融；新旧历史实验只作旁证。四臂顺序在 prepare 前按固定 root 映射平衡封存，不按结果变更。

每 Q 自己的 chi 决定 dynamic 分配，公式仍为：

```text
gain=max(chi2−chi3,0)/max(chi2,1e−12)
b3=floor(10*(.25+.5*gain)+.5); b2=10−b3
N2=3*b2; N3=2*b3
2*N2+3*N3=60
```

uniform 为 b2=b3=5，即 15 个 H2 +10 个 H3。合法 chi 使 dynamic b3∈[3,8]，故 N2≥6、N3≥6，两槽 bank 可行，不改变分配门槛。每臂依 `(active_Q_cost,H,index)` stable argmin；记录 winner 的 H，不能强制更换 H 来凑物理可比性。

| 项目 | 每根 fresh 主提案预算 | 四根预算/性质 |
|---|---:|---|
| 共享 nominal + AD forward | H2：2+2；H3：3+3，共 10 predict | 40 predict；两个 metric 共用同一 raw J/nominal |
| 四臂候选搜索 | 4×60=240 predict | 960 predict |
| 每臂 winner 整段+分段独立回放 | 每臂 2H，四臂最多 24 predict | 最多 96；不得 padding/refund |
| fresh planner predict 总量 | `250+2*sum(H_winner_arm)`，最多 **274** | 最多 **1096**，不是旧 568 |
| action_encoder forward | 与 predict 分别计数；上限 274 | 上限 1096；三 action context tokens/entry |
| predict+action_encoder module entries | 上限 548 | 上限 2192；不是 FLOPs |
| raw Jacobian | 2，形状 192×20 /192×30 | 8 次 materialization；不按 metric 重复 AD |
| 数学 VJP 方向 / jacobian API | 384 /2 | 1536 /8；实际 backward engine 次数未知须单列 |
| Q-conditioned signal+ridge SVD | 2 Q ×2 H ×2=8 | 32；不含 metric 构造、export 审计 |
| cached-context planner visual encoder | 0，禁止调用 | 继承 history/goal 生成成本未知，不是零 |

每臂独占 cap 66=60+最大 6 回放；shared cap 10；root cap 274。正常完成实际总数为 266/268/270/272/274 中之一，取决于四个 winner 的 H。AD forward 已在共享 10 内，VJP/backward 是另外开销，不能称总预算匹配。两 metric 在同一 schedule 下候选前向预算相同、raw AD 共享、每臂回放上限相同；实际回放若 H 不同则不同。

Qnew 构造成本单列：若复用已封存 metric，则新 runner 不重复 encoder，但绑定其 goal vectors/catalog/Q 与原构造 journal，记作 inherited，不是免费。若另行重建，则独立先封存并计每根 `6+K_goal_cells` 次 encoder 和 existing-projector forward、两次 metric SVD，再供 planner 读取；不可在 planner visual-encoder 禁令下暗跑。原 metric exporter 对旧证据的 saved-array SVD 复核、新 exporter 数组审计、读盘、CPU ridge 运算、CUDA 时间/显存都分别记账，不并入 predict 计数掩盖。不能因某个已有 panel 恰好每根 20 张就把一般合同硬编码 20。

## 6. 评分公平性、H 分层与物理边界

uniform 两 Q 使用同样 H2/H3 候选数，最适合隔离 metric 的编译效应；dynamic 两 Q 的 H 分配本来可能不同，应标记为 metric+scheduler 联合效果。额外对四臂保存的全体候选可做双 Q 交叉评分，纯数组、不重新择优 dispatch，但不能把这个不同来源 bank 的交叉排序称“相同候选”的隔离试验。

全局 winner 的 physical 比较必须预先保留四根完整 census：same-H2、same-H3、mixed-H、partial/failed 分开，不按有利物理终点筛根。same-H winner 子集也只是内生选择后的描述，不是固定 H 的确认性检验。若要严格固定 H 效果，需在结果前另行定义 H2-only/H3-only arms 及其搜索/回放预算；不能事后取各 H 最佳、免费增加回放/物理，或把 mixed-H 补齐到 H3。

新 exporter 必须输出 root、arm、metric id、goal、原/active Q、source/active input id、模型绑定、winner H、raw actions、全 history context、预测代价、journal/报告/selection 哈希。整段与逐 packet replay 必须与 candidate 的**完整**预测 context 精确一致，固定同一 CUDA backend（包括 SDPA/math 设置）；不可用 endpoint-only allclose 放宽原门槛或切 backend 后忽略误差。

未来真实终点编码必须使用相同 frozen projector、相同 goal0 和该臂 active Q；可并列保存双 Q scores 但不能重选动作。实际 pose/official/state 指标是外部评估，不回流编译损失。

现有 `p3p4_goal_patch_replay_20261027_v1.py:43/154/231` 与 `p3p4_goal_patch_physical_20261027_v1.py:41/86/163/180` 是绑定、double replay、分层和终点编码接点；其固定两臂、已选 H、36 replay predict/90 physical commands 是**旧选择的合同**，不适配新四臂。只能新增 consumer/schema，复用兼容 guard/reset/evaluator 纯逻辑，不能修改旧白名单或拿旧 result 追认新来源。

## 7. 实现前置条件与 model-free 测试要求

后续新实现需要先满足以下可执行项，本文不提供虚构 CUDA 命令：

1. main 接受本候选槽位与四臂上限，指定 sealed goal metric prepare/report 路径及 SHA；当前未在本文替 main 选择新的执行 artifact。读取完整四根 goal catalog/向量/Q，不读物理结果选版本。
2. 新模块/测试/innerdoc 实现独立 schema，验证原 context/goal/模型与 active metric；root set 精确一致、唯一且无缺失。用字典稳定身份 join，seed 映射显式。
3. 纯 NumPy/接口 mock 测试：Q 的 shape/有限/对称/幂等与哈希错配；goal 不变；zero residual/zero rank；ridge KKT/ball cap、192 归一化正则陷阱；tanh/time-major/action normalization；确定随机尾部和 duplicate 保留；chi 全边界预算恒等式；四臂 tie/argmin；坏源码/模型/metric/动作/根绑定拒绝。不得使用 CPU 神经模型。
4. 仅接口 mock 测试 instrumentation：预 dispatch 计费、274/root 和 66/arm 独立限制、AD/predict/action_encoder 失败不退款、视觉 encoder 禁止、模型 frozen/backend 设置、所有 partial arrays 的即时挂载和 durable journal、double replay 不符拒绝。这里的 mock 不能认证真实 CUDA AD 或实际模型一致性。
5. runner 对每个 attempted 操作先落盘 intent；J、nominal、K/r、ridge proposal、各候选与每次 replay 有结果即保存。失败保留原 exception/trace 与后续 source/hash/model/hook-cleanup 错误列表，不能 final SHA 校验覆盖首错，也不能只在整根成功后才附数组。缺失臂/根显式 blocked，不自动换 seed/重试。
6. main review 全部 source closure 与纯测试通过后，新独占路径 prepare；源码、metric、模型、上下文、bank policy、四臂顺序和预算冻结，再由 main 决定 CUDA。物理另行授权、另 seal；planner 成功不能代替真实 outcome。

复用 saved raw J 也可先做纯数组编译检查，但必须标 `inherited_derivative`，不能报作 fresh AD、不能把省下的 10 predict 重新分配到搜索而仍称本合同。无论使用哪条路径，派生数据、source closure 与成本 provenance 都要保持完整。

## 8. 本次交付验证

只读核对上述三份主源码 SHA、函数/实际调用链与预算；新增本文。未执行新模型、AD、物理或任何 outcome 参数试探。本文提出的是可审查的下一实验合同，不是模型质量、接触能力、P3 scheduling 优势或 P 阶段准入结论。
