# P2 wall-capped comparison v2：prepare 与记账接口

本版是 **wall-capped comparison，不是严格 equalFLOPs**。v1 三件套保持冻结。实际耗时和实际资源消耗是响应变量，不要求两臂逐位相等；不因 FLOPs/engine backward 未知而阻止这项探索。共同约束是事前封存的时间/资源上限、相同模型输入和固定遍历规则。无 P1/P2 科学准入、freshness、训练来源或 equal-total-compute 声明。

本次只实现 prepare/verify、调度成本和 RAM accounting ABI；MAIN 后续实现实际 CUDA runtime。worker 未生产 prepare，未运行神经/AD/物理，未读取新物理结果。没有改旧源码、结果、Q、normalizer 或 gate。

## 固定条件与共同上限

复用 v1 的真实 saved-only authority 接缝，完整重建原 planner strict receipt 与原 prepare，不信单个状态标志：

- 原 report SHA `56e8b4f2df522f7a0c4cb9bc37ad74bcbc5d4f89e2c5cd601d42cd0eb071d8ef`
- 原 receipt SHA `c491555fda5bb543af69f70987541edb0abb99fa32c559f3aa6190901393017f`
- 原 planner source SHA `3111312477ff3cd9a764dc7d063861c1becaaf58885dc0fb25c2f3ec73d15fd5`

prepare 保留原两 off known roots 完整 slots、task/context/goal/Q、model hash 及所有 artifact/source/model bindings，再绑定本三件套；不读新物理 report。根不按任何新 outcome 选择。固定串行顺序 root0 joint、root0 composition、root1 composition、root1 joint，避免同一策略总先执行；仅2根，非独立统计确认。

每 root×策略独立支付以下共同预算，不能跨根/跨臂借用或缓存免费旧结果：

| 上限 | 数值 |
| --- | ---: |
| 同步 host-monotonic GPU search wall cap | 450s |
| predict / action_encoder / predictor / pred_proj，各自 | 1902 |
| Jacobian | 85 |
| mathematical VJP requested rows | 16320 |
| requested J elements | 489600 |
| requested AD primal rollout steps | 255 |
| root SVD / ridge | 2 / 1 |
| GPU allocated / reserved | 4GiB / 6GiB |
| 新 encoder / projector / physics / training | 0 |

450s 是新比较的固定预算（四 run 合计最多1800s搜索目标，不含加载/strict）；不是根据新物理收益或完成时间调出的参数。模块调用和 AD 尺寸都计账，不用 predict 近似抵消 AD。VJP rows/J elements/AD horizon 在 J intent 记为 requested 工作，失败不退款；它们不是 engine 内部实际 backward 或 FLOPs。实际进入的 primal 和三个模块分别按独立 intent/return 计数；若 AD 中途失败，requested 与 completed 的差额保留，不冒充全执行。FLOPs/engine backward 仍为 null，不影响共同 cap 对照成立，也不推出 equalFLOPs。

## 两种有限、可中断的固定策略

共同步骤：zero full15 nominal3，完整产生 fallback；再 root J primal3 与原 bank 的2 SVD/1 ridge；再按原 index0..41 全部 raw15 seed 各3 forward。两臂都自行付费完成此42项扫描，不继承旧 model scores/J。Q、原完整context3、action history、goal192、原public normalizer与 full30 logit L2 `.75sqrt15` 全部不变。

- **joint**：扫描后，round-major 两轮，每轮依次 seed0..41；每轮在原 root context 联合优化全部30参数。复用冻结 v1 `local_joint` 规则：raw192×30 J 加原连续 tanh chain，半径一半的负梯度点，alpha固定 `[1,.5,.25,.125]`；incumbent+4候选稳定排序，不改善不替换。每轮21 predict、1 J，无新 seed。
- **composition**：扫描后，seed-major `H2(i),H1(i)`。H2 unit 包含自己固定 prefix5 的 replay1，以及冻结原 `old.optimize` H2 的14 forward；H1 unit 为7 forward。两次优化都使用自己先前完整 imagined context，不读真实状态；没有将此称为实际观察反馈。每 seed22 forward、两个192×20/192×10 J。

每个完整 unit 结束后，对该 seed 的完整 full15 witness 更新全局 anytime incumbent；保留 root nominal fallback，未完成/失败 unit 不提交。仅严格降低原 model terminal mean192 Q 才更新，平局保留较早完成者。不得比较 native/RGB/outcome 来决定候选、根、停止或策略。全部42 seed/两个优化阶段的 fixed schedule 预先封存，未完成项保留 missing，不能把预算截断后的已完成子集称为全库结果。

末尾 winner whole15 与独立三个 raw5 segments replay 共6 forward，验证全部3个 context。每次新 search unit admission 都保留这6个 forward/模块额度；资源不足时可直接用剩余额度执行 replay，不再优化。**不预留未知 CUDA 耗时**：deadline到来后不能偷偷追加 replay；若无时间则 `export_missing=true`，保留此前完整 anytime witness，不能进入需要终态 replay 验证的下游物理流程。

全自然完成时每根 joint=1902 predict、composition=1062 predict，均85 J。composition 比旧 standalone936 多126，原因是新增、单独支付的完整42 seed scan。两策略实际耗时和调用量可以不同；自然停止后不 padding、不用空算等到450秒，不要求耗时相等。失败不退款、不重试，也不因某根失败换根。

## 时钟边界与响应变量

MAIN 的 injected clock 必须先 CUDA synchronize，再读取 monotonic host clock。起点在模型加载并验证后、第一 root nominal intent之前；CPU优化、SVD、张量转存、搜索日志与最终 replay 均留在此窗口，不能停表。模型加载/完整 saved strict 时间在窗口外单独报告，两臂不能偷共享 search 结果。单GPU串行执行，禁止其它同时运行 GPU job 污染比较。

每次 unit/call intent 前、callback return 后、memory sample、commit 和封存都检查同步时钟。elapsed>=450 禁止新 callback；已经启动的 CUDA callback 可越过deadline，完成后记录实际 overrun，停止后续 dispatch。这是 **cooperative wall cap，不是 kernel硬抢占保证**；overrun是可见违规响应，不隐藏、不删根、不追加额度。deadline之后完成的更优 witness记为late，但不进入 capped incumbent。自然完成或预算/失败停止后 `seal()` 固定实际终点，后续另一个臂运行/strict耗时不再污染此 run 的 elapsed。

`sample_memory(allocated,reserved)` 接真实 CUDA allocator数据，维护峰值，超过共同 cap停止新 dispatch；未采样为 null，不编造0。runtime 还须独立保存 native allocator/同步来源。实际 wall/peak/全部调用向量按根成对报告；`compare_reports` 不要求 elapsed 或资源相等，保留 within-wall 与资源响应，不签科学结论。

## MAIN runtime 接缝（本次不实现 NN）

1. 从本 prepare 读取 slot，完整验证 source/classes/model hash，固定TF32 off、autocast off、MATH SDP、原native float32。原 `HostBackend` 有冻结1068总forward cap，**不能直接当新1902预算 runner，也不能 monkeypatch**；MAIN须新局部 budget-aware adapter。
2. `Ledger(root, arm, synced_clock, binding=slot)`。按 `schedule(arm)` 调用 `start_unit(label)`；它先确认整个 unit 的最坏资源额度加6 replay reserve，未满足则不进入任何模型callback。capacity reservation不是付费调用，只有真正 intent 才进入累计数；已进入/失败调用永不退款。
3. 在 actual predict 前 `p=ledger.intent('predict')`，原三个模块低层callback前各 `intent(resource)`，完成后各 `returned(id)`，最后 `returned(p)`。模块严格AE→predictor→pred_proj。J前 `intent('jacobian',horizon=h,dimension=d)`，其 primal predict及模块仍逐次计数，J成功返回必须正好h个完整 primal；不存在把J primal当免费/重复扣算。root SVD/ridge各自intent/return。
4. 每个 intent 事件必须由新 runtime **durably flush后再dispatch**；当前 Ledger 自身仅维护 RAM events，不是durable writer。异常时逐层记录 `returned(...,success=False)`，未返回、未开始部分和真实错误全部保留。不得把 exception转成成功、退款或藏掉pending。
5. 完整 unit `finish_unit()`；`commit({'logits':full30,'contexts':all3}, original_goal, tag)` 维护 anytime cost/时刻与literal raw bits。该函数核域、3帧schema、目标和动作history进展，**不证明latent由真实模型产生**。新 runtime/strict必须将 witness逐一绑定完整已返回call、same context/raw/goal/source和实际输出数组，不可拿RAM witness作实验依据。
6. 首次预算拒绝、原生callback错误、内存超限或自然完成时停止；资源拒绝且尚有时间可使用预留replay。结束同步、保留pending/error并 `seal()`，保存 `report()`。无 incumbent或export missing均保留固定4分母，不挑根补跑。

accounting report不是model strict receipt。prepare标记 `runtime_implementation_required=true`；当前不存在生产 run/consume CLI，不会误宣称实际328/170等计算已经完成。MAIN runtime 后续需独占新stem、完整实际模型source closure、逐call张量与资源日志、beforeafter hash和独立saved strict。本版仅让资源公平性定义可实施，不替代这些验收。

## CLI

以下仅供MAIN审读后运行saved-only prepare/verify；worker未执行。工作目录 `/home/gbk/code/LeWorldModel/le-wm-main`。

```bash
.venv/bin/python -m experiments.spectral_reachability.agent_p2_compute_matched_protocol_20261029_v2 prepare \
  --planner-output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_three_program_20261028_v1_cuda_MAIN01 \
  --planner-receipt /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_three_program_20261028_v1_consume_MAIN01.json \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_compute_matched_protocol_20261029_v2_prepare_MAIN01.json

.venv/bin/python -m experiments.spectral_reachability.agent_p2_compute_matched_protocol_20261029_v2 verify \
  --planner-output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_three_program_20261028_v1_cuda_MAIN01 \
  --planner-receipt /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_three_program_20261028_v1_consume_MAIN01.json \
  --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_compute_matched_protocol_20261029_v2_prepare_MAIN01.json \
  --prepare-sha ACTUAL_V2_PREPARE_SHA
```

exclusive own新metadata，4MiB cap、fsync；verify全量重建。v1的不可等算结论作为冻结历史保留，v2不继承其“不等实耗即拒绝”的门。

## 测试与局限

使用 addy-test-driven-development：先得到缺模块ImportError RED。随后两次实际RED分别为停止后重入错误类型与RAM context缺真实layout；修正接缝/fixture，未放松模型context校验。最终 `.venv/bin/python -m pytest -q tests/test_agent_p2_compute_matched_protocol_20261029_v2.py -x`：**22 passed in 1.66s**。

RAM覆盖两臂全固定生命周期及1902/1062精确调用、共同cap但耗时不同、budget refusal、真实intents失败不退款、J primal/模块嵌套、late witness、anytime平局、封存时钟、不合法context、replay reserve、内存上限、固定根/实际prepare重建ABI。所有为RAM记账证据，不是CUDA timing、神经forward、物理收益或P2成功证据。计时噪声、实际 FLOPs、engine backward开销、早停频率与wallcap下最终模型/物理收益仍待MAIN新runtime实测。
