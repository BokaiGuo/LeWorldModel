# 最小 prospective 动作效应干预合同：20步反馈→25步真实终态

范围：新 `agent_p1_prospective_action_effect_20261028_v1` source/tests/本文，准备阶段；不执行 prepare 或任何生产 forward/physics，不训练、不拟合Q或残差参数、不按结果筛根/词/对，不打开 fresh reserved。不编辑既有冻结产物。本轮使用TDD，先观察缺模块失败，再实现纯RAM调度与精确数学。**当前没有生产 run CLI／原生 adapter／新 strict consume，不能声称可直接启动真实实验**；下述完整实验合同和可调用调度核用于 MAIN 接下一步原生接缝实现，prepare本身不是dispatch许可。

## 1. 先排除已做实验，明确新增量

已只读核文件、schema/status 和 SHA，不重跑原消费者或实验：

| 已有覆盖 | 固定产物SHA | 本轮不重复做 |
|---|---|---|
| 原真实teacher15上下文下五词预测、真实20终态，80对中76可比/4未知 | margin `f723c5eb73ecebd0615a2ded0aa4b741c6f132ca5f6063c673f12d5e98539143` | 不把已有H1重新包装成prospective |
| 全部三帧实际5/10/15、6..15真实动作history、PNG→latent→module输入的严格saved定位 | history `7d171fea5ad553541dfd6393e4bf2530992ef916c92fcc39295590de7337d3a2` | prepare重新调用原saved `history.build()` 并全报告canonical比较；不是信单个true |
| T3三项、相消、8种移除组合 | T3 `03eb182af71d4cfd4acfe3ee25ef1ad8e253ce3cd80e82f37a2eeba2abcf888e` | 不把mask移除当可实现干预/修复 |
| 自身真实动作 H1/H2、teacher/context替换 | horizon report `e319035d9e13a6ae2dc4cbfa5cefb9eb2e6ccd49dbcf35408fda1704f5f34253`，status recorded，无errors | 不重做context开放滚动归因 |
| 官方rollout/cache同输入 API | API report `38ad7134ba7f13b7244a9bc9ad311d476ccd67a0df53daf9fa7f314e7876ae3e`，completed，无errors | 不再提尚未执行的API对照 |
| I/R实际rolling接线，boundary5/10/15至terminal20 | saved receipt `182d5ce14a6363b287607c511d3bc4dde85769618e8c8082029f090fb19c1232` | 不重复rolling历史审计 |

P1原40预测已经是**全部三帧真实的teacher15**，不是P2 own-imagined midpoint。P2 actual-mid改善7/8、不同批own-imagined 3/4→4/4均不进入本合同authority/归因；不可将其外推解释P1全部12反转。本prepare先完整重建history定位，核全部40槽的frame5/10/15同condition、原plan/anchor、原raw6..15 history与packet6..20 chronology后，才生成新任务。缺一即拒绝，不发起“true-mid替换true-mid”。本worker只是读原源码/产物明确上述接线，未自行执行这个昂贵saved重建。

现有40库面板在15→20；已有rolling终点20，不提供其后20→25同起点全部五词干预。**新增问题**：经过一次模型选择的真实动作、获得真实20步RGB反馈后，冻结模型对同起点 do(五种原raw packet) 的真实终态作用是否预测正确？共同终态残差能否解释目标margin，而不需要动作效应误差？另仅在每个相同25步物理终态做一次target-overlay renderer切换，隔离renderer对编码动作效应的影响。不增加API/多horizon/新Q/新控制修复因素。

## 2. 固定全cohort与最小闭环

固定8 anchor：root0/1 × off/on × Z/S，固定原五词 `zero,uZ,-uZ,uS,-uS`，每词仍5×2原float32 raw动作。全部候选 signed bits 从冻结 selector 读取，不重新求方向、幅度或根。已封存 stage15 model-only selection 索引为 `[3,8,13,18,21,25,31,36]`；这是原模型argmin，不是按native收益或误差选。模型、normalizer、goal192、identity metric、原state7 task/outline与renderer contract全不变。

时间线：

1. 每anchor从原完整reset/goal/warmup重放到10，执行原prefix到15，再执行**原先封存的模型选择suffix**到20。前15步与旧teacher严格复验；15→20的raw/state/snapshot/RGB可对原被选分支复验。不是从state7瞬移或隐藏状态不全的snapshot clone。历史index12缺图依然缺图，不重写旧80对。
2. 在真实20步反馈编码一次得到z20。由原teacher context `[z5,z10,z15]` shift得到 `[z10,z15,z20]`。历史动作只取真实11..20即 `prefix5+selected_suffix5`，按原float32 mean/std仅归一化一次。新 `forward20` 是这一纯数学接缝；禁止用预测z20、旧20图或跨episode借图。
3. 在该同一完整context/过去下，固定顺序预测五个21..25 packet；全五完整三token原AE→predict API→predictor→pred_proj trace与最后192输出先保存。原identity192 float64 MSE argmin（数值完全相等才按原word index）封存本次闭环动作与预测，**在任何21..25的真实执行或标签之前**。终态/物理标签不可进入forward包或选择器。
4. 同一个leader到20的环境继续执行zero分支；另四个环境各从原reset严格重放相同20步。每个重放20的完整state7、完整snapshot digest、actual raw20、PNG/pixel均与leader逐字节相同才允许各自do(packet)。未达到/不一致为该槽unknown，不改起点、预算或词。因为重放终点必须相同，完整接触/solver内部状态也必须在snapshot审计范围内；仅state7一致不够。
5. 五分支分别实际执行5步，记录所有attempt/return、每步完整state/reward/termination、实际raw bits、native指标及观测guard；完成25步后各编码自己的同condition RGB。模型在步骤3选择的分支终态就是该闭环策略的结果，其余四条为同起点的研究干预，不是额外四次部署选择。选择在标签之前固定，zero先执行只是采集顺序，不是控制fallback。
6. 每个已完成25步的**同一物理终态**只切换原target-overlay off/on renderer，保存另一视图并编码，随后恢复原renderer。不得改变goal/native state、速度、接触或时间，不多走physics。before/after完整snapshot/实际state7/目标身份及恢复状态由独立事件闭合；若漂移，alternate label unknown，原合法终态标签仍保留。不是P5的terminal颜色普适反例，不沿不同物理轨迹凑off/on比较。alternate视图是标签域干预，绝不重做选择或修正模型。

这形成两个连续的模型选择packet（15→20、20→25）和一次新增真实反馈replan，是最短能新增反馈决策的library闭环。不是只做离线重评分；也不声称此library策略就是原D2P或优于GN/CEM。**没有部署任何残差修正**，不添加多arm控制修复对照。校准实验的do(action)足以鉴别参考残差与动作效应误差；若要证明某种修复因果提高任务收益，必须另行事前冻结对照实验，不由此自动推出。

## 3. 可识别量、严格反例与解释边界

对同anchor同起点，记模型预测 `p_w`、真实编码 `a_w`，零词固定为0：

`b0=a0-p0`；`h_w=(a_w-a0)-(p_w-p0)=(a_w-p_w)-b0`。

这些是**直接观测差值**，不是均值拟合/回归/共享偏置估计器；只作标签域算术，不写回冻结模型、Q、控制器或候选排序。五个词全部列出，零词自身 `h0=0` 不作为额外成功样本。

- 若所有可观察 `h_w=0`，模型在这组真实动作干预上的latent action effects正确，即使b0很大。只能说明该固定动作库/上下文，不是任意动作、任何任务或物理坐标的正确性。
- 若某个 `h_w!=0`，精确纯共享终态偏移假说被该动作对照否定；此处“否定”是等式命题，不是新统计science gate。所有非零值和大小都输出，无调参epsilon。CUDA重复误差若没有独立界，不从极小非零断言实质病因；本合同不偷加重复forward估噪、不宣称已知误差界。
- 混合误差普遍可能：b0非零且h非零，不能强行二选一。h还可能含真实动作导致的RGB/encoder变化或未见动力学误差；它识别的是冻结预测→实际编码的动作效应误差，不证明内部模型某个模块唯一有错。跨anchor比较还含状态/历史变化，不假装为隔离干预。

目标margin按原192 MSE。设 `r_i=p_i-g`，`m_p=mean(r_i²)-mean(r_j²)`：

`m_a-m_p = B0_ij + H_ij`，其中

`B0_ij=2mean((p_i-p_j)*b0)`，

`H_ij=2mean(r_i*h_i-r_j*h_j)+2mean(b0*(h_i-h_j))+mean(h_i²-h_j²)`。

source以原binary32精确有理数核验残差=0，MSE仅除192一次。保留全10对/anchor（总80）及所有unknown。zero missing时b0/h/B0/H均null，**其它两臂都完成的pair真实margin仍保留可比**，不按zero缺失删anchor。不用B0/H数值筛对/掩盖相消。

此参考b0不同于旧T3的 `c=e_j`；不可将旧B/R/Q必要5/4/5挪成新因果占比。一个可复算反例是 `p_w=w, a_w=w-3, g=0`（192维均重复该值）：h全零而目标排序可反转。因此准确动作效应不保证目标ranking稳定；反过来 `a0=p0`、某非零词有额外误差，则b0为零仍存在响应误差。

物理结果仅独立标签域：原native success/全state7 distance、allXY、objectXY、angle、velocity分别报告，保留success与distance的速度语义差异，不合成新分数/不换metric。原生distance反转可被velocity驱动，不等价普遍encoder-物理不对齐；本实验也不证非贪心前缀优势、freshness或强基线收益。sample unit仍known根2，不是40槽/80对。

renderer直接可识别的量为 `D_w=(a_alternate,w-a_alternate,0)-(a_original,w-a_original,0)`，所有真实终态物理state保持不动。逐词保存精确mean(D_w²)；若非零，说明同一物理动作作用的编码表示受该renderer干预影响，而非物理任务变了。与原model差作比较时预测项相消，所以不需要另跑alternate-context预测，也不改变原goal。若D为零且h仍非零，只能保留hidden state/冻结动力学/其它表示因素未分辨，**不能自动归因hidden state**。单张RGB不观测速度是可检验的表示限制，但本最小合同不做破坏原接触状态的velocity setter实验，无法唯一识别隐状态病因。

## 4. 最小新增预算与失败合同

每anchor：5个完整重放20起点（leader复用为zero），共5×25=125 public steps；1次20反馈编码＋10次25双视图标签编码；5次forward。8 anchor合计：

- public steps最多1000（不是8×25部署成本）；40实际候选干预，8条模型选中的部署路径；
- image encoder/projector各最多88；AE/predict API/predictor/pred_proj各40，三token不是三次API，全部batch1；无AD/J/SVD/新训练。此前尚未运行的48编码草稿不包含renderer因果分离，现新增40次同物理终态alternate-view编码并事前固定为88，不是运行后扩容；
- 新goal/history/source早期编码从原严格authority读取，重放子分支20仅核图像不重复编码；相同像素但不同新episode标签仍各付25编码，不dedup/退款；
- 原compiler/encoder/40物理/teacher/API/rolling费用单列继承，不能宣称整个实验等FLOPs。internal physics、constructor/reset/set_state/goal setter/render等须逐actual attempt/return计账，不能用public1000顶替。

原NativeStop、guard fail、earliest terminal即保unknown，不强行执行到25；第25步本身返回terminal可作为真实terminal25评价。预测/封存失败时当前anchor五槽均保留，没有候选dispatch/假seal；别的anchor仍保位置，不按失败替根。被选分支未知时policy outcome未知，不用别的成功候选fallback。没有原生adapter异常回收验证前不派发。新80对unknown数量由实际证据决定，**不能强制复刻旧4缺失或悄悄把旧12补齐**。

本prepared合同刻意保留未审计原生资源caps为null并使run_authorized=false；MAIN应在第一次真实调用前新冻结完整caps、deadline、磁盘final reserve、model loadedbytes与reset/构造ledger，未完成不能把本文件叫ready-to-run。预算更改需新版本，不能事后扩容/减样本。预计public/forward/encoding数量是事前上限，不是已发生费用。

## 5. 现有runner的具体复用接缝与必须新写的薄adapter

源文件固定以下4个runner SHA，另固定history定位器SHA `0f59ee8ad57a2e6bea619cdca07edb4cbc3eb9a84652d97c019dd3a6ef4fcb33`。仅在MAIN调用prepare/verify时lazy import并执行原history的saved build，不执行任何模型或环境；其余runner不import执行build/run，不改globals：

1. `p1_moving_live_feedback_physical_20261027_v1.py` 的 `metered_class/native_factory`、`Store/Meter`、`state_seal/capture`、模型原loaded/unchanged校验与finally回收。其setter只在reset内允许，禁止为了本实验新增中途state restore。旧Meter caps需新子类，不能提高原全局cap。
2. `agent_p1_teacher_cross_library_physical_20261028_v1.py` 的 `place/dispatch/observe` 思路复用原同condition观测、actualraw/完整snapshot和NativeStop检查；原 `episode` 固定40条15→20，不直接调用/改名伪装25步。新adapter须扩明示绝对步而非修改原reference语义。
3. `agent_p4_moving_multiboundary_20261028_v1.py` 的 `normalized/roll` 作为actual20 packet输入的独立对照，`encode_boundary` 证明byte-identical视觉alias是可行模式；**不能直接调用其validate_request**（boundary白名单5/10/15且goal mode I/R），不能把20/25谎报为15或造第4次旧反馈。新视觉ABI必须在receipt明示真实20/25，alias只作为旧visual handler传输，双向字节核验且费用不借账。
4. `p1_moving_live_feedback_teacher_fidelity_cuda_20261027_v2.py` 的原CUDA `Native` 模块hook/AE→predictor→pred_proj追踪与Store/no-AD约束可复用设计；其 `validate_request` 和Ledger硬编码12身份/48事件，不能直接套到40新槽或猴补。新adapter保持相同真实forward表达式 `model.predict(z3,model.action_encoder(outgoing3))[:,-1]`，新显式identity与160个API/module事件账本。原modelhash、float32、MATH/TF32off/autocastoff/CUDA-only仍必须严格相等。

新source提供 `jobs`、`forward20`、`validate_forward`、`validate_start`、`drive_anchor`、`effects`，MAIN可直接调用。`drive_anchor(job,goal,adapter)` 的adapter接口是：

- `replay20(job,word)`：reset+原20步重放；word0仅一次20编码，返回 `{fingerprint,forward,handle}`。fingerprint含actual raw20/fullstate7/snapshot digest/PNG/pixel。forward恰latents3/history2；handle为原生环境私有能力对象，不进入forward。
- `predict(request)`：只收schema/anchor/word/latents/history/raw5_bits，返回原native float32[192]。goal仅在调度器算原MSE时使用，terminal/path/native全部拒绝。需完整真实module hook和请求→输出receipt。
- `seal(value)`：全五预测、原成本、选中词/rawbits与起点fingerprint durable封存，返回seal文件SHA；抛错即不dispatch。真实adapter应签发只对应封存rawbits的dispatch能力，记录event链，不只返回一个true。
- `advance5(start,raw5_bits)`：只能在该anchor seal后执行，原guard+原native dispatch，完成25再编码actual endpoint；返回actualbits/绝对步/完整state7/terminated/truncated/complete/endpoint。再返回同state切换renderer后的alternate_endpoint及renderer_audit(before_sha256/after_sha256/restored)，原生完整事件验证不能只靠此摘要；失败只让alternate标签unknown，原标签保留。完整native early stop或guard receipt由adapter保存，不能用本核的布尔字段代替。所有后续分支同一fingerprint核对；最终adapter上下文管理器在任何退出都关闭全部env、hook恢复、model后验和partial报告。

调度核是可调用算法准备，不是独立安全边界：真实receipt/normalizer/model binding/loadedbytes/源闭包/renderer+native evaluation/内部预算/hashchain/artifact全集和strict consume仍由新adapter证明。当前仅RAM fake adapter走通，不把它伪装成已实现的HOST runtime。

## 6. 当前CLI与明确剩余工程

工作目录 `/home/gbk/code/LeWorldModel/le-wm-main`；这些prepare/verify都是纯saved元数据，不执行模型/环境：

```bash
.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p1_prospective_action_effect_20261028_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.agent_p1_prospective_action_effect_20261028_v1 prepare --output results/agent_p1_prospective_action_effect_20261028_v1_prepare_MAIN01.json
.venv/bin/python -B -m experiments.spectral_reachability.agent_p1_prospective_action_effect_20261028_v1 verify --prepared results/agent_p1_prospective_action_effect_20261028_v1_prepare_MAIN01.json --prepare-sha <实际SHA>
```

prepare核固定输入文件SHA及5个source SHA，先完整重建原teacher15 history闭包、核全部三实际帧和真实action history，导出紧凑证明摘要与8jobs/40raw-word合同、原task/goal refs、原模型policy/tensorhash及本三文件SHA；不读取reserved，不复制旧49k闭包。verify同样重建，不重复执行实验。它认证原context历史，不冒充**未来20→25新实验**的原生closure。文件只允许本stem新路径exclusive写，不覆盖。

剩余工作只有具体落地而非另加诊断：MAIN审读本设计；新独占native20/25 adapter＋固定所有资源caps/loadedbytes/reset账本；新strict saved consume核完整物理/编码/module桥和全80分母；MAIN在实际调用前另行冻结授权。**没有run命令**，本文不编造可执行CUDA CLI或生产结果。此准备worker不启动这些实验。

测试另覆盖完整合成prepare布局/旧seam篡改拒绝、全三实际帧/时间/condition/raw history/重建篡改拒绝、actual20 context shift/一次normalization、seal失败零dispatch、全分支重放漂移/early stop unknown、同state renderer变化/漂移与exact反例。未运行真实prepare/build或生产实验；这些测试不认证CUDA/物理adapter。

最终新34项＋冻结T3结构33项＋原margin47项联合 **114 passed in 5.22s**。这版在任何生产prepare之前冻结；旧48编码草稿和106测试状态均被本未运行的88编码合同与114测试状态取代，不涉及修改任何生产结果或科学门。
