# P3 prefix / terminal 时间分解与 causal teacher-forcing 只读归因

日期：2026-10-27。仅分析真实 HOST CUDA / physical 的封存 saved artifacts；本交付不运行模型、物理、训练或优化，不修改旧源码与结果，不调参数。新增测试是实际保存数据的算术/来源回归，不是新增独立实验根。

## 结论先行

此前称“oldQ 动态受损 root2/3”的两根，指 **一基第2/3根，即零基 root_index=1/2**；零基 root3 的同Q两调度实际为 tie。

两条受损根在 step5 **均没有 predicted→actual 编码 Q 排序反转**：

- root1：step5 的预测 Q、实际 Q、物理 pose 均认为 dynamic 更差；终态 step10 才观察到“预测 dynamic 更好、实际 Q 更差”。这定位的是已保存边界间的排序变化，不是误差最早发生的物理子步。
- root2：step5 与 step10 的预测 Q、实际 Q 都偏好 dynamic，但物理 pose 已在 step5 更差。因此是可观测的 Q-proxy / pose 不一致，不应误写为 latent 排序反转。
- 新真实 teacher forcing 的全部17项 native MSE、ownQ 投影 MSE 都较 imagined rollout 下降；然而 root1 dynamic 的目标代价低估扩大，预测调度排序仍错。故不能将问题完全解释为前缀 rollout 误差累积，更不能宣称 teacher forcing 已修复控制。

这里 actual Q 指真实 RGB 经冻结 encoder→existing projector 得到的 native192，再使用原 Q 评分；它不是物理状态真值。物理 pose 单独由原始 state 算出。

## 证据、分母及固定定义

仓库根为 `/home/gbk/code/LeWorldModel/le-wm-main`。以下路径均相对此根；所有 SHA256 为本次只读复核值。

|封存输入|路径|SHA256|
|---|---|---|
|planner report|`results/p3_joint_horizon_goal_metric_scheduler_20261027_v1_cuda_main01/report.json`|`23eae72d18fa02e9997859311bef282fc4f5e3478790d846524a3fa6f6b70950`|
|physical report|`results/p3_joint_horizon_goal_metric_physical_20261027_v1_main01/report.json`|`6c3659e199a8151fee2fa2c020a0cfe63db00b691b947cdbeb762c3c18d6004a`|
|prefix prepare|`results/p3_joint_prefix_cuda_encoding_20261027_v1_prepared01/prepare.json`|`ec969feae661973b845eb38677dd4d0a90d22cd990bda65d54569966c481504e`|
|prefix receipt|`results/p3_joint_prefix_cuda_encoding_20261027_v1_cuda_main01/receipt.json`|`ca6cdef177b2b8fe5e9739f2a07d2254b1183730ff9c340f52d96707f322a802`|
|prefix numerical report|`results/p3_joint_prefix_cuda_encoding_20261027_v1_numerical_main01/report.json`|`147c3cca6a66789fbd5052fc4f4017b4e0a6b63c59038cfac1fbc19c01da77de`|
|teacher-forcing prepare|`results/p3_prefix_teacher_forcing_20261027_v1_prepared01/prepare.json`|`1018bc326957644c168ddea1985dfa74f495264dc57b934960d61460a70efcd9`|
|teacher-forcing actual report|`results/p3_prefix_teacher_forcing_20261027_v1_cuda_main01/report.json`|`c5bd1f15a4a8aa7dd60050f8ce5a681aa20c1966e1715cdc3e1df013e9fba499`|

根身份，不按报告数组位置配对：

|零基 root_index|root_sha256|
|---|---|
|0|`9173218f71898716eac8fe0f3946b8e2cf20527ac71d84555c20c4e1b7fbf333`|
|1|`23b8de7837c39fb4e0fcffe7678732a9d840c57de7a0c8376e22076770ef6050`|
|2|`4afefa3be02a327aa69c22f38c17376d82340ba352aaf4aa29f95b070f5503ca`|
|3|`cf1b9bb913bf7815c9184d5de649b77bf09cc268dd80905a0feca9bd2a2edbf2`|

4根×4臂=16实际分支；15条 H2，唯一 H3 为 root1 goal_patch_Q uniform。所有16分支 step5 + 唯一H3 step10 =17个新 prefix；复用原16个真实 terminal 编码，合计33个观测边界。不存在新增独立根；重复/tie 分支不去重。

固定 `C_Q(z)=mean((Q@(z-goal))²)`，192维均值，**不是除以 Q 的 rank**。误差定义为 `native_MSE=mean((actual-predicted)²)`，`Q_MSE=mean((Q@(actual-predicted))²)`。表中 Δ 恒为同根、同 Q 的 dynamic−uniform；负值表示 dynamic 在该项更低。没有跨 Q 相减或混排。

物理 `pose=||objectXY-goalXY||₂+20*abs(wrapped(object_angle-goal_angle))`。7维 state 为 agentXY / objectXY / object_angle / agentvelocityXY，末两维不是 object velocity。所有16条物理终态 task success 为 false。

## Same-H：两条受损 oldQ 的时间分解

下表独立加载17条新 NPY、原 terminal_vector、原 goal/Q 与分段 replay contexts 重算。小数显示舍入，测试用完整双精度结果。

|root|step|uniform predicted Q|dynamic predicted Q|uniform actual Q|dynamic actual Q|Δpred Q|Δactual Q|Δpose|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|1|5|0.947424402|1.236097631|1.146765432|1.326884527|+0.288673229|+0.180119095|+7.520713|
|1|10|1.106308586|1.089102358|1.163649221|1.373396035|−0.017206228|+0.209746815|+71.857102|
|2|5|1.380465381|1.201464719|1.432514699|1.326001630|−0.179000662|−0.106513069|+4.509433|
|2|10|1.259073906|1.149465572|1.307942349|1.284264846|−0.109608333|−0.023677503|+11.635172|

对应 step10−step5 的同臂 Q 变化：

|root|schedule|predicted Q 增量|actual Q 增量|
|---|---|---:|---:|
|1|uniform|+0.158884183|+0.016883789|
|1|dynamic|−0.146995273|+0.046511508|
|2|uniform|−0.121391475|−0.124572350|
|2|dynamic|−0.051999147|−0.041736784|

root1 dynamic 从 imagined step5 向 step10 预期改善，但真实 Q 反而变差；原 rollout 终态 native MSE 从 step5 的0.128627070增至1.327098305，ownQ MSE 从0.126991913增至1.256750067。uniform 对应 native MSE 为0.058472859→0.069284153。此处有晚段预测失配，却不证明单独某次 Jacobian 或某个 suffix 物理机制错误。

root2 在 step5 的物体 XY 误差 dynamic9.146354 < uniform18.036117，但 angle error dynamic2.801799 > uniform2.131839 rad；终态仍为 XY 更小（12.413374 vs17.159996）但角度更差（3.053394 vs2.234304）。因此 pose 劣势与其角度项相容。root1 终态 pose 劣势主要伴随物体平移误差扩大（dynamic88.172133 vs uniform15.648456），角度误差反而略小（2.425455 vs2.458784）。不能从这些状态差异反推出精确接触事件。

### 完整 same-H census，不只展示坏根

共7对 same-H：以上 oldQ root1/2 两对非tie；另5对为 oldQ root0/3、goal_patch_Q root0/2/3，其 step5 与 step10 的 Δpredicted Q、Δactual Q、Δpose **全部精确为0**。这五对重复动作仍逐条保留、支付运行成本，不当成五条独立机制验证。

所有16轨迹的第一个保存非零 native error 边界都是 step5，包括上述 tie；故“step5有误差”不能推出“调度已导致损害”。本次没有 step1–4 的 latent 编码，最早误差子步仍不可定位。

### Mixed-H 独立列示

仅 root1 goal_patch_Q：uniform H3 vs dynamic H2。它不是 same-H matched terminal 结论的一部分，即使比较共同绝对时间也保留 mixed-H 标签。

|arm|step|predicted ownQ|actual ownQ|native MSE|ownQ MSE|
|---|---:|---:|---:|---:|---:|
|uniform H3|5|0.099610782|0.108011950|0.029320645|0.004365076|
|uniform H3|10|0.122896720|0.135911036|0.035460923|0.005289590|
|uniform H3|15|0.077532822|0.099026215|0.064059631|0.007640207|
|dynamic H2|5|0.115137906|0.133620256|0.026464553|0.002631925|
|dynamic H2|10|0.085741281|0.100089667|0.166362949|0.017502035|

step5 Δpred=+0.015527124，Δactual=+0.025608306，Δpose=+1.016162。不同长度终态 D10−U15：Δpred=+0.008208459，Δactual=+0.001063452；不得与 same-H 配对合并作共同终态处理效应。

## 新真实 teacher forcing：全部17项，不筛选坏根

固定原动作；每个已观测 interior boundary 只预测下一个5步 packet，不优化、不反馈重规划、不 rerun physics。H2 的 step5 输入 latent history 为 `[initial[-2],initial[-1],observed5]`；H3 的 step10 为 `[initial[-1],observed5,observed10]`，没有未来 terminal 注入。action history 保留该边界原分段 replay 的两条 normalized between-state tokens。

I=原 imagined rollout；TF=真实 causal histories 的 one-next-packet prediction。native MSE / ownQ MSE 均相对于**同一个实际 target**；ownQ 依该臂固定 Q。以下17项包含全部重复与唯一 mixed-H 两段，不跨 Q 平均。

|task|root|Q / schedule|boundary→target|native I|native TF|ownQ MSE I|ownQ MSE TF|
|---|---:|---|---|---:|---:|---:|---:|
|00|0|old / uniform|5→10|0.602890048|0.354696672|0.588976379|0.348122182|
|01|0|old / dynamic|5→10|0.602890048|0.354696672|0.588976379|0.348122182|
|02|0|goal_patch / uniform|5→10|0.050795230|0.005178878|0.004851756|0.000688713|
|03|0|goal_patch / dynamic|5→10|0.050795230|0.005178878|0.004851756|0.000688713|
|04|1|old / dynamic|5→10|1.327098305|0.939230386|1.256750067|0.916323067|
|05|1|goal_patch / uniform H3|5→10|0.035460923|0.020158361|0.005289590|0.002526541|
|06|1|goal_patch / uniform H3|10→15|0.064059631|0.019526738|0.007640207|0.002410574|
|07|1|goal_patch / dynamic H2|5→10|0.166362949|0.080406304|0.017502035|0.008964026|
|08|1|old / uniform|5→10|0.069284153|0.014957474|0.067337276|0.014637167|
|09|2|goal_patch / uniform|5→10|0.160429735|0.081725808|0.017727955|0.009979826|
|10|2|goal_patch / dynamic|5→10|0.160429735|0.081725808|0.017727955|0.009979826|
|11|2|old / uniform|5→10|0.145992988|0.044468440|0.139891633|0.042763241|
|12|2|old / dynamic|5→10|0.103367759|0.057574043|0.092418108|0.055126965|
|13|3|goal_patch / dynamic|5→10|0.159105915|0.038302505|0.022416975|0.007627096|
|14|3|old / uniform|5→10|0.212898107|0.027673760|0.211113644|0.027152269|
|15|3|old / dynamic|5→10|0.212898107|0.027673760|0.211113644|0.027152269|
|16|3|goal_patch / uniform|5→10|0.159105915|0.038302505|0.022416975|0.007627096|

oldQ 8/8、goal_patch_Q 9/9 的上述两种 MSE 下降。这里分母是预测任务数，不是独立根或独立动作数；全部残差仍非零，不能把 MSE 下降等同目标代价校准或真实控制改善。

重点 oldQ 终态目标代价对照（相同 terminal actual）：

|root|schedule|imagined predicted Q|TF predicted Q|actual Q|
|---|---|---:|---:|---:|
|1|uniform|1.106308586|1.191616138|1.163649221|
|1|dynamic|1.089102358|0.918994374|1.373396035|
|2|uniform|1.259073906|1.305048957|1.307942349|
|2|dynamic|1.149465572|1.249554543|1.284264846|

root1 的 TF 调度排序仍偏好 dynamic，而 actual Q 偏好 uniform。dynamic 目标代价低估 actual−predicted 从0.284293677扩大到约0.454401661，虽 native / ownQ MSE 下降。这正是“向量预测误差更小”不必等于“目标方向代价更准确”的本例反例；不能将整个失败只归于 imagined 前缀漂移。root2 TF 仍与 actual Q 一样偏好 dynamic；真实 pose 的相反排序没有被改写。

TF 是对已知真实边界的离线条件输入替换，支持“观测历史替换可降低这些保存 target 的向量误差”这一受限结论；不是闭环执行试验，也不唯一分离 encoder proxy、conditional dynamics、历史分布偏移或接触机制。两臂 TF 输入是各自真实状态，不是强行把两臂放在同一个物理状态的动作因果比较。

## 源码与保存链独立审阅

只读 prefix runner：`experiments/spectral_reachability/p3_joint_prefix_cuda_encoding_20261027_v1.py`，SHA `9973c1a4e8bd5e49fba64d5ce25f072a878f05fc6c89064a913b2edaaed03fdd`。

- `prefix_vectors` 从逐 packet replay 重建 normalized action history、滚动 latent chronology，再导出 prefix；不是盲信 H2 `[1]` / H3 `[0]/[1]`。新增测试不导入该函数，独立重建同一时序。
- `collect/verify` 绑定 planner/physical/source seal、同根 goal/两Q、selected actions、分支 journal 与 RGB/像素；实际17帧按所有分支内部边界导出，无坏根选择。
- `Ledger/execute_cuda` 分开计17 encoder与17 existing_projector；NPY fsync/hash及逐帧 attempt/result、journal 持久化。保存的 model-before/after 与 source_verified 证明该 receipt 下的检查通过，不构成独立重算模型输出的证明。
- `consume` 分开继承16 terminal 编码；直接 native192/原Q算术，没有重编码终态或物理重放。

只读 teacher-forcing runner：`experiments/spectral_reachability/p3_prefix_teacher_forcing_20261027_v1.py`，SHA `cfa5826986f82e962cee2e82a4477f982d1e0c7c22880e60a207827344f4f4d3`。本次未修改、未 prepare、未 launch。审阅与实际 saved 复算未发现阻断本次成功 CUDA 的 causal/binding/count bug：

- `causal_context` 仅拼接截至当前 boundary 的全部实际 prefix；H3第二段不会混入 imagined5，也不会注入未来10/15。actionhistory未换成零值或错误包。
- `material` 使用 externally pinned prefix receipt / numerical SHA，经 base.verify 重建原所有来源；绑定17个 NPY，目标严格为下一 prefix 或原 terminal，固定原动作。新增测试独立逐任务对齐原 planner segments、root/arm、实际 target、input context 内容和 actionhistory。
- `CudaReplay` 每次消费 `(5,2)` raw，产生一次 `(1,3,10)` action encoder和一次 `(1,3,192)` predict，顺序与 `Cap` 一致；实际34个完成 entry 按17任务保留，无去重。使用 CUDA/no_grad/MATH SDPA，不调用 encoder、AD 或 physics。
- 原始 predicted context 在评分前持久化，再写逐任务结果；finally 尝试恢复 hooks、复核模型和来源。外部进程硬杀或磁盘故障时的绝对落盘保证不能由成功路径测试推断；此处没有观察到这些故障。报告未单列 action_encoder 数值总数，但34条有序 entries 可独立恢复17次，不能漏账。

## 预算、验证与剩余缺口

资源分开，不重复收费，也不把继承成本写成免费或等总FLOPs：

|层|实际新增成本|继承/复用|
|---|---|---|
|original physical|本审查不新增|165 selected +240 warmup steps，16 terminal encoder+16 projector|
|prefix actual CUDA|17 encoder+17 existing_projector；0 predict/AD/physics|原模型、goal/两Q、planner与物理轨迹|
|teacher-forcing actual CUDA|17 predict+17 action_encoder；0 encoder/AD/physics|上述17 prefix及原16 terminal，不再次编码|
|本交付 saved-only 审查|0 neural /0 physics /0 optimization|只读 NPY/JSON、NumPy重算；没有新增独立根|

新增回归文件：`tests/test_p3_prefix_error_attribution_20261027_v1.py`。不导入任何运行器或模型，绑定实际报告SHA，独立复算全部33边界×两Q及17 TF任务×两Q，复核原始state→pose、滚动latent/normalized token内容、完整分支及计数，并包含错Q评分和重复身份反例。运行：

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p3_prefix_error_attribution_20261027_v1.py
```

结果：30 passed。另只读运行 main 的 teacher-forcing tests：8 passed；这些含mock合同与saved census检查，不是第二次真实 CUDA。

本交付填上“缺真实 prefix 编码/缺条件输入诊断”的证据缺口，但以下仍未解决：step1–4内首次误差时刻、确切接触日志/接触因果、encoder/goal/Q的物理充分性、未见根上的泛化与严格P1/P3准入。TF residual 与 pose mismatch 均仍存在；不能推论唯一根因、理论失败、物理改善、SOTA或全 pipeline 计算成本等价。所有数据仍是原四根 calibration。
