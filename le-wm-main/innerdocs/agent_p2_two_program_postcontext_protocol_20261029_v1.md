# 两完整程序的跨 context 迁移：事前模型协议（待 MAIN 批准）

2026-10-29。仅 protocol 与 RAM 公式核验；无新 NN/AD/physics/fullstrict、无生产准备、无子代。既有 source/results/canonical 和 Pauli six-target 工作不改。P1 未通过，P2_mechanism、physical_release、scientific_admission 全 false。

## 1. 已有资产及不重复边界

内层根目录 I=`/home/gbk/code/LeWorldModel/le-wm-main`，下列相对路径以 I 为基准。

实际 `results/agent_p2_saved_program_linker_20261029_v1_MAIN01.json`，SHA `82f73b810e5cd34b12907f11d0d063914578263ae4e41163e9581ac3ce3f5567`：168条候选链、336程序capsule、两个known根，各原臂42候选，默认168 linked/0 blocked。其 evidence 明确 new_model_calls=0、upstream_fullstrict_rebuilt_here=false；认证的是已保存的同输入转移，不是未见 context 的程序可执行性。

实际 `results/agent_p2_compute_matched_protocol_20261029_v2_prepare_MAIN01.json`，SHA `fbc151b975409eff95bffa084707531f20d1659cbcf08ea23e5da1b490db4bd2`：预注册 wall-capped，不是 equalFLOPs。其后 wall-capped runtime 已实际完成，并非仍缺 runner：report `results/agent_p2_wallcapped_cuda_20261029_v1_cuda_MAIN01/report.json` SHA `1359439a994f91f8eb18e0128fd53e0b0aea800ff281876c6cbf84e222f6e3ba`、receipt `results/agent_p2_wallcapped_cuda_20261029_v1_consume_MAIN01.json` SHA `7b3eba575a9cf39a565c1956a26a937828509b313be7e4aecb9f95e7f0b67ac1`（MAIN 已交付权威；本轮不重消费）。旧三程序及该运行已经在自身 imagined cp5/cp10 重编译，不重做或改名宣称新机制。

旧完整程序 authority：

- prepare `results/agent_p2_three_program_20261028_v1_prepare_MAIN01.json` SHA `86af2d24f73e769fb3b91fa8a571f8ca265df95c93af780d3924ea644585d461`。
- report `results/agent_p2_three_program_20261028_v1_cuda_MAIN01/report.json` SHA `56e8b4f2df522f7a0c4cb9bc37ad74bcbc5d4f89e2c5cd601d42cd0eb071d8ef`。
- receipt `results/agent_p2_three_program_20261028_v1_consume_MAIN01.json` SHA `c491555fda5bb543af69f70987541edb0abb99fa32c559f3aa6190901393017f`。
- 原 producer source SHA `3111312477ff3cd9a764dc7d063861c1becaaf58885dc0fb25c2f3ec73d15fd5`。

本协议只检验一个尚未由默认 linker 认证的接点：把某完整已编译程序，搬到另一完整程序的模型 postcontext 上，实际重新运行它的原动作。不是 metadata连接，也不是把其旧root-context预测接在前一终态后。这个小切片是固定程序重用/接缝诊断，不声称全库组合搜索或新的优化器胜利。

## 2. 固定输入、程序身份与新时域合同

固定 root_slot=0,1，原off known2、原task/goal192/identityQ/model/normalizer/raw10 warm history不变。不看物理、RGB outcome、成功标签或新评分选根/程序。每根固定：

- A=`r{root}/joint_H3_no_composition/c01/p0`，原bank index1（root_dls_progress）。
- B=`r{root}/joint_H3_no_composition/c02/p0`，原bank index2（既定SVD顺序首项）。

这是原bank编号的事前指定，不是取旧winner或按Q排序；不检查表现后更换编号。两者各是原**一个完整15步程序**，planned_steps=committed_steps=15，均有原root context、compiler身份及完整三包预测。禁止把三个raw5 packet改称三个program，也不把原H2仅承诺5步capsule偷扩为10步已执行程序。

因此最少两个完整程序需要30控制步、六个raw5 packet，而不是原H3/full15对照的别名。时间为模型 control0→15→30，不是actual cp5/真实反馈；B 的旧输入在control0，其新调用输入在control15。原goal保持，但**30步目标时距是新探索性合同**，不能合并成原full15任务结果，必须 MAIN 批准后实现。

原每程序30logits半径 C15=.75√15、NumPy tanh(float64)→float32 raw映射、raw[-1,1]、once-normalization全部保留。组合60logits的新联合域声明为 C30=.75√30（两原球的外包球），联合对照也使用该球；它不等同于两个分块球，不冒称原合同逐项未变。固定A/B各自须通过原合法见证；concat须严格合法，不repair、不逆atanh、不加epsilon。未来joint可跨分块重新分配logit预算，明确是更强、不同支持的无组合挑战者，而不是纯程序边界因果隔离。

本轮只读四capsule并RAM核算：两根concat norm分别 `3.0618621784789726`、`3.061862178478972`，均≤C30=`4.107919181288746`，concat chart rawbits与两原rawbits字节串连接完全相同。这不是新模型转移验收。

## 3. 必须真正计算的模型连接

完整context C=(Z3,H2)，Z3为3×192 latent history，H2为2×10原between-state action token history。设 Fθ 为原冻结 native predictor 的一包转移，动作先原始float32 raw、原normalizer恰好一次，再原action_encoder→predictor→pred_proj；转移保留完整history roll。令 R(C,U) 为连续调用Fθ，不可只传last latent。

每根 composition arm 新付以下调用，严格按顺序：

1. 从认证原root C0，用 A 的literal15raw运行三包，得 CA=R(C0,UA)。新结果逐位核完整原A保存context，不把读取旧值算新调用。
2. 从**这次新返回的 CA**，运行 B 的literal15raw三包，得 CAB=R(CA,UB)。完整保存B每次原始输入、normalized输入、模块输出、三个postcontext。
3. 新独立whole30 replay：R(C0,concat(UA,UB))六包，与上述六个完整context逐位相同。
4. 新独立segmented replay：先A三包、后B三包，各自接前一返回完整context，再核全部六context。即额外六包，不把步骤1/2重命名算作独立重放。

终态Q=mean192((z30-goal)²)。另可保存“B旧root预测 vs B新context预测”的向量/Q差作context迁移诊断，旧数值单列 inherited：不相减声称任务收益，不用它选择程序，也不能把旧B输出作为新CAB。即使新旧terminal相同，完整context/输入及所有转移证据仍不能省略。

## 4. 强无组合对照：相同full30初值的联合两轮下降

joint arm独立从C0起，u0=concat(A.logits,B.logits)，目标/Q/chart/联合球与composition一致。先验不选择替代初值，不调用新bank/SVD/ridge。每轮固定1个 literal raw192×60 J（六包primal），沿原continuous tanh chain构造logit梯度：g=(2/192) diag(1−tanh(u)²) Jrawᵀ(z−goal)。这是原连续chart优化约定，不声称对float32 cast求数学导数。

令 v=−.5 C30*g/||g||；g=0时v=u。d=v−u；若gᵀd≥0则d=0。固定alpha=[1,.5,.25,.125]，评价incumbent加四候选，按(Q,index)稳定选，平局incumbent；保存全部，不按效果提前停止。每轮新完整incumbent forward6、J primal6、四候选24、selected replay6，共42predict。第二轮在上一选中点重新求J，禁止沿用原192×30 root J、旧rootcontext J或第一轮J。两轮后whole30+segmented再12predict。

联合对照最终Q不得高于其本次初始incumbentQ；初始incumbent整六context须与composition同动作结果一致，否则工程failclosed。此guard仅控制modeledQ，物理效能未知。不得比较旧joint15评分与新30评分声称增益。联合臂可以胜出；没有“组合必须改善”的门。

**实现边界**：现旧 `local_joint`/`legal` 有full30-logit形状限制，不能直接将60logits喂入，更不能patch globals。批准后应新局部维度明确的H6 rollout/J/guard实现，复用冻结单包native primitive及保存ABI。若无法验证六包完整roll或预算接口，则对应槽blocked，绝不回退截成15步。

## 5. 完整分母、继承账与新增账

固定2roots×2策略=4槽，composition每槽固定A+B两个program。旧全42候选两臂/168链目录作为来源完整保留，不把本轮2个预指定program称为重评全42。其他候选不参与新评分，不在新分母中假记成功。缺A/B/context/合法见证/源closure时，原槽null/blocked，不换程序、不换根。

| 每根每策略自然完成新增资源 | composition | joint两轮 |
| --- | ---: | ---: |
| predict / AE / predictor / pred_proj（各） | 18 | 96 |
| literal raw J | 0 | 2 |
| mathematical requested VJP rows | 0 | 384 |
| J requested elements | 0 | 23040 |
| AD primal packet steps（已含predict） | 0 | 12 |
| encoder/projector视觉 / SVD / solve / physics / train | 0 | 0 |

all2总新增228predict，各228个三个native模块、4J、768requested VJP、46080 J elements、24AD primal packet steps。engine backward及FLOPs未知/null，不由VJP猜测，不宣称equal compute。composition18=3+3+6+6；joint96=2×42+12。两策略不得免费复用彼此forward/J。失败intent不退款，未返回与未开始分开保留；超时不得把pending记complete。

继承成本单独列：原root编码/170J/原三程序全库预测及已执行wall-capped/physical/RGB成本；只引用原账，不把它们记成本轮新增，不与新4J相抵，不将已付模型评分当免费的新context转移。

拟议caps（待MAIN批准，不是运行授权）：whole-stage1800秒（verify前起计，含读取、加载、计算、日志与封存），每策略槽search450秒、allocated4GiB/reserved6GiB、输出128MiB、应急reserve2MiB、磁盘floor200MiB。whole-stage与各槽双约束，不能把四个450秒再加加载称仍符合1800秒；不足时保留partial。已有同步callback可见overrun后禁新dispatch，超时不得签完整4槽；消费者必须校验同一wall边界。输出caps不扩大已有输入reader caps；完整旧source/root数组按其真实封存ABI读取。具体host实现需MAIN审批后另做，不继承旧硬1068forward cap为新合同。

## 6. 可执行接点和未来输出 schema（当前没有runner）

当前无需新数据获取。MAIN只读检查可执行：

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
sha256sum results/agent_p2_saved_program_linker_20261029_v1_MAIN01.json results/agent_p2_compute_matched_protocol_20261029_v2_prepare_MAIN01.json
jq '[.catalog[] | select(.arm=="joint_H3_no_composition" and (.candidate_index==1 or .candidate_index==2)) | {id,status,planned_steps,committed_steps,program_sha256,input_context,scope}]' results/agent_p2_saved_program_linker_20261029_v1_MAIN01.json
```

未来新own-schema prepare接收 linker path/SHA及其 arguments.authorities 三path/SHA，绑定全部实际引用capsule/program_sha、root/context/goal/model/normalizer/domain与source闭包。继承MAIN完整receipt和linker引用IO证据，须注明full_upstream_rebuilt_here=false；独立重核新调用全输入/输出、完整context、rawbits和新ledger，不伪造旧consume成功。新模型权重before/after不变、TF32/autocast off、MATH/nativefloat32、HostCUDA显式授权不可省。

未来固定输出每row：root/arm、A/B原id与program_sha、old_B_input_context、new_B_input_context、六raw packet与全部context、逐call模块IO及array SHA、initial/finalQ、全部joint两轮5候选/新J、whole/segmented witness、继承/新增计账、stage与slotwall、missing/error。completed=false时终态及Q=null，不凭父context补全。原scope与新scope分列，父程序原root输入永不改写为新CA。

本轮只有本协议，无prepare/run/consume CLI；上面是实际已有保存资产的可执行读取接点，不是假装新planner已可运行。MAIN批准后再实现独占runner+strict，模型输入封存后仍不读物理标签。科学P2、arbitrary目标tracking、fresh资格、物理部署均不是该runner产物可签的权威。

## 7. 本轮 RAM 核验与剩余风险

已在内层 `.venv/bin/python -B` 做纯NumPy/保存JSON核算：四个指定capsule均linked_saved_program、planned/committed=15；两根联合域与literal tanh/rawbits检查通过。合成完整history转移验证 R(R(C,A),B)=R(C,A+B)，且从旧C直接运行B不能代替新CA；无模型类/torch导入、无旧producer consumer调用、无输出结果写入。预算整数式228/4/768/46080/24核对通过。

这些检查没有验证真实新context预测、六包AD、模型泛化或任务效能。主要剩余风险：H6新时域/外包域需要明确批准；joint更大动作支持和更多计算不能称纯模块化因果；旧模型在长滚动/迁移context外推未知；仅known2且程序有原目标条件化来源，不是独立确认。若MAIN只批准原full15时域，就应拒绝本协议并另定真正两程序的时域合同，而非偷偷拆包或把现有三程序别名重跑。
