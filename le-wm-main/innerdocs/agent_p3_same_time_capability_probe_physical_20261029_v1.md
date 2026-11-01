# Stage C：10条U/Z2自然重放部署器的schema/RAM准备

本版**不是生产部署器**，没有生成prepare或未来authority。仅本stem源/测/文；完整读取上位协议全部阶段，按阶段C及第7节预算实现纯事件状态机、有界落盘和fixture自有consumer。协议SHA `b35af390fc74e6ebd44a9879784308ec74947fb28f640ce5e795cc263e65dccf`。experiment-code技能仅用于fixture、失败运输和严格状态机测试，不训练/择优扩搜索。

## 1. 已核接点与精确停止线

原冻结`agent_p3_secant_controls_first5_physical_20261029_v1`的`guard_proof/Meter.call/Store/publish/endpoint_rows`给出51guard先于control6 intent、原生hook计账、排他64MiB运输和实际同时间checkpoint语义。本版不import或调用这些producer/fullstrict，不修改它们。

Jason的`agent_p3_same_time_cp10_rgb_bridge_20261029_v1.fixed_rows/consume`已由MAIN实际HOSTCUDA及完整consume完成；本worker仅核源码与receipt字段ABI，不重复consume。四个实际权威已记录于源码AVAILABLE_STAGE_A（不是未来占位prepare）：

- prepare：`results/agent_p3_same_time_cp10_rgb_bridge_20261029_v1_prepare_MAIN01.json` SHA `f2fac71516c36bdb654a9a84bbc0115d6155c04ed10231bc6b87b99ed0bda846`。
- report：该stem `_cuda_MAIN01/report.json` SHA `b5744f311d3f412cfc5016bf37c8854cc7b5272a6c37e62055c7f3aa68f2c657`。
- completion：同目录 `completion.json` SHA `37588f050ed9381b191673e27c0b6fcab3ddf0c726e63a54bb75d9579963facf`。
- 完整receipt：`results/agent_p3_same_time_cp10_rgb_bridge_20261029_v1_consume_MAIN01.json` SHA `af67caf8536a9637d709fcfb11c947d993e566bd4de5914725f4640a75de4d6f`。

- 固定顺序root0 C/Z/P、root1 Z/P，5行；`fixed_rows`连接原plan与actual response10，full51、66图、cp0/5/10及off前两帧身份。
- 行含`source_receipt/source_report/source/history_sources`、`causal_raw10/causal_raw10_bits`、`normalized_action_history`、`original_native_goal`、原cp5 task；最后history为absolute10/15/20。
- 实际`consume`返回`tasks[5]`、counts、bindings、completion及source_authorities；任务行含`task/task_sha256/identity/source/causal_raw10/absolute_steps`。实际闭包记录source SHA `40d51e4d5ad5fce5db9417fecc639e0288f286b678a5e7b0eb4e22adea392b77`。仅阶段A输入桥已可用，不等于U计划已可部署。

Pasteur七signal/五U producer尚无最终冻结ABI/receipt交付给本任务。**不得猜其字段名、构造一个外观合法的producer receipt或生成占位prepare**。下一生产接线至少需MAIN提供：

1. 阶段A四权威已齐；仍需阶段B最终source/tests/doc闭包及真实prepare/report/completion/全consume的路径/SHA，生产接线须精确join两者report、全5/全7分母、frozen model与原task/normalizer/Q。本版不从阶段A发明阶段B的路径或SHA。
2. 每端点原生source reset/seed、warm10/role raw10、完整51与66观察的原始pre/post/receipt、state/RNG/goal/public与actual response10。bridge单张cp10 PNG/history3不等于完整66物理guard引用；须由其source_authorities回到已认证原C/ZP物理引用，局部认证边界待MAIN审定，不回调fullstrict。
3. 每U原(5,2)f32 raw、一次chart/normalize证据、同actualcp10任务、whole/segmented两个完整模块IO逐位相等、实际部署tokens、源码/模型前后及完整producer消费回执。仅whole输出相等/只有MSE相等/线性solve值均不够。
4. 已审原生extractor映射到下述full witness与事件：须保留原51/66每条pre/post/receipt全部字段、原goal/RNG/public属性，不能只保state7代替；原snapshot的已有字段**不假定包含rng**，本schema的snapshot.rng/public_state是将原source witness无损装入的待审边界，不是宣称原native字段已经存在。此映射尚未实现，因此生产明确不ready。

冻结之后需另NEW接线版本、MAIN审读，才能实际prepare→native run→生产consume。当前`prepare/run`无论参数如何均报具体依赖缺失、写入0文件；P1科学false不是拒绝工程运行的原因，缺的是真实authority及尚未审定的ABI/extractor。

## 2. 固定10槽与两道guard

job0..9分别为五端点各U/Z2。每job独立constructor→原reset/place→warm10→prefix5→角色raw5→probe5，无live env共享、hydrate/poststate reset或goal变更。每job正常25public/254physics/81explicit renders；10job正常250/2540/810。每资源actual intent/returned/failed独立计账，partial不退款、不补步。

51观察完成、actual control5 return及原raw/history/RNG/goal/public exact，才可durable accepted guard并派control6；66观察及actual control10/control10 raw history exact后，第二道durable guard才允许control11。守卫写入/fsync失败不能激活；失败不可重试挑过门。逐帧source引用、动作顺序、所有overlay、PNG/pixel、完整snapshot均保留，不以RGB或state单项相同替代full guard。

本fixture绑定严格标`fixture_only=true`，不可变成生产证书。U无双重放时raw15=null，blocked槽无constructor/动作、endpoint/null及费用保留；Z2独立槽仍为原10动作加5个little-endian f32正零，不因U缺失替代其成果。有效U仅接受fixture中明确提供的raw和双IO等值，**这项RAM检查不认证真实producer**。不搜索、不再次tanh/clip。

终态只有实际plan15、absolute25；earlystop未到则null，不把early frame移作cp15。原native terminated/truncated保留，禁止之后public动作；原模型能力值/任务成功/native指标在本运输层未知null，不制造成功。所有10槽和原8逻辑比较分母将由后续生产汇总保留，本版不输出能力对比结论。

## 3. 全运输shape与硬上限

每job正常计数：constructor1、close1、reset2、set_state4、goal_render2、goal_bind1、goal_setter3、public_step25、physics_step254、explicit_render81，共374个operation receipt；intent+terminal=748事件。另81观察、2guard、run_started/branch_closed/run_finished三事件，共834。多层operation原嵌套返回可保留，不能把physics全部折为单个public成本。状态机逐资源硬cap，不以失败或partial退费。

81图=`3×[(warm0..10)11+(plan0..15)16]`。每图RGB224 PNG文件保守上界151834bytes，81图12298554bytes。每个PNG和实际解码pixels均SHA核、排他写、全部81保留，即使相同也不去重。

metadata覆盖**全部序列化副本**的强制配额：binding2MiB、journal20MiB、report14MiB、completion1MiB、failure1MiB，共38MiB。再加PNG上界及额外8MiB收尾reserve，保守总60533050bytes，小于64MiB（67108864）。非final写始终留8MiB；全目录aggregate cap64MiB；free floor200MiB；whole wall900秒，包括输出/hash/封存，completion后若超时额外failure记录并失败。真实上游映射后的metadata/原生PNG若不满足此硬配额，**先报MAIN，不截字段/降精度/减少图/扩cap**。目前证明的是该schema加硬配额可以运输，不是假称未提供的native extractor实际序列化已经合格。

全事件哈希链/canonical JSON、每PNG、binding/report/completion及metadata quota均独立核；report重复保存全部op receipt、81帧、guard/attempt、failure、pending unknown。自有`consume_fixture`从每个事件重建整份report/终态/counts、核完整文件inventory及所有PNG，输出明确fixture状态；partial/failed可保留但绝不称正常完成。`consume_batch_fixture`严格接收有序10个独立目录的实际临时SHA，逐一调用自有fixture消费器，全10槽费用与missing相加，禁止复用同目录或改job贴标签；仍不认证生产。无上游fullstrict/模型/物理导入。

## 4. 现在可执行的RAM命令

工作目录`/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p3_same_time_capability_probe_physical_20261029_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.agent_p3_same_time_capability_probe_physical_20261029_v1 describe
```

`describe`只stdout输出schema/预算/明确缺口，不写manifest。`consume-fixture --output <临时fixture目录> --report-sha <实际临时SHA> --completion-sha <实际临时SHA>`仅核本schema fixture；测试已覆盖完整81图/834事件的落盘→全量consume、早停、blocked U、失败、篡改/配额/超时final运输。MAIN当前不应执行`prepare/run`：二者按设计拒绝，不会生成缺权威prepare。

worker本轮不执行NN/physics/生产/fullstrict、不创建后代；科学所有门不变。交付后冻结本版；待真实两上游与extractor审计齐全，MAIN另批准NEW生产接线，不原地偷偷解禁。
