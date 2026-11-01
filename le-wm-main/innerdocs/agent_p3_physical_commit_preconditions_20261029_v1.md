# P3 physical commit：最小saved输入解析与fail-closed manifest

仅本stem source/tests/doc；不改canonical、封存源码/结果或comparison v2。使用experiment-code技能组织持久保存和RAM回归，不跑生产prepare/NN/physics/fullstrict，不开agent。Bohr正在实现R orchestrator，本模块不假定它的未来report/receipt schema、字段路径或SHA。MAIN须拿实际产物后显式绑定；本轮不创建虚构ready示例产物。

## 可执行入口

cwd `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
.venv/bin/python -m pytest -q tests/test_agent_p3_physical_commit_preconditions_20261029_v1.py
.venv/bin/python -m experiments.spectral_reachability.agent_p3_physical_commit_preconditions_20261029_v1 --template
.venv/bin/python -m experiments.spectral_reachability.agent_p3_physical_commit_preconditions_20261029_v1 \
 --contract ACTUAL_TYPED_CONTRACT.json --contract-sha ACTUAL_CONTRACT_SHA \
 --output results/agent_p3_physical_commit_preconditions_20261029_v1_MAIN01.json
```

template只向stdout给四槽null输入骨架，`build(template())`实际返回四槽BLOCKED。CLI输入和所有documents为绝对实际路径+外部SHA；new-only输出限制本stem/16MiB/xb/fsync，拒绝覆盖。没有run/CUDA/physics入口。

## 通用typed contract：不冒充producer schema

顶层精确字段：`schema=agent_p3_physical_commit_preconditions_20261029_v1.input, documents, authorities, slots`。

documents是别名→`{path,sha256}`；缺实际产物填null，不编造未来SHA。authorities精确为planner/physical/history，每项null或：

```text
{report:报告别名, receipt:完整strict别名,
 report_schema:实际schema, receipt_schema:实际schema, receipt_status:实际完整strict状态,
 report_sha_pointer:receipt中绑定report SHA的JSON pointer,
 source_bindings:[{file:{path,sha256},sha_ref:{$ref:report或receipt别名,pointer:其源码SHA位置}}]}
```

MAIN在运行前固定这些实际authority身份/ABI；本工具不因任意status字符串自签外部strict真实性。外部SHA由MAIN认证，本工具只检查实际字节、schema/status与报告SHA/source指针join，不导入或调用其consumer，不重复fullstrict。每个源码绑定不仅核本地SHA，还必须等于对应报告/receipt里的真实字段，renderer SHA须在physical authority源码绑定中；模型source before/after也逐文件核对。模型loaded_bytes必须为实际正整数，不能用文件个数冒充读取字节。

slots固定`(slot,root,mode)`为`(0,0,fixed),(1,0,adaptive),(2,1,fixed),(3,1,adaptive)`，每项字段`slot/root/mode/projection`，projection为null或以下八个逻辑section。每个section可以直接是证据引用，也可结构化组装字段：

```text
task: {$ref:planner_report别名,pointer:/实际task路径}
release: {
  plan_available: {$ref:planner_report别名,pointer:/实际布尔字段},
  ...各个逻辑字段逐项引用实际位置...
}
```

递归投影只接受`{$ref,pointer}`或非空字典结构；不接受literal数字/字符串/布尔/列表作为证据。数组整体用真实pointer引用，禁止把缺失字段补成true/空errors/合成latent。planner section只能引用planner已认证report；anchor/restore_reference只能引用physical已认证report；actual_history只能引用history已认证report。其他文档即使有SHA也不能绕过角色认证。引用不存在或证据错位ValueError；整section缺失/null、authority缺失、documents null或没有released plan生成BLOCKED，不生成可执行动作。

这八项是**本工具的显式typed ABI，不是声称未来producer已有同名字段**；可通过pointer组装原字段，无法映射时保留null/BLOCKED。若未来真实语义不同，需要另版显式适配并审读，不能伪造旧receipt或在合同填常量“通过”。

## 八项逻辑输入及检查

|section|精确逻辑字段与保存核验|
|---|---|
|task|原core task：binding/root_context/goal/projection/action_mean/action_std；core.validate_task核原task/goal/Q/normalizer/context/action-domain digest，192维|
|anchor、restore_reference|各含root_sha256/task_id/condition/goal7/goal_pose3/renderer_sha256/checkpoint/prefix51/warm_raw10/prefix_raw5。checkpoint含absolute_step/plan_step/png/file_sha256/pixel_sha256/state7/snapshot。实际cp5=warm10后absolute15；两者除checkpoint PNG路径可不同外，其余完整51/raw/state/snapshot/PNG字节hash/renderer/goal精确相同，不用浮点容差|
|actual_history|frames3/raw10/latent3/history_tokens/goal192/normalizer_sha256/context_sha256。frames格式同checkpoint，绝对步5/10/15（局部plan -5/0/5），每帧绑定同prefix51对应off帧；末帧必须精确anchor cp5。raw10严格warmup尾5+prefix5，不能用新plan full10或imagined history替代|
|model_input|latent3/history_tokens/outgoing_tokens/raw_commit5/goal192/input_fields。input_fields精确`[latent3,history_tokens,outgoing_tokens,goal192]`，原raw5另作为动作见证；native state只参与评价/restore，不进planner。两个history token均重建，outgoing为两history加一原始raw5归一化token|
|release|plan_available/status/released_plan/core_plan/whole_replay/segmented_replay/guards_passed/errors/context_sha256/task_sha256。typed状态必须released、真实gate=true、errors空、完整released==core plan、两replay等于完整path，原checked_path认证全部packet history，不从diagnostic_best救援。原始producer若没有可映射的这些语义则BLOCKED，不擅自推断|
|resources|真实原core Ledger receipt完整27轴/全部events/caps/attempted/completed/计时/unknown。调用纯saved core.audit_ledger核逐事件delta(shape/directions)/scope/顺序/status/计时及合计，不是fullstrict；released要求全部returned、attempted==completed、replay_calls≥2。无release也审其已收费账并保留attempted/completed，不退款|
|model|expected_model_hash/before/after/loaded_bytes/sources_before/sources_after；模型state hash与task binding一致、前后完整一致、正loaded bytes、源码前后一致且实际SHA匹配。不会加载权重或断言训练来源已认证|

prefix51严格17组triplets：warmup0..10、plan0..5，每组overlay0/1/2和off/on/off、pre==post同state、receipt state==snapshot.state；cp5须关联plan5 overlay0。三张输入PNG独立decode为原224×224 RGB，核文件SHA与真实像素SHA，拒绝错时间/错PNG/错snapshot。本模块不重新编码latent；latent与已认证history report及完整task context逐位绑定，encoder来源由外部history完整strict承担。

所有raw须lossless float32、合法[-1,1]、Nx2；计划长度为正packet倍数且≥5，首5raw一次提交。逐元素NumPy float32 `(raw-mean)/std`，不/100、不clip、不tanh；两个incoming token及outgoing3完整逐位校验，保留负零。mean/std来自同原task，std>0。自己compact JSON摘要用io.digest；task/context/normalizer/plan用原core.digest，显式区分schema编码，不拿一个codec替换所有producer身份。

同root两策略必须完整task/anchor/history一致且时间同actual cp5；两个root身份不得简单复制。同一个root的候选/分支不变成新根，fresh_roots=0。

## 输出语义：没有READY、没有真实restore证书

- 缺证据/无release：`status=BLOCKED`，四槽保留，缺输入字段明确列missing/reason；无release的完整失败账保留尝试费用。不发布该槽commit_raw5。
- 全部typed保存证据通过：`saved_inputs_validated_pending_runtime_guard`，只保存首5raw/hash、task/context/anchor/plan身份、资源合计和证据pointer。不是环境已经restore成功，更不是物理运行授权。
- malformed/tampered/source不闭合/错资源等直接ValueError，CLI不写成功manifest；不吞异常伪装成可用输出。

所有状态`ready/physical_release/execution_authorized/P1/P3/C3/freshness/new_independent_roots=false`，每个已验证槽`runtime_restore_proven=false`。真实driver必须在**新环境public_step6之前**独立完成reset/warm10/prefix5及51帧/cp5 state/snapshot/PNG/raw/goal causal guard，durable记录后才能dispatch首packet；本工具不执行环境，不可能预先签这个事实。不把本manifest当Bohr producer receipt、physical prepare或canonical gate。

## 回归与待接线

最终本文件回归41 passed/2.40s，三文件git diff --check通过。RAM fixture仅用于typed ABI，不是真模型/物理数据。覆盖null骨架BLOCKED、完整四槽仍ready=false、不同known-root同root对照、三个actual时间与prefix51、PNG字节/像素、signedzero、wrong rawdomain/length、双history/重复归一化/outgoing、release/guard/replay、source authority pointer join、loadedbytes、模型/hash、27轴及unknown/失败不退款、source越角色引用、JSON pointer escapes、实际CLI新写/拒覆盖和无torch/pymunk冷导入。

未完成项明确为：MAIN提供Bohr实际planner report+完整strict、原physical和actual-history完整strict与实际pointer合同；本模块不认为它们尚未实现，也不猜未来schema。若实际报告缺typed所需语义，保持BLOCKED并列缺字段，再由MAIN决定新增最小适配证据。生产NN/physics/fullstrict与真实restore guard仍归MAIN，不由本轮运行。
