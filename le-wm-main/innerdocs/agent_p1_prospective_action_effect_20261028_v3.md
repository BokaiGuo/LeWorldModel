# P1 v3：仅 saved consume 加固，不复制生产 runner

v2 source/tests/doc 原 SHA 全部固定；v2 prepare/run 和原始输出完全保留。本 v3 只新增独占 source/tests/本文。无新模型/环境/训练/拟合/物理调用，也无新 prepare 或 run CLI。MAIN 可运行已冻结的 v2 producer，然后对同一 report SHA 运行 v3。

新增两项强制验收：

1. 对**完整、未投影** canonical hashchain 的每条事件核精确 kind 白名单。该集合来自 v2 runtime 及实际调用到的原物理 helper emitter，含合法失败事件；未知 kind 无论在 replay20 前后、有无 job scope，一律拒绝。其它 producer 的 `operation_completed` 等不是该 runtime 的事件，不因旧20步投影过滤而放行。
2. 读取每个 saved uint8 RGB224，调用冻结 `agent_p1_teacher_library40_encode_20261028_v2.preprocess_pixels` 独立生成原 `[1,1,3,224,224]` float32 preprocessing，与 saved preprocessed 的 dtype/shape/bytes **完全一致**才通过。原 helper SHA `f7e3abb842cc185c77fe217922cd83ce709f4a6bf8fce0d9337c87bf4f66f75b`。不能以 preprocessed==encoderinput 的自洽关系代替像素来源证明；联合篡改这两个张量仍拒绝。无容差或 normalization 调整。失败的 image attempt 只要已保存 preprocessing 也检查；前处理前失败且没有此回执的 partial attempt 保留未知，不伪造编码。

source/prepare、artifact census、native20 原冻结 trace/payload consumer、全25步原生/guard/renderer、模型模块桥、selection 时序、全部40槽/80对等仍复用固定 v2 consumer 代码强制执行；仅下节已经独立验证的 terminal event 信封比较由 v3 局部适配，非另一套替代 native 验收。输出保留复用代码的结果、明确的 compatibility 记录及新增 event census/preprocessing 检查、原冻结源和本三个文件 SHA，**不声称原 v2 consume 自身通过**。原 producer 目录只读；新 receipt 必须放到目录外的新 v3 stem 路径。完整/partial 状态沿用原证据边界，均不签 P1 admission。

## 实际 v2 line630 失败：完整保留，限定适配

MAIN 的原 v2 consume session86151 exit1、没有 receipt，失败于 `all trajectory frames and terminal triplet join global journal`。worker 对实际 report seal `2e7794d32602ca2b045202099fedb6b99ac5988d6255ee68870032219530084c` 和完整 journal 只读复核：全部40/40 terminal triple 的差异**仅为 journal 多出 `index/previous_sha256/event_sha256`**；只移除这三字段，40/40 payload canonical byte-identical；另40/40所选27帧 trajectory receipt byte-identical，mismatches=[]。因此此失败是 producer `Store.observations` 保存原始 event body，而 `Store.emit` 加 hashchain 信封后，consumer 错把两个表示整体直接比较；不是这次检查发现的真实状态/RGB/raw差异，更不是原 P1 理论负结果。

v3 `terminal_projection` 对每条已知 terminal row 要求 journal event **恰为7个原body字段+上述3个信封字段**，report triplet 恰为7个body字段；信封类型/hash、原完整 hashchain、row/anchor/word/job、suffix10=absolute25、triple次序、全部pre/post/receipt/snapshot/state/PNG/pixel逐字节关系都验证。不能删额外“metadata”，不能递归忽略 receipt 内任何字段。每条实际完整 triple 的原canonical bytes只映射到其已经独立证明完全相同的原body bytes。

实现用 `FunctionType` 复用固定 `v2.consume.__code__`，建立一个**局部复制的 globals namespace**，仅替换该局部实例的 `canonical`：当且仅当 canonical bytes 恰为上述已验证实际 triple 时采用映射，其余任意值完全原样。v2 模块、函数对象、全局 canonical、任何helper都不修改；实际 journal/read_events/hashchain 不投影，原 chronology/state/raw/metric/guard 校验仍用未改事件。要求每个已知row恰好发生一次该比较映射，receipt记全部使用次数；并记录原v2失败和这个兼容路径，而非伪造原v2成功。该局部改动的作用仅修复 line630 的表示比较，数据/门槛容差为0。

这两项是实现/验收完整性缺口，不是已验证的原 P1 理论失败、模型反转机制或 gate 失败原因；不改变任何 native metric、阈值、原目标、根/词/对选择与 freshness 规则。

工作目录 `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p1_prospective_action_effect_20261028_v3.py
.venv/bin/python -B -m experiments.spectral_reachability.agent_p1_prospective_action_effect_20261028_v3 consume --output results/agent_p1_prospective_action_effect_20261028_v2_MAIN01 --report-sha 2e7794d32602ca2b045202099fedb6b99ac5988d6255ee68870032219530084c --receipt results/agent_p1_prospective_action_effect_20261028_v3_consume_MAIN01.json
```

CLI 中 report SHA 是 MAIN 通知的真实 v2 report seal；worker 没有重复该实验，也未在 source-first 交付前自行生产新 consume receipt。测试仅合成 saved mutation 与既存40库的只读事件集合，不执行真实 v2 prepare/run。覆盖任意事件位置未知 kind、错 producer kind、原224像素桥、同时更改 preprocessed+encoderinput、重签像素、partial preprocessing，以及不可绕过原 consumer。测试不是新 CUDA/physics 结果。

新增实际shape回归使用上述真实report首行body，并构造带三信封字段的canonical journal事件；复现原比较必失败，再验证局部投影通过。对state/snapshot/pixel SHA/step即使重签事件hash仍拒绝，额外metadata或坏hash拒绝，v2全局canonical始终不变，映射使用次数必须恰为1。

## 实际 saved consume：未通过；继续保留精确 native 检查

worker 已只读调用一次当前 v3 对同一实际 report 的消费（未写 receipt，无新实验）。信封比较已越过，随后仍被冻结 v2 line635 `same-task official native flags and full7 distance, including velocity` 拒绝，exit1。不能把这次消费称为成功，也不输出新分析结论。

只读逐步定位1000条 raw_return，得到：warmup400条中400条不匹配最终 row.task goal，但**400/400精确匹配该分支 `placement_recipe.native_goal`**；prefix200条和suffix400条均与最终task goal精确匹配，0差异。第一例 row0/warmup1：保存 distance=`56.512040741808306`，placement goal 重算完全相同，而用之后绑定的task goal重算为 `242.84466686189273`。原 `phys.episode` 明确 warmup 先用 `place()` 返回的 native_goal，10步后才 `bind_goal()`；v2 consumer错误对所有步一律用最终row.goal。这是另一个 consumer 的时间/goal协议错位证据，不是允许更改 native metric/goal，也不是原P1负结果解释。

**当前v3仅完成信封适配+两项加固，未对这第二处native-goal检查作任何适配、跳过或容差。** MAIN 若运行当前CLI将继续看到line635阻塞。下一版若处理，必须按原真实 goal绑定时序独立逐条核原native公式、完整goal/state与绑定事件，不能简单放行该错误。当前冻结保留原始失败和新失败，尚无严格消费成功receipt。

最终联合验证：v3 **22项** + 冻结v2 **27项** + 冻结v1 **34项**，共 **83 passed**；测试通过不是上述实际consume通过。v2 source/tests/doc三个冻结SHA未改。
