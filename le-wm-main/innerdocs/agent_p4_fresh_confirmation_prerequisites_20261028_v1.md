# P4 fresh confirmation：有界资产核查与准入缺项

2026-10-28；仅新增本文。旧 P4 source/tests/doc、MAIN源码/结果及dirty改动均不编辑。
本轮只读文件身份/预声明/来源metadata、运行既有纯metadata接口和RAM测试；无神经、
物理、网络、训练、重采样或下级代理。不重新打开候选 planner/native outcome来挑根。

## 当前结论

**没有在本次限定核查中取得可直接使用且获认证的“真实未打开 fresh root”资产。**
现有四槽是事前固定的未来候选规则，不是四个已采集的物理根：root manifest及consumer v2
各四行 `actual_root_identity=null`，`source_run_authorized=false`，
`freshness_certified=false`；v2 `certified_fresh_roots=[]`。
不能把null改成新目录/seed hash，不能从已知seed无冲突推出global freshness。
“从未被任何人打开”本身没有完整访问/披露记录，本轮判定 **unknown**，不伪签未打开证明。

MAIN通过的rolling消费回执：
`results/consume_zero_and_rolling_MAIN_20261028_v1_rolling_MAIN01.json`，
SHA `182d5ce14a6363b287607c511d3bc4dde85769618e8c8082029f090fb19c1232` 已实际重核。
它证明既有known-root接线，不新增根数、不证明freshness。
P4 MAIN02 prepare SHA `8528e8d7f9b688b1c93be154305ccc35b8ed6218de63e736b1ae567691c432a2`，
四层仍为原source_slot0/1各GN/D2P，I/R共8臂，**独立任务n=2，不是8或4**。
其原sourcepack仍是
`results/p1_constructive_moving_history_source_20261027_v1_main01/sourcepack.json`，
SHA `c2d8d573f94884ffc081163f25f2bfa269f98ffe8155887704109219910ffec6`。

## 已有可复用规则：不再造选根器

外层 `../docs/chat_freshroots_20261027_preregistration.json`，
SHA `79e1f45e79d692cae72376c91a4830c94b6ac1f53bb346eeaa085715e4197d7e`；绑定协议
`../docs/D2P_MOVING_FRESH_ROOT_PROTOCOL_20261027.md`，
SHA `42f1bcbf517dfd46b8ba05525663d82efac32dbb5fb24f87a8ca572bef507a2f`。

|slot|root_seed|reset_seed|witness sign|
|---|---:|---:|---:|
|0|203027101|203027201|+1|
|1|203027102|203027202|-1|
|2|203027103|203027203|+1|
|3|203027104|203027204|-1|

规则是固定slot顺序、`SeedSequence([root_seed,0,0])`、每槽一个proposal/attempt，
分母恒4；无候补、重试或按outcome/cost/difficulty排序。冲突直接拒绝，不自动递增seed。
warmup原float32前5 `[.1,0]`、后5 `[0,.1]`；原recipe/witness不改。
保留duplicate/guard/degenerate/partial/not_started及负结果；合法source发布k/4不得冒充4/4。
source资格不是planner胜负，失败source不能无声从整体报告删除。

历史排除须跨root/reset两个seed命名空间及预留span，覆盖失败、未选、重启、goal-render/
explicit-state reset。物理去重除root-goal hash，还须state7、wrapped pose、snapshot、RGB3/raw10；
换goal/velocity/角度2pi/renderer不能让旧物理pose变新。
沿用原协议固定XY≤1e-9 pixel及wrapped角≤1e-12 rad的数值别名排除，不调整阈值。
独立性检查在witness dispatch前核实际根；不能等看到planner/native结果才决定保留谁。
部署只能获得blind RGB3/raw10/goalRGB，不能用witness或state选动作。

## 精确缺项与接点

|缺项|目前真实状态|需提供什么；不能怎样替代|
|---|---|---|
|原历史reset覆盖|现有catalog8来源、16继承refs、10次已发现dispatch；认证候选0|原A–H逐调用完整reset ledger，或原加载闭包+可穷尽包络；B三次含前两次失败，不能仅拿最终report|
|历史域及外部批准|旧618快照及后续1033路径差集并非全历史|MAIN给出机器/工作树/时间cutoff、失败/未选/预留及后续delta路径SHA，明确scope的外部批准；不自置approved=true|
|实际新根身份|四候选actual_root_identity均null|未来获授权采集后实际state7/pose/snapshot/RGB3/raw10及历史/批内排重；新seed本身不够|
|真实未打开的blind资产|当前固定manifest无获批新blind路径/SHA，未取得其披露记录|实际来源封存与访问/披露边界；缺记录写unknown，不把“尚未采集”写成“已封存未打开”|
|新source producer|协议提出p1_constructive_moving_fresh_source_20261027_v1；本轮文件名核查未见该实现|新独占source/tests/prepare及完整durable原生source合同，MAIN审读执行；不得patch旧两根source的slots/globals|
|四新槽下游接线|现有P4固定旧source_slot0/1和旧task exports|新适配器显式获批4blind→16图像编码→真实task exports→新planner层→P4 driver；冻结旧文件，不假冒旧SHA|
|确认性P4合同/准入|known moving20步工程证据不等完整长时efficacy|P1及采用组件前置、冻结目标分布/终止规则/固定arm与完整资源合同、原native success及强对照；由MAIN另审，不借rolling通过放行|

原success位置含agent+object；object-only投影非正不等完整任务失败。unknown费用/native指标不能补0；
equal execution horizon不等equal FLOPs/Joules/总算力。新的四槽设计不能倒签为现有已运行8臂的freshness。

## 本次实际核查、证据SHA与限制

只读核对：

- `results/chat_freshroots_20261027_root_manifest.json`：
  `3627dd2561a496ea2a0831c2aafa11337363ac1e9ca01285907c9f20e3a7b838`；四行null。
- `results/chat_freshroots_20261027_consumer_manifest_v2.json`：
  `2b80412eb2efc7b721d6edb754ce5d9d5d513e17744706aa1e5bab56c785f7e7`；认证列表空。
- `results/agent_freshroots_resume_20261027_coverage.json`：
  `00a3fffeaaa8943e851d129b941a7f07feaa1db387117beb98110e4ec95f5eeb`；metadata范围有缺口。
- `results/agent_freshroots_resume_20261027_next_delta.json`：
  `5ef07ab7815fff0c392feef59309c0116596e95d541eb95413d1d966a8b2a29b`；本轮只核原envelope字节SHA，不重新展开或冒称重审1033文件。
- `../docs/D2P_FRESHROOT_AUTHORITY_REQUEST_20261028_JASON.md`：
  `46e9df59cb1f1d9bb344132a9a39a996c85cc26c5b79ef45c7b5d4dc4b3af39b`，已有精确A–H补证请求，无需重做。

本轮对inner `results/`、`experiments/spectral_reachability/`只做文件名发现：
basename含fresh_source/fresh_root/freshroot/freshness/exclusion/authority的regular file共39；
其中results路径未发现含response/approval/fresh_source的候选名称。
这是有界命名线索，不证明别处/不同命名资产不存在，更不是历史或披露全覆盖证明。
没有打开这39份文件的任意outcome payload，没有重扫巨大会话日志或运行旧source。

复用的纯saved/metadata入口已实际调用：
`agent_freshroot_authority_20261028_v1.requests`+Reader重核固定catalog后返回8/16/10、certified=0。
其source SHA `f7a30fcad18483e1848170acdd1fe4bd96e7d08b38804b18f1cf2cf1087870e7`。
既有RAM回归本轮实际 **38 passed in 0.13s**；这是schema工程测试，不是freshness科学证据。
本任务只需文档，没有新增可选checker：已有接口足够描述缺项，复制验证框架不能产生原历史事实。

## MAIN下一条可执行命令

从inner目录运行（只读stdout，无模型/物理；不授权后续run）：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_freshroot_authority_20261028_v1 request
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_freshroot_authority_20261028_v1.py
```

下一实质动作是由MAIN向原运行方取得上述authority response/原始附件，并审查其真实性及coverage；
syntax/byte验证不能自行解除认证。未取得前，fresh source及confirmatory P4均fail closed。
本文不提供模型/物理run命令，不声称研究goal完成。
