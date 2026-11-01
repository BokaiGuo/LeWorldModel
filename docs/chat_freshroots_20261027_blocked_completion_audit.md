# 完成性核查：历史执行依赖尚缺，不签发freshness

2026-10-27，范围仅本独立新根goal。此文是新的完成性/阻断核查，不修改任何既有封存文件。本轮前两轮分别实际增加80份prefix身份、80份显式父源关联和84条历史消费index，属于真实进展；但同一历史执行绑定/覆盖批准阻断持续存在。本轮重新读README、主线程最新来源记录，并核查实际文件/测试，不依赖上一轮结论。

## 逐要求核查

|要求|当前权威证据|判定|
|---|---|---|
|path+SHA追溯历史身份|原root manifest、80项prefix supplement、consumer v2和ancestor links逐项精确绑定|已交付；不等于完整历史覆盖|
|能解析的unknown实际解析|80/80 prefix记录、8组、父源80/80唯一关联、16条E/H selected root-goal连接|取得进展；614原缺直接身份refs按旧合同保留，不能全部清零|
|历史reset上界及历史源码依据|重新执行原quarantine.build；固定8sources/16refs；Git/postrun/later-seal证据仍只能提供条件包络|未达成原执行完整认证；0/8认证|
|新抽样与失败分母事前封存|原preregistration精确SHA，slot0..3固定seed pair、4分母，无替换/重试/结果筛选|已交付，未执行抽样|
|机器root manifest及coverage证据|root_manifest、coverage、consumer_manifest_v2、prefix supplement均已保存且重核SHA|已交付；明确false而非伪pass|
|消费者及实际metadata闭包|本轮重新运行consumer v2得到84记录、80关联、0missing/ambiguity，四候选blocked|软件/metadata复算通过，不是source授权|
|测试及独立复核|本轮联合199 passed in 1.07s；上一轮80条父字段及SHA/pointer独立复核PASS|软件合同证据，不是freshness|
|全历史coverage批准|没有收到外部MAIN精确content-SHA/scope批准；618仅旧快照|未达成，不能本对话自签|
|四个实际fresh根|四候选实际state仍null，全部blocked_missing_actual_identity|未达成，certified_fresh_roots=[]|
|不重复P1/不训练/不降CPU/保留旧结果|全部新增本前缀；神经/物理调用0，未使用子智能体/删除/推送|本对话未突破权限边界|

因此完整科学目标**不能标complete**。已有候选清单仅供MAIN事前确认，不可作为已认证独立实验根或运行许可。

## 本轮现场查核范围

对原8个运行目录执行只读文件清单检查：

```text
pusht_d2p_p0_packet_contract_v2_20261026_cuda
pusht_d2p_p0_preflight_20261024_cuda
pusht_d2p_p0_preflight_packet_layout_repair_20261024_cuda
pusht_d2p_p1_outcome_admission_20261024_cuda
pusht_d2p_p1_packet_contract_v4_calibration_20261026_cuda
pusht_d2p_p1_reconfiguration_replay_bound_20261024_cuda
pusht_d2p_p1_rollout_alignment_repair_20261024_cuda
pusht_d2p_p1_root_anchored_v5_calibration_20261026_cuda
```

各路径以 `/home/gbk/code/LeWorldModel/le-wm-main/results/` 为根。共找到8份report，另E目录三份postrun源码；这些原目录没有新增原执行prerun/attempt ledger。此结论限定所检查目录，不声称所有机器/会话/离线备份均不存在。

对docs/innerdocs/results的文件名只读检索coverage approval、fresh approval、reset ledger/attestation、authority approval、delta inventory标识，没有发现可接入的新批准资产。文件名检索不是coverage证明；更重要的是**本对话尚未收到任何可认证的外部MAIN批准文件/精确SHA**。主线程最新记录仍称两个moving roots为known calibration；其新实验不提供本任务缺失的原8run历史执行绑定。

重新运行历史quarantine消费者，仍返回8/16、certified_source_count=0；辅助Git/快照/封存字节可核，但它们不证明原8run的执行绑定。不得以同schema、当前代码、postrun、后续seal、一次重建复现或条件8n span替代。

## 恢复推进所需最小外部输入

1. 原A–H逐项report SHA→实际执行producer blob→collector/下游/env闭包的原始绑定，或与report绑定的完整reset/attempt ledger；必须包含失败、未选、重启与预留。F需确切v2证据，E/H不能倒灌postrun/后续seal。逐项path/SHA及missing清单见原manifest `historical_reset_evidence.rows`。
2. MAIN给出覆盖全部相关family、失败/未选/预留范围的显式authority及618快照后delta path/SHA清单，裁定original_four等范围；提供精确content-SHA及scope批准。本对话不自签、不重标family、不用采集新目录冒认证。
3. 上述满足后，才可由MAIN按已封存四槽协议生成实际根，并在witness dispatch前提供历史/批内state/pose/snapshot/RGB3/raw10排重证据。四槽目前不能启动。

完整根身份缺项还需原字节绑定的seed/reset/snapshot/history等桥；已核父源只定位祖先/分支，不等价于采样state。继续添加synthetic fixture或同一metadata的另一份summary不能产生这些历史事实。

## 交接入口与SHA

当前首选消费者manifest：

`/home/gbk/code/LeWorldModel/le-wm-main/results/chat_freshroots_20261027_consumer_manifest_v2.json`

SHA `2b80412eb2efc7b721d6edb754ce5d9d5d513e17744706aa1e5bab56c785f7e7`。

完整历史seed/span与8/16证据仍由其精确SHA绑定的base manifest提供：

`/home/gbk/code/LeWorldModel/le-wm-main/results/chat_freshroots_20261027_root_manifest.json`

SHA `3627dd2561a496ea2a0831c2aafa11337363ac1e9ca01285907c9f20e3a7b838`。

源码、测试、父源链接及其他SHA见 `/home/gbk/code/LeWorldModel/docs/chat_freshroots_20261027_consumer_v2_handoff.md`。本轮重核的prefix supplement SHA仍为 `2cd81f5741fbef9fea0eae62b913e9ad73b09fa8c9d02e525876d6854a91d7e9`，ancestor links SHA仍为 `4d52a593773bf7dd5c323c61b2a1e68e43941a56ceef67609c7dbcb4075189ca`。

该阻断在连续多轮goal中均实际存在，且可安全取得的父源/metadata已推进到可审计交接；当前无法自行补出原运行事实或外部批准。本轮将goal标blocked而非complete，等待上述真实输入，不继续重复同一缺口声称进展。
