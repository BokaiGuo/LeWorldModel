# Intersection physical v2：P1 spec 时序及 ABI 核验方法

## 当前状态：前瞻时序未证实

本次只读检查时（2026-10-29 本机），没有发现 `agent_p2_intersection_physical_20261029_v2` 的 results prepare，也没有发现 `agent_p1_intersection_goal_adapter_20261029_v2` 的正式 spec。未检查或读取任何物理输出目录、report、receipt 或结果，不以文件未发现证明任何地方未执行，也不把文件修改时间当可信事前证明。未来出现文件时，本报告不会自动验证其存在或时间顺序。

已只读检查的物理 v2 源码 SHA `e2309903a600cc536d9d12ef746014ebfdaa0bea6a6097c326c43c46a9dbdc35`，文档 SHA `c0f92afc1dba78b2efe7f50ffeb9a7d15abc295879adabdc2fb9c0d1c91e435b`；这些是本次观察值，是否最终冻结须 MAIN 确认。物理 `contract()` 显式固定分母 6，要求 order 六项、两 known roots、off、H15、cp0/5/10/15；源码 `consume` 输出 `.consume`、六行、完整 plans、binding 和 `physical_report_sha256`；成功状态为 `intersection6_native_recorded_not_admitted`，部分状态 `partial_or_failed`。物理 adapter 是本次 intersection 模型 v2 的新物理执行，不是历史 P2 六槽。

P1 spec 投影目前源码 SHA `267f61e19c1967f35048efc9925c93c98ae7e038e042b343b4c258af6845f4cd`；物理 receipt 归一化源码 SHA `6d29bdc97ae3f8c919dcb59a3ceb1f8f7697f441018cca47ceac73a4f2d23637`。二者有 RAM fixture，但尚无真实物理 v2 prepare/receipt ABI 的本次实际验证，不能以 fixture 宣称真实消费通过。physical v1 不是此实验的正式输入。

## 要证明的时序与可信记录

需同时证明下列各项，不可由单个本地 SHA、mtime、`ps` 检查或个人自签推断：

1. MAIN 最终审读并封存 physical v2 source、tests、doc 和物理 prepare 的实际 SHA；prepare 必须含与该源对应的 bindings、六槽、原 root/goal7、cp0/5/10/15。外部来源及其已知局限须一同记录。
2. 在本次六槽物理 outcome 对制定者可见之前，MAIN 用该已封存 prepare 和 source SHA 生成并审读完整 P1 spec，固定六槽、两 known roots、每个 goal7、四 checkpoint、空 RGB pair；记录 spec 字节 SHA、完整 spec 和分析源码 SHA。空 pair 表示视觉碰撞**未验证**。已知历史 P2、当前模型计划/结果及其他已查看信息必须明确披露，不声称整体全盲。
3. 使用能够独立核查时间／顺序的外部机制，在物理 dispatch 前持久提交 spec SHA 和 physical prepare SHA，例如由 MAIN 选择的受控 append-only 审批日志或可信时间戳。记录事件 ID、提交主体、不可变凭据、数据摘要与签署／见证验证方式；若该机制是自签或可事后改写，仅算声称，不足以单独证明前瞻。此处不指定或伪造任何外部机构。
4. MAIN 的物理启动授权应明确引用该先前 spec commitment 与 prepare SHA，并留一个可核验的授权/dispatch 顺序见证。物理源码 `execute()` 的 `run_started` 只记录 physical prepare SHA 和 contract，并不含 P1 spec SHA；其 journal 可证明该运行绑定了某物理 prepare，但**单靠该日志不能证明 P1 spec 已事先封存**。如果没有关联的外部事件，时序为 unknown，不能补写为前瞻。
5. 未来 physical 完整 receipt 到达后，MAIN 核验 receipt/report/source/prepare 全 SHA 和外部 authority，再执行独立归一化、全分母和 saved verify；任何早停及失败留 null。事件与 receipt 不能自动证明制定者未看 outcome，必要时另有可信流程记录；无记录则仅报告程序时序，不宣称盲态。

此流程是待外部执行与核查的方法，不是已经完成的 attestation，不授权或运行物理；若 physical dispatch 已经发生而此前没有可验证 spec 承诺，只能以后验分析处理。

## 后续 source/prepare ABI 只读核查

MAIN 提供冻结 source SHA 和 physical prepare SHA 后，可对**prepare 本身**做只读 identity / spec 投影回归；不调用物理 `verify`（其会重建上游完整 saved consume），不读物理 report/receipt：

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
sha256sum experiments/spectral_reachability/agent_p2_intersection_physical_20261029_v2.py /ABS/V2_PREPARE.json
.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_agent_p1_intersection_goal_adapter_20261029_v2.py tests/test_agent_p1_intersection_goal_receipt_adapter_20261029_v1.py tests/test_agent_p1_prospective_goal_semantics_20261029_v1.py
```

经 MAIN 确认可在本次 outcome 前且允许生成后，才按 `agent_p1_intersection_goal_adapter_20261029_v2.md` 的 project 命令输出 spec、核对其实际 SHA 并进行上文可信封存。当前不执行 project；实际 prepare 若含缺计划、source SHA 或 contract/ABI 不符，现有投影 fail closed，不复制旧 v1 或补假字段。之后才可使用已有 P1 合同 `prepare`，其 `ready_for_execution=false` 不等于授权运行。

未来 receipt 核验注意：cp0 是 goal bind 后观测，native 距离及 success 只能标记为从原公式计算，不能称该时刻 public step 返回；cp5/10/15 才能连接相同步返回的原 reward/success。原七维 native 距离含相对非零目标 agent velocity，success 不含速度；不修改原 gate、不按结果筛槽。输出 wrapper 的 `receipt` 必须由 MAIN 审核并独占保存才能输入原 P1 合同，wrapper 的外部 receipt hash 不是自签新实验权威。
