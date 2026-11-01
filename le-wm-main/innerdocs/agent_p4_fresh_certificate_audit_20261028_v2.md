# P4 fresh certificate audit v2

独占 v2 saved-only 审计/consume。v1 保持冻结；本版本修复 v1 的内部 pin typo：closure 的真实 SHA 为 `7c7e1792c5a17986aaf3a38539a280ec46d5910eb4d9351babbe9bf2a70b3576`。所有 v2 pins 均由当前实际文件 SHA 取得并在 `audit()` 开始核验。若 pin、schema或状态被篡改，审计 fail-closed；正常输入应返回 blocked，而不是因为自身哈希不闭合而失败。

## 真实结论

当前没有可合法消费的 freshness/history certificate。两个真实 construction NPZ 仍保留并逐成员核验：

```text
s0 96c3d5b644696f17852d42b900a05e3e9ab183060c89a36db883988caa4dbf9d
s1 6dcaf39084cfbe50783de81a95454afd78095a9b12348c1af7654afbbf3b1050
```

它们只被记录为 `construction_only_not_fresh_or_history_certificate`。sourcepack `c2d8d573f94884ffc081163f25f2bfa269f98ffe8155887704109219910ffec6` 明确 `global_freshness=reused_recipe_calibration_not_fresh_not_same_task`、`new_independent_confirmation=false`、分母2/可达2。

root manifest `3627dd2561a496ea2a0831c2aafa11337363ac1e9ca01285907c9f20e3a7b838`、consumer manifest `2b80412eb2efc7b721d6edb754ce5d9d5d513e17744706aa1e5bab56c785f7e7` 均无 freshness/history 认证；coverage `00a3fffeaaa8943e851d129b941a7f07feaa1db387117beb98110e4ec95f5eeb` 未完成且原执行边界为0；exclusion manifest `a17bc732c9a021a4ca81a7e4da573c752ad1f66eafe08b1a3dad2b64464d7060` 未获批准、source preconditions未满足。四 reserved roots 未打开。

## 接口与阻断

```bash
.venv/bin/python -B -m experiments.spectral_reachability.agent_p4_fresh_certificate_audit_20261028_v2 audit --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p4_fresh_certificate_audit_20261028_v2_report_MAIN01.json
AUDIT_SHA=$(sha256sum /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p4_fresh_certificate_audit_20261028_v2_report_MAIN01.json | cut -d ' ' -f1)
.venv/bin/python -B -m experiments.spectral_reachability.agent_p4_fresh_certificate_audit_20261028_v2 consume --report /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p4_fresh_certificate_audit_20261028_v2_report_MAIN01.json --report-sha "$AUDIT_SHA"
```

正常终态固定为 `blocked_no_legal_fresh_certificate`，`certified_fresh_roots=[]`、`history_certificate=null`、`approval_created=false`、`certificate_created=false`。consume 逐字节重建同一 blocked report，篡改会拒绝。没有通过“不存在证据”推导不存在历史；要解除阻断仍需原运行方/MAIN提供历史 reset/load/access/disclosure覆盖、cutoff/scope、实际新根 identity与历史/批内排重证据。新执行不能补历史证书。

测试：

```bash
.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p4_fresh_certificate_audit_20261028_v2.py
```

只读saved metadata与NPZ头；0神经、0物理、0网络、0 reserved-root访问。测试和blocked报告都不是freshness科学通过。

本轮实际终态：9 tests passed；saved audit report已生成，SHA `15e3ece87e36eba469a0cc2188efbb7edaa94fa00f1562f9e6adc2052e4e4ae0`；对该报告执行consume成功返回同一 `blocked_no_legal_fresh_certificate`。v1未改。v2冻结后任何修复只能新stem。
