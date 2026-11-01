# P4 sourcepack / freshness / history certificate audit

独占 saved-only 审计与 consume：`agent_p4_fresh_certificate_audit_20261028_v1`。不修改既有 P4 source/consumer、freshness/history metadata、MAIN结果或共享文档；不打开四个 reserved fresh roots，不执行 producer、模型、环境、物理、网络或采集。

## 实际证据结论

当前合法 fresh/history certificate：**不存在，可消费状态 blocked**。

已存在的两个 `s0_construction_certificate.npz` / `s1_construction_certificate.npz` 是真实可解析的 construction artifacts，分别 SHA256：

```text
96c3d5b644696f17852d42b900a05e3e9ab183060c89a36db883988caa4dbf9d
6dcaf39084cfbe50783de81a95454afd78095a9b12348c1af7654afbbf3b1050
```

它们含 goal、goal_rgb、witness commands/logits、future/observed states、root history、raw history；本工具只读取NPZ成员及shape/dtype，不以数组内容签发freshness。真实 sourcepack SHA `c2d8d573f94884ffc081163f25f2bfa269f98ffe8155887704109219910ffec6` 明确：`global_freshness=reused_recipe_calibration_not_fresh_not_same_task`、`new_independent_confirmation=false`、`fixed_denominator=2`、`reachable_count=2`。故只能标为 `construction_only_not_fresh_or_history_certificate`。

四槽 root manifest SHA `3627dd2561a496ea2a0831c2aafa11337363ac1e9ca01285907c9f20e3a7b838` 与 consumer manifest SHA `2b80412eb2efc7b721d6edb754ce5d9d5d513e17744706aa1e5bab56c785f7e7` 均显示 `freshness_certified=false`、`source_run_authorized=false`、`history_certificate=null`，consumer 的 `certified_fresh_roots=[]`。coverage SHA `00a3fffeaaa8943e851d129b941a7f07feaa1db387117beb98110e4ec95f5eeb` 显示 `coverage_complete=false`、`original_execution_bounds_certified=0`。exclusion manifest SHA `a17bc732c9a021a4ca81a7e4da573c752ad1f66eafe08b1a3dad2b64464d7060` 显示 `approval.approved=false`、`source_run_preconditions_met=false`、`freshness_certified=false`。

因此不能由两个 construction cert、当前 sourcepack、seed不冲突、文件存在或“没有看到更多证书”推导历史排重完成，更不能把未知写成空历史或批准。四 reserved root 不曾由本工具打开。

## 阻断与接口

固定阻断：root/consumer 无认证根；history certificate null；历史 coverage 与原执行边界未认证；exclusion approval false；sourcepack 明确非fresh/非同任务；new independent confirmation false；reserved-root identity/access/disclosure 仍 unknown。`approval_created=false`、`certificate_created=false`、`certified_fresh_roots=[]`恒定。

```bash
.venv/bin/python -B -m experiments.spectral_reachability.agent_p4_fresh_certificate_audit_20261028_v1 audit --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p4_fresh_certificate_audit_20261028_v1_report_MAIN01.json
AUDIT_SHA=$(sha256sum /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p4_fresh_certificate_audit_20261028_v1_report_MAIN01.json | cut -d ' ' -f1)
.venv/bin/python -B -m experiments.spectral_reachability.agent_p4_fresh_certificate_audit_20261028_v1 consume --report /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p4_fresh_certificate_audit_20261028_v1_report_MAIN01.json --report-sha "$AUDIT_SHA"
```

测试：

```bash
.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p4_fresh_certificate_audit_20261028_v1.py
```

`audit` 只会返回 blocked report；`consume` 只接受逐字节重建的同一 blocked report，不会生成 certificate/approval。测试通过不是freshness科学证据。要解除阻断，必须由原运行方或 MAIN 提供可验证的历史 reset/load/access/disclosure 覆盖、明确 cutoff/scope、四槽实际 identity 与批内/历史排重证据；新执行或本工具自身不能替代这些历史事实。
