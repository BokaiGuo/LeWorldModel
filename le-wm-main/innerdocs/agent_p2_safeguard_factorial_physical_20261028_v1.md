# Zero-anchor factorial physical adapter v1

仅8条 zero-anchor unguarded full10 的新物理干预；known2 roots × off/on × 原两arm，全部保留。原root/warm10/prefix5/taskgoal/condition不变。guarded branch4 exactbits复用原incumbent control，其余7条复用本次对应zero unguarded，绝不读取outcome选择动作。本版本不执行terminal编码、不训练、不作科学准入。

## Authority与真实接线

固定原factorial prepare SHA `9ad5980d7943f11dc5e6d53c285f37dc4a86e69620d136036bcdd656d3e9d469`，MAIN实际runtime strict SHA `8a59a189feaedf3b215b7aa9245c3b027228e2a51cfef32a47df286c664e5d89`。读取实际报告/prepare/32cells，重算literal material ABI、32cell seal/context/goal/prefix/full10 bits/原Q；全部原bindings、runtime产物与自身三件套SHA闭合。复用MAIN已接受strict，不再次调用旧consume_compile；`full_upstream_rebuilt_here=False`。不伪造旧compiler receipt。

直接调用未修改的 `physical.execute`、`physical.assemble`、native kernel；Store继承v2 ReplayGateStore.emit，step5原condition PNG bytes/pixels/state/snapshot完全匹配后才允许suffix入口。新plan保留parent_Q/witness/report identity但移除无前缀的旧predicted_Q/witness，新suffix_Q/context/witness来自实际32forward封存cell。planner_report_sha256改绑真实factorial runtime报告。

内部native_prepare保留旧native schema/contract，这是未修改executor的兼容ABI。其中历史compiler480/1328等labels不是本次计算声称；外层明确compatibility_ABI_only、已付factorial32模型成本与本次神经调用0分栏。全部物理计量由原native Meter记录，预声明8 constructor/close、16reset、160public、1632physics、528explicit render等caps；早停/失败不补样本。

900s从run进入verify前起计，不能中断单次native调用；64MiB输出、2MiB final reserve、200MiB disk floor。超时仍允许原native journal ABI写run_finished失败终态，不假complete。metadata文件with内flush/fsync，关闭后fsync目录。constructor显式with_target=False/render_action=False；原native流程只切overlay，taskgoal不改。

consume完整原assemble、journal hashchain、全部PNG/census、exactmid先后次序及闭包末验。稳定输出 `physical_report_sha256`、`plans`（8完整新物理plan含commands）、`rows`（8原assemble strict rows）、guarded_reuse、实际counts。partial状态保留全部8，不授physical/science release。MAIN独立32cell分析无需依赖其源码。

## MAIN执行命令

在仓库根目录，先审读/单测；本worker不运行prepare或物理。

```bash
.venv/bin/python -m pytest -q tests/test_agent_p2_safeguard_factorial_physical_20261028_v1.py
.venv/bin/python -m experiments.spectral_reachability.agent_p2_safeguard_factorial_physical_20261028_v1 prepare --runtime-output results/agent_p2_safeguard_factorial_cuda_20261028_v1_cuda_MAIN01 --runtime-receipt results/agent_p2_safeguard_factorial_cuda_20261028_v1_consume_MAIN01.json --output results/agent_p2_safeguard_factorial_physical_20261028_v1_prepare_MAIN01.json
.venv/bin/python -m experiments.spectral_reachability.agent_p2_safeguard_factorial_physical_20261028_v1 verify --prepared results/agent_p2_safeguard_factorial_physical_20261028_v1_prepare_MAIN01.json --prepare-sha ACTUAL_PREPARE_SHA
.venv/bin/python -m experiments.spectral_reachability.agent_p2_safeguard_factorial_physical_20261028_v1 run --prepared results/agent_p2_safeguard_factorial_physical_20261028_v1_prepare_MAIN01.json --prepare-sha ACTUAL_PREPARE_SHA --output results/agent_p2_safeguard_factorial_physical_20261028_v1_MAIN01 --authorize-physics
.venv/bin/python -m experiments.spectral_reachability.agent_p2_safeguard_factorial_physical_20261028_v1 consume --output results/agent_p2_safeguard_factorial_physical_20261028_v1_MAIN01 --physical-report-sha ACTUAL_REPORT_SHA --receipt results/agent_p2_safeguard_factorial_physical_20261028_v1_consume_MAIN01.json
```

所有新metadata/output固定NAME下划线前缀，不覆盖既存文件。无未来SHA预填。测试仅RAM native接口和已保存literal算术，不能认证真实physics、freshness或P1/P2；生产prepare/run/consume由MAIN执行。成功包括原agent+object合同，与object-only coverage分栏；warmup旧goal reward不当当前taskgoal收益。
