# Common-resource 六槽 raw15 native 物理适配器

本三件套仅实现执行准备、原生执行器和严格 saved consumer。worker 只运行 RAM fixtures，不运行神经、物理或生产 prepare；MAIN 亲审后运行。旧 wallcapped physical、GRASP、runtime、原结果一概不改。两个已知 off 根，不打开 reserved fresh 根，不训练或放宽任何科学门。

## 输入 ABI 与真实 authority

唯一 producer 是 `experiments/spectral_reachability/agent_p2_common_resource_grasp_cuda_20261029_v1.py`。调用其真实 `consume(planner_output, planner_report_sha)` **完整重建**六槽，将整个返回值与 CLI 指定 SHA 的持久 receipt 比较；再调用其 `verify(report.prepared, report.prepare_sha256)`。source SHA 在 import/consume 前核验。不得传入旧 standalone GRASP750、wallcapped 四槽或并栏 summary，不伪造旧 receipt/schema，不等待或根据物理 outcome 挑计划。

固定顺序：root0/off joint、composition、grasp；root1/off grasp、composition、joint。每个 `arm{i}.json.gz` 的原始 `result.export` 再按 report artifact SHA 读取，保留 full30 logits 与三段 replay contexts，用于直接验证 joint30 L2 半径 `.75*sqrt(15)`、float64 tanh→float32 chart、原 raw15 位值、原 normalizer 和3包 action-history 接线。禁止 atanh 反推/重新投影/补动作。三种方法共用原 task SHA/ID、full3 latent+action2、goal192/Q 和模型 hash；producer compact-newline digest 与 own physical spaced-JSON digest 分开认证。

runtime receipt 没有统一的 root/arm account schema：joint/composition 账户有 root/arm/export_missing；grasp 是实际 GAccount 原字段，不能伪装成旧谱账户。全6 accounts/resources 原样继承，attempted/completed 分轴、不退款、不把 API cap1902说成 equal FLOPs。实际模型运行/report/receipt SHA 尚不在本实现内预填，均由 MAIN CLI 提供。`physical_release=false` 不是自动准入；MAIN显式工程 AUTH 与完整严格链同时必需。

## 原生执行与 strict

新 own-schema prepare/report/consume，不调用旧 execute、不继承旧物理 row、不修改 module globals。复用冻结 wallcapped raw15/observe/audit_labels，以及 mixedmode 的 placement、goal bind、native hooks、dispatch、metrics 和 Store 字节/链编解码。不可直接复用其固定4槽 orchestrator/validator，故在新文件中显式扩展为六槽；无动态代码生成/exec。

每个可用计划全新 env：原两 reset+placement、原 warmup10、精确 source RGB历史/原生7维 root身份，一次绑定原 goal，再原始15个 raw float32；cp0/5/10/15。每次 capture 真正 off/on/off triplet，不以 imagined context 恢复物理。native terminated/truncated 立即停、不 padding；未达 checkpoint 明确 missing/null，native reward 是含 velocity 的原7维距离，success仍原 jointXY+angle 条件，wrapped angle仅诊断。保留每一步全7维 state/snapshot/reward、velocity/angle诊断、requested/actual bits、task/goal/render来源。不删缺失 export：该槽保留且不 constructor、不换计划。工程失败也保留六槽分母。

满程上限：6 constructor/close、12 reset、150 public_step（60warm+90plan）、1524 physics_step、24 setter、18 goal_setter、12 goal_render、6 goal_bind、486 explicit_render/PNG。attempt/return/failed 全记账；hooks仅factory返回后安装，不声称截获constructor内部。未知 journal kind/操作资源拒绝，独立全量census、嵌套intent-return、branch顺序/hashchain、PNG原字节与pixel/state guard、原source warm trace、goal绑定时序、全量reward/native7/angle和提前停逐项复核。全部artifact census及来源文件前后hash，未知文件拒绝。

新输出128MiB、final reserve16MiB、floor200MiB、整段900秒（含prepare verify及final hash），prepare/receipt metadata上限128MiB，排他新路径；失败产物保留。神经调用0；物理账只由MAIN实际run产生。完整6臂执行也不意味着P1/P2机制成立、freshness或独立样本认证。

## MAIN CLI（inner 根；所有变量均为实际值）

```sh
.venv/bin/python -B -m pytest -q tests/test_agent_p2_common_resource_physical_20261029_v1.py

.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_common_resource_physical_20261029_v1 prepare \
  --planner-output "$P2_COMMON_OUTPUT" --planner-report-sha "$P2_COMMON_REPORT_SHA" \
  --planner-receipt "$P2_COMMON_RECEIPT" --planner-receipt-sha "$P2_COMMON_RECEIPT_SHA" \
  --planner-source-sha "$P2_COMMON_SOURCE_SHA" \
  --output "$P2_COMMON_PHYSICAL_PREPARE"

.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_common_resource_physical_20261029_v1 verify \
  --prepared "$P2_COMMON_PHYSICAL_PREPARE" --prepare-sha "$P2_COMMON_PHYSICAL_PREPARE_SHA"

.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_common_resource_physical_20261029_v1 run \
  --prepared "$P2_COMMON_PHYSICAL_PREPARE" --prepare-sha "$P2_COMMON_PHYSICAL_PREPARE_SHA" \
  --output "$P2_COMMON_PHYSICAL_OUTPUT" --authorization MAIN_COMMON_RESOURCE6_FULL15_PHYSICS_ONLY

.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_common_resource_physical_20261029_v1 consume \
  --output "$P2_COMMON_PHYSICAL_OUTPUT" --report-sha "$P2_COMMON_PHYSICAL_REPORT_SHA" \
  --receipt "$P2_COMMON_PHYSICAL_RECEIPT"
```

planner-output/receipt/prepared须绝对路径，新metadata与输出basename须以本stem加下划线开头；不存在即创建，存在拒绝覆盖。worker无生产prepare/run/consume记录。待MAIN取得真实common-resource完整report/strict/sourceSHA后才能生产prepare；不填未来SHA、不自签scope或freshness。
