# P3 commit-H3 vs MPC：独立 saved-only native metrics v1

仅本三件套，保留所有旧封存。使用 experiment-code 技能组织独立实现/保存回归；不采用训练/NN/物理/图表通用模板，不调用producer或任何full strict，不按结果挑根、bank、时点、指标。原解释文档SHA固定为 `13a1be54bf3e87885563bade7302c60a1745ce4715460abad47f949443fe3ba6`，analyze核实际字节；不倒签原P3预注册，不签P1/P3/C3/freshness。

## 固定比较与精确控制

固定四槽 `(root0,endpoints),(root0,midpoints),(root1,endpoints),(root1,midpoints)`，全off/known2；cp0/5/10/15/20，每类对照20检查点槽。主对照commit−原adaptive MPC，辅助commit−同root/bank fixedH1，分别4对，不选表现较好的一组升级主分析。

独立NumPy float64从原goal/state7计算joint/agent/object XY、abs angle、wrapped angle、原native angle `min(abs(delta_angle),2π-abs(delta_angle))`（可能负）、velocity L2、全native7距离/reward。success仍为jointXY<20且原native angle<π/9；不删velocity、不用wrapped angle重写native。距离/Q不存在合成加权分数；本工具**没有latent/Q**，也不伪造未编码cp20。

逐cp输出两侧metrics、commit−reference及精确正/负/零符号，success保留两侧bool/null。每条保留event-time terminal和at20，早停不复制终态至缺失检查点，不将missing置0，不做CI。两根不是四独立根。

真实计划必须为原adaptive b5封存whole winner，原完整request也需一致；raw20逐位等于原prefix5+原winner raw15。与原MPC前10动作float32 signed bits必须一致。report的actual operation receipts逐attempt核raw_bits；从SHA绑定journal抽取每条plan `returned_transition`，核actual/requested位串/dtype、goal、native label、每条response、停止后不补动作。这里不调用全journal/native/PNG认证，继承外部full receipt；只独立展开本对照所需字段。

主对照cp0/5/10分别独立核state float64 bytes、完整snapshot canonical JSON（含负零/位串）和pixel SHA；与fixedH1只要求cp0/5控制，cp10后是不同程序，不强求相同。控制不匹配仍输出指标，但 `effect_interpretable=false`、`control_failure_not_scientific_negative=true`，明确工程身份问题，不用于机制负结论；缺控制为null、不冒充通过。没有松动物理runner cp5 pre-dispatch guard或外部full strict。完整同动作物理路径标记还需实际20返回；`latent_fidelity_available=false`始终不伪造编码。

## 真实 ABI 与认证边界

未来输入只用显式实际SHA，不能预填。physical schema `agent_p3_h3_witness_physical_20261029_v1.{prepare,report,consume}`：

- report：`execution_binding.prepared/prepare_sha256/planner_authority`、rows4、actual_operation_receipts、journal_sha256、counts/errors/status。
- receipt：`physical_report_sha256`、完整rows/plans、planner_authority、bindings、counts/errors/status。没有杜撰`complete=true`；它可以真实认证partial，consumer逐槽保留status/errors，而无法对齐完整动作/labels时fail closed，不伪称成功。
- plan：root_slot/arm/condition、task_id/selection_digest、commands20、prefix_commands5、suffix_commands15、native_goal7、source_witness.winner/request。
- rows：own_returned、checkpoints/missing_checkpoints、responses、final_state、terminated/truncated、first_error/secondary_errors；原生早停仍保留固定分母。snapshot及像素的真实性由指定full receipt提供，本工具核跨运行精确一致性，不重新读取/认证PNG或运行环境。
- 原shared schema `agent_p3_shared_panel_20261029_v1.{report,consume}`；原report/strict SHA固定为 `668db35d79eb650419e4efa42dfc21a15cdf91725700b2f202e2a2ec975bcd6a` / `d23ffa48bc954306bc27a15f924da8daafffc9b4bc76a91e59aa949ab1abbafd`。原strict必须complete，未来physical strict要绑定这组原planner authority，且全rows/plans及counts/errors/status精确匹配自己的report/prepare。

不重复跑full strict；不会下载模型、加载torch/producer、读取未来outcome定规则。native resource_counts全量保留，不凭predict数或缺新模型调用签equalFLOPs。输出保留external authority/原bindings并绑定本三件套；不重新遍历fullreceipt全部文件。新旧source/模型闭包属于已有strict。本工具仅读显式report/receipt、report指定prepare和journal、原report/strict及本文/冻结解释文档，无全目录漫游。

## CLI

在 `/home/gbk/code/LeWorldModel/le-wm-main`，路径用canonical absolute。原shared的两个默认路径是MAIN01，SHA不可换成另批结果；允许显式路径承载同字节。

```bash
.venv/bin/python -m pytest -q tests/test_agent_p3_commit_vs_mpc_metrics_20261029_v1.py
.venv/bin/python -m experiments.spectral_reachability.agent_p3_commit_vs_mpc_metrics_20261029_v1 analyze \
  --report /ABS/H3_output/report.json --report-sha ACTUAL_H3_REPORT_SHA \
  --receipt /ABS/H3_full_consume.json --receipt-sha ACTUAL_H3_RECEIPT_SHA \
  --output /ABS/agent_p3_commit_vs_mpc_metrics_20261029_v1_MAIN01.json
.venv/bin/python -m experiments.spectral_reachability.agent_p3_commit_vs_mpc_metrics_20261029_v1 verify \
  --summary /ABS/agent_p3_commit_vs_mpc_metrics_20261029_v1_MAIN01.json \
  --summary-sha ACTUAL_METRICS_SHA
```

输入JSON/journal各≤256MiB，输出排他创建本stem，禁止覆盖；重复key/NaN、错误SHA/动作/goal/label/固定顺序/伪checkpoint拒绝。verify确定性重建本saved计算，不调用producer/fullstrict。

## 交付状态

首轮 **28 passed /0.48s**；补齐warmup零计划观察与native成功早停后最终 **30 passed /0.50s**。测试为纯RAM/临时JSON落盘，覆盖四槽×五cp×两类配对、独立native7/velocity/原angle公式、raw负零、完整winner/request、actual与requested bits/attempt bits/goal/reward/termination/response/checkpoint篡改、cp5/10 state/snapshot/pixel控制、辅助fixedH1不强制cp10相同、14步截断与成功、warmup零计划观察、完整analyze→verify/覆盖拒绝/journal SHA变更拒绝。fixture有意使用独立deepcopy避免producer/consumer期待值共享别名；临时文件测试只替换RAM authority hashes，不读取未来生产文件，不修改任何真实封存。

已仅核对原shared报告固定八臂真实检查点布局均为0/5/10/15/20，没有运行真实analyze/verify/full strict。未来physical report/fullreceipt实际SHA由MAIN给出并生产消费；尚未完成项只有MAIN审读/独立测试及最终实际运行，不预填futureSHA。worker不生产、不启动环境，fixture不是科学证据。无NN/physics/fullstrict，无旧文件变更。
