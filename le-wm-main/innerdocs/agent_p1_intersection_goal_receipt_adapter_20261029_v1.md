# Intersection physical v2 → P1 saved receipt 归一化（待真实输入）

该独占 adapter 根据 MAIN 已认证的 intersection physical v2 `.consume` 字节，将固定六槽、每槽 cp0/5/10/15 映射到原 P1 goal-semantics receipt ABI。继承完整 strict receipt，不重新运行上游 full strict；逐项核对实际外部 receipt/report/prepare/source SHA、原六计划和 task/selection/root/goal、source/prepare/report 在 receipt 中的 binding、全槽顺序、实际 checkpoint/snapshot/response 对应和明确 missing。非零 goal7 保留，原 reward 和 native success 不改；不选根、不补早停、不签 P1/P2 或 fresh。

cp0 是原任务 goal 绑定后的 root 观测而非该时间的 public-step 返回：本 adapter 用原公式重算并在 `details` 中标为 `derived_from_bound_cp0_not_native_return`。cp5/10/15 使用同一步实际 response 的 native success 与 `-reward`。如 checkpoint 已保存但该步 response 因工程失败未闭合，则输出 null 和原因，不造 native label；其他缺项亦保留 null 与原 status/error。goal RGB 当前固定 null，spec `rgb_pairs=[]`，不输出速度视觉碰撞声明。

输出为 `{schema,receipt,details,bindings,denominator,evidence,P1_admission}`，内部 `receipt` 是原 P1 合同的输入；为进入该合同，应由 MAIN 审核后将该字段独占保存，并以实际 SHA 传给原工具。原 strict receipt 和 spec 来源完整时序认证由 MAIN 负责，本工具不能由哈希推断前瞻性；源码 ABI 目前依据已落盘但未确认冻结的 v2 源码，真实 receipt 还没有，fixture 不是物理结果。physical v1 不接受。

等待 MAIN 实际 v2 source/prepare/spec/physical report/strict receipt SHA；当前不生成正式 spec、receipt 或调用本 CLI。所有路径绝对，输出文件须不存在：

```bash
.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_agent_p1_intersection_goal_receipt_adapter_20261029_v1.py
.venv/bin/python -m experiments.spectral_reachability.agent_p1_intersection_goal_receipt_adapter_20261029_v1 normalize --spec /ABS/SPEC.json --spec-sha SPEC_SHA --prepared /ABS/V2_PREPARE.json --prepare-sha PREPARE_SHA --source-sha SOURCE_SHA --receipt /ABS/V2_CONSUME.json --receipt-sha RECEIPT_SHA --report /ABS/report.json --report-sha REPORT_SHA --output /ABS/agent_p1_intersection_goal_receipt_adapter_20261029_v1_MAIN01.json
.venv/bin/python -m experiments.spectral_reachability.agent_p1_intersection_goal_receipt_adapter_20261029_v1 verify --spec /ABS/SPEC.json --spec-sha SPEC_SHA --prepared /ABS/V2_PREPARE.json --prepare-sha PREPARE_SHA --source-sha SOURCE_SHA --receipt /ABS/V2_CONSUME.json --receipt-sha RECEIPT_SHA --report /ABS/report.json --report-sha REPORT_SHA --output /ABS/agent_p1_intersection_goal_receipt_adapter_20261029_v1_MAIN01.json
```
