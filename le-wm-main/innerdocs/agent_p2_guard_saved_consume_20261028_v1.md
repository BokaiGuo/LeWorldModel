# P2 guard terminal saved consumer

仅新增本stem源码、tests、本文，不修改已冻结guard/next/native kernel、结果、共享文档，
不派生agent。不执行GPU/CPU神经/physics，不训练/拟合/下载/筛根/改门槛。
使用experiment-code技能固定输入、对账与失败回归；技能中训练/按结果改善等不适用本授权。

## 真实输入与复用边界

MAIN实际HOST CUDA `agent_p2_runtime_guard_20261028_v1_known_MAIN01` 已terminal exit0。
本worker只读其真实输出，不重跑。实际report SHA
`0cfbe37a7783c6df331305e95e3c058a9b006f72bb1ca5b59d24fae047ee23a0`，
guard prepare SHA `02269ae5d08d1f645e9bcf2473a7c7cbb2a16f3125c3fad62c23859380cc6c48`，
native prepared canonical SHA `1d79f5b17002334434454dfb5a94e6d918de1d603afdf84918e5ded6eb980d1b`，
MAIN review SHA `52599f94ff9e60db7673310521a1536b9d24fe033bcca8be6206e814946918c8`。
这些是已落地字节/已核身份，不是预造未来SHA。CLI仍要求外部显式传入，不自动信任报告自述。

原native.selected_actions整包入口要求其自己的completion.json、cuda_completed报告、
独立finalization文件，不能直接读取wrapper不同格式；不伪造这些文件/不改旧loader。
复用原 `validate_actual`、`validate_operations`、`validate_result`、
`validate_array_artifacts`、spectral_bank、candidate_bank、suffix_solution、原Q/witness重算。
后四者仅对保存NumPy数组做原规划器数值验证，不训练或拟合Q/normalizer，不构造模型。
原guard.verify/native.load_prepared重验全部source/input seals；guard源SHA固定
`616ec94445681ede449e525a45016b4e7d849ec488181674f570e25f5ca721d3`。

## 不仅验证日志hash

每根先严格检查row/file SHA与header中root/ready/native prepare/guard source/resource contract。
日志链必须完整、seq连续、前后SHA正确，唯一header/root_terminal、时间单调。
再由consumer自己的事件解释器重建，不调用guard.DispatchGuard重放自己的receipt：

- 固定123 paid groups：shared base、joint80 candidate/replay、conditional40 candidate/replay。
  shared只计paired unique一次，两条standalone各全额归属；所有primitive在组内分配不重复收费。
- 867 reserve→entry→return：664真实模型API、83AD/SVD/solve、120metric；entry必须紧跟reserve，
  return必须有原entry，不能缺失、重复、失败、pending，不能凭intent推算执行。
- 每个forward request/return逐项精确等于actual_model_entries，含scope/index/module/
  API samples/context tokens；每个AD/solve请求/返回精确等于原operations流。
- 每个candidate_terminal hash与该真实候选完整原event（含suffix/J/Q/history/witness/actions）
  闭合；120候选和120metric不丢失败槽、不只导出赢家。原validator验证每个Q/有界solve、
  稳定最小选择、原raw动作、完整上下文、两条winner replay，不另做神经重放。
- 82个原full nominal/J数组逐文件SHA/shape/value/root/goal/Q/prefix/context严格核验，
  array_saved事件与原receipt一一对应，保存边界必须等于原expected_array_progress，不能晚存。
- 独立累加intent/subreserved/returned各轴并重建完整guard snapshot，逐轴核原AD operation counts；
  root各332predict/332AE、120metric，cohort1328/1328、480candidate/metric。
- 原report8行census必须和已核4根guard全部480状态逐项一致。原参数/buffers before/after两组
  SHA必须存在、格式合法、完全相等，model/source flags须true；原权重/config及source closure
  由prepare再次复核。整个输出文件布局固定，不容纳额外/重复/foreign/partial文件或symlink。

模型before/after是封存的执行方tensor hash声明；saved consumer不加载模型，不能独立计算
CUDA时刻tensor哈希。guard/native源约束其产生路径，不将hash一致性当独立模型重放证明。
host return也不是独立kernel profiler；同步成功由已绑定guard源码的root终结路径保证。
未知backward-engine/FLOPs/历史获取保持null，不把数学VJP换成backward API或FLOPs。
图像/physics/endpoint本轮未执行，不凭latent/Q称任务收益。仅四个known roots，不是fresh确认。

## fail closed与接口

`consume(output, report_sha, prepared_path, prepare_sha, review_sha, confirmatory=False)`：
必须先有外部SHA匹配且status=known_root_completed、error=null、secondary=[]的terminal
report，才读取任何row/prepare/部分产物。没有report或未终结决不根据已完成根推断整批完整。
pending/failure/missing任何证据都抛异常，不导出部分plan。CLI失败返回1，保持4/8/480固定
分母、selected_plans空；仅在外部report SHA匹配时保留原declared_census（含失败/pending），
不填补缺失状态、不重试/补根/删除。失败诊断不等于实测完成量。

成功返回8个已核selected_plans用于审读（slot-major/原arm顺序），没有physics调用或放行；
P1/P2与physical_execution_authorized始终false，confirmatory读取前拒绝。
输出独占本stem、fsync，全部输入再次SHA检查，绑定consumer source/tests/doc SHA。
入口JSON解析及其SHA取同一份已读取字节，避免两次读取分别绑定不同版本。
保存验证的SVD/solve/Q/hash/JSON费用是额外CPU validation，不抵扣原运行费用。

在inner repo实际saved消费命令（不触发GPU/physics）：

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p2_guard_saved_consume_20261028_v1 --input results/agent_p2_runtime_guard_20261028_v1_known_MAIN01 --report-sha 0cfbe37a7783c6df331305e95e3c058a9b006f72bb1ca5b59d24fae047ee23a0 --prepared results/agent_p2_runtime_guard_20261028_v1_prepared01.json --prepare-sha 02269ae5d08d1f645e9bcf2473a7c7cbb2a16f3125c3fad62c23859380cc6c48 --review-sha 52599f94ff9e60db7673310521a1536b9d24fe033bcca8be6206e814946918c8 --output results/agent_p2_guard_saved_consume_20261028_v1_MAIN02.json
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/agent_p2_guard_saved_consume_20261028_v1_tests.py
```

真实saved测试+RAM变异覆盖语义对账（刻意绕过hash链仍须拒绝）、pending/缺return/错实际入口/
错操作/错candidate hash/错数组保存边界/unknown→0/部分报告/模型hash/固定分母等。
直接变异原full数组的value/shape/context/receipt/stage和原候选raw动作/history/Q/suffix J/
checkpoint provenance，确认复用的native validators实际拒绝，而不是只测试其被调用。
测试与saved重算只是工程一致性，不是科学通过。MAIN仍独立审源码及真实消费；不释放物理，
不解锁P2。独立P1/freshness、原任务物理收益/精确恢复和完整费用证据仍未满足。

## 本轮授权补齐：packet ledger与实际返回接缝

用户明确授权继续完成本consumer；只修改本stem三文件，不动producer或原CUDA输出。
旧saved01保留其原consumer SHA，不将旧报告冒称新源码验证。MAIN复跑使用上面新输出名，
写入为exclusive create，已有文件会拒绝覆盖。

新增 `reconcile_packet_ledgers`：两臂分别164/84条packet，逐条连接planning/replay实际
AE→predict成对返回（328/168个module returns），scope/phase/path/order/samples/context
逐项闭合。compilation/AD另由原独立journal解释器计费，packet仅交叉核验，不重复加账。
严格packet/result字段集合，拒绝私加flops或full_resource_equal声明；字面整数计量拒绝bool，
host wall time必须有限且非负，但不将其解释为同步device计时。候选缺失/失败/pending不可
导出完整成功；失败CLI仍保留固定4/8/480分母及可绑定原census。

每根输出 `arm_resource_comparison`：用已经独立对账的standalone资源向量逐轴比较，
预测前沿相等而AD/SVD等存在差异，未知flops等仍null；明确full_resource_equal=false、
hardware_cost_equivalence_certified=false。这里没有测得FLOPs，也不能证明硬件等成本。
新增RAM变异覆盖packet缺失/重复/错实际scope、夹带假费用/等资源声明、bool计数、负计时、
漏候选及失败候选；已有全部原数组/Q/solve/witness重算与ledger变异测试继续执行。
实际saved数值重算不是独立神经复现，更不是任务物理成功证据。

本轮实际联合回归（consumer 55 + 冻结guard 20）75 passed in 60.05s：

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/agent_p2_guard_saved_consume_20261028_v1_tests.py tests/agent_p2_runtime_guard_20261028_v1_tests.py
```

另对上述真实report以consume API只读重算并打印摘要，不写新结果报告，保留旧saved01。
本轮未运行CUDA/physics/CPU神经；未独立重算CUDA时刻模型tensor hash或实际硬件FLOPs。
