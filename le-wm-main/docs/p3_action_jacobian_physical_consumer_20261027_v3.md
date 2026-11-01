# Action-Jacobian fixed-H3 physical consumer v3

独立v3入口，只接受实际v3 CUDA report、v3 scheduler prepare和v3协议；旧sealed
源不改，v2 report/旧物理终点不能顶替。本轮只tests与prepare，不运行physics/CUDA。
使用experiment-code分离输入/执行收据；使用TDD先验证缺模块失败，再覆盖v3合同。

## 唯一计划来源

`results/p3_action_jacobian_budget_schedule_20261027_v3_cuda_main_v1/report.json`
SHA256 `c13dab20228fd590c3f9375e29b6be905a8db083942a547c862d97c90cf53921`。
对应scheduler prepare SHA256：
`27988bdb563563bd2cb0ea5ef3f69851627a2cae4d648e1e545f2e68c7d5c134`。

校验v3 schema、全部4根×uniform/dynamic八臂、冻结model/config/goal/Q/context、
packet SHA、每arm JSON、events journal、所有inputs与source seals。离线callback
仅返回保存的edge contexts，重新执行v3数组调度逻辑：8次SVD重算root-H1 action
Jacobian chi、原公式和allocation、144条边顺序/重复、H3-only winner及model cost。
这8次数组SVD单独记录prepare_numpy_svd_calls；不发生神经forward或fresh Jacobian。

chi保持 `r=Q(goal−nominal)`、`K=QJ` 的列空间残差比；原10维零action-logit
root-H1 Jacobian继承源由v3 scheduler封存。不能换成v2有限位移chi。仍原完整16bank、
H3、Q、goal、K2=2+floor(4chi+.5), K3=8−K2；consumer不重选动作或改公式。

全部8plans的三个整数indices独立重建15×2 rawbank commands。动作相同也保持两个
槽位、重新reset、独立执行和记录，不合并tie、不复用任何旧/同臂物理结果。
model阶段上游1152planning+24verification=1176 predict samples，只在cohort计一次。
每臂147 predict/action-encoder和441context-token收据逐坐标检查；新增predict0、
新Jacobian0。rootJac获取成本始终inherited_unknown_cost：predict/encoder/backward/
VJP/walltime为null而非0，四份共享继承不乘以8；不能称全pipeline等成本或FLOPs匹配。

## 新执行与安全合同

每臂一次，总planned120steps。重新生成同四calibration roots，reset/warmup，核验
whole-stack history RGB、goal RGB及同根两臂初始context一致，才dispatch当前v3路径。
root source只使用旧来源的identity/history/goal，不使用旧10/15步终点。reset/warmup
开销不在selected-path计数内，也不宣称零成本。

复用不可变durable底层physical_branch/guarded_execute；不调用v2 consumer的prepare
或execute入口。每rawstep fresh RoyalBlue presence guard，attempt/return journal
fsync，每张RGB保存/hash；state/evaluator只作评估，不进入guard和动作选择。此为
open-loop计划+safety presence guard，不是MPC，也不保证阻止首次出界。

新wrapper先拒绝非v3 plan来源，再检查恰好15steps。结束后核验commands前缀、
attempt/return顺序/计数、journal和RGB hashes。full_H3需完整15returns与fresh终态；
guard中止/termination是partial，异常/stale frame不能冒称实际终态。分类按root：
同context两臂完整15步才full-path；同context、fresh且相同步数的非完整结果为
guard-partial；其余unmatched。stale端点不报pose差。保留所有异常、guard abort、
blocked槽位，无替根/补步/重试。底层branch JSON为通用durable收据；v3 supplemental
endpoint JSON、physical_report、最终report均含v3 schema/来源。

## 可选终态CUDA编码

默认physics-only，不加载模型。`--encode-terminals`：必须先durable保存所有物理分支
及physical_report，再HOST CUDA加载冻结模型；全量验证weights/config，只model.encode，
禁止predict，无CPU神经fallback。每分支fresh actual endpoint最多一次，包括合法guard
端点；8个fresh端点就是8次，不因命令tie减为4。stale端点不编码，不补执行。
实际encode calls/samples/failed_calls与outer attempted_encode_samples分列；setup或
encode失败仍保留收据和所有physics。最终source/model复核；先写report再非零exit。

## 主线程执行

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
.venv/bin/python -m pytest -q tests/test_p3_action_jacobian_physical_consumer_20261027_v3.py
.venv/bin/python -m experiments.spectral_reachability.p3_action_jacobian_physical_consumer_20261027_v3 --prepare --output /home/gbk/code/LeWorldModel/le-wm-main/results/p3_action_jacobian_physical_consumer_20261027_v3_prepared
.venv/bin/python -m experiments.spectral_reachability.p3_action_jacobian_physical_consumer_20261027_v3 --execute-physics --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/p3_action_jacobian_physical_consumer_20261027_v3_prepared --prepare-sha256 REVIEWED_PREPARE_SHA256 --output /home/gbk/code/LeWorldModel/le-wm-main/results/p3_action_jacobian_physical_consumer_20261027_v3_main_v1 --encode-terminals
```

review后使用独立交付的prepare SHA；去掉encoding flag即physics-only，所有output
必须exclusive NEW。四根仍是已开calibration、新独立根0，P1/P3均false；非神经tests
仅验证合同，不构成物理结果。旧v2终点不会追认为v3结果。
