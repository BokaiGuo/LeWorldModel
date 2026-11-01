# Paired quota：真实 child 原生物理交接 v1

独占新增 source/tests/doc；不修改 paired runtime 六文件、原物理桥或任何封存证据。
worker仅RAM/saved测试，不运行CUDA、模型、physics或网络；MAIN审读后执行。
科学P1/P3/C3、eqcompute、freshness、compute/wall优势保持false。

## 可执行链

`prepare → verify → run → consume` 是本新schema的真实入口，不借旧schema白名单。
report和完整strict receipt的绝对路径及SHA均为显式参数，尚未终态不得填写未来SHA。
已知 paired prepare是 `results/agent_p3_paired_quota_runtime_20261028_v1_prepare_MAIN01.json`，
SHA `46e191695e6e121324daf691bafe1b2862a36349e11232822957d8ca6785e6d8`；
它不是CUDA结果，也不能代替report/full strict。

所有外部主边界先核SHA，再完整调用冻结 `paired_runtime.consume`，逐字比较持久完整strict；
因此重做saved数组/账校验但0神经调用。旧原生证据经冻结
`agent_p3_child_native_physical_saved_20261028_v2.consume` 全量认证；
其原始verify/P4.consume/full-array/native PNG/source闭包保留，不只读旧摘要。
历史物理report SHA `56c3e97208597bdc8424dcfb263c8840683c6b3413abeb79d01fa46762dc5df6`。
全部调用均只读；verify/consume重复完整saved校验，可能耗时数分钟，不启动第二次实验。

## 完整分母和复用规则

固定24child×fixed/adaptive=48槽、2known roots，不把24child当独立根。
保留原 `decision`、单列 `execution_quota` 与 pairseal SHA；整pair paid-refusal、
not_reached 均保留，物理结果为null而非0。

按原slot/child/method一一配对，不跨槽搜索、不查看reward/success决定复用：
完整observed ancestry、原任务binding、同source manifest、原执行prefix、非零raw10历史、
全child latent/action context、全部warmup/prefix原生snapshots/body float64 bits、
PNG/file/pixel身份、真实goal向量/goal RGB/goal_pose均须精确同一；
完整raw5用float32 dtype/shape/bytes SHA认证（含+0/-0差别）。
仅同command且完整旧consumer通过的分支可 `reused_saved`，原结果/早停/旧plan SHA原样引用；
`fresh_execution=False`，不计新physics，也不称独立确认。
H1新增12候选可能改变winner，绝不凭word/slot/greedy或winner序号继承旧轨迹。
其余compiled raw5列 `pending_new_physics`，没有旧结果，MAIN真正执行。
旧权威损坏/缺完整consumer直接拒绝，不做结果驱动fallback或强制重跑。

## 原生执行和严格消费

直接调用冻结 `physical.execute` / Store / Meter / native_factory 公共接口，
不修改其globals、不动态exec/AST改写。新runner只调pending分支；复用/拒绝/未到达不创建环境。
每次恢复原同根seed/placement/off-on真实goal，10步原非零warmup及原executedprefix；
全部恢复PNG/native snapshot与原P4逐帧同一才允许selected raw5进入。
原生earlystop立即停止，不补5、不padding、不reset后续动作；错误fail-stop保留48槽与日志。

每pending分支上限（b为原child boundary，own≤5）：constructor/close各1，reset2、
set_state4、goal_render2、goal_setter3、goal_bind1；public≤10+b+5；
physics≤4+10×public；PNG≤12+b+5。setup真实调用数采用已修正saved v2事实，
不是旧错误reset3等硬编码。计attempt/return/pending，nested单位不相加。
恢复warmup+prefix成本全算新费；历史993public/10094physics等只列历史，不计本次实际费。

新strict validator保留冻结完整原生校验，路由/分母改为动态48而非硬编码41/5/2；
另核闭合journal词表、顺序、当前branch归属、全部PNG与guard、raw位模式和目标精确性。
复用分支保留原full_old_row；新分支必须独立完整native日志/PNG/实际账通过。
完整strict输出48项 `physical_result`（缺失null）、new_physical_resources与historical_resources分开。
native success沿原jointXY+wrapped-angle原阈值；native7 reward为7维距离负值，
包含velocity，不能由reward提升推断物理任务成功；同时保留object/jointXY/angle/velocity分栏。

工程cap：output512MiB/final128MiB、journal80MiB、每PNG256KiB、disk floor200MiB；
wall600s从execute入口（不含verify）到cleanup、最终report写入前，不称whole-job600s。
默认拒绝执行，必须 MAIN_APPROVED_PAIRED_QUOTA_NATIVE_PHYSICS 授权；run仅compact summary。
容量/计账/早停/源身份失败不放宽科学门、不隐藏负结果；无新神经加载。

## MAIN命令

所有路径绝对；替换占位符为真实已终态report/完整strict及SHA。不得用prepare SHA填report。

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
/home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python -m pytest -q /home/gbk/code/LeWorldModel/le-wm-main/tests/agent_p3_paired_quota_physical_handoff_20261028_v1_test.py
/home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python -m experiments.spectral_reachability.agent_p3_paired_quota_physical_handoff_20261028_v1 prepare --runtime-report /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_paired_quota_runtime_20261028_v1_cuda_MAIN01/report.json --report-sha256 c8030ebc8c48f74688f4c2ab732c0b2adb184b49a30a4e5ab4bdce5f0bbfbf83 --strict-receipt /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_paired_quota_runtime_20261028_v1_consume_MAIN01.json --strict-sha256 3b62bc4cbf669ecbea10f5ecbf9863a1d96d2432ce029b75059c4edbc2838e6f --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_paired_quota_physical_handoff_20261028_v1_prepare_MAIN01.json
/home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python -m experiments.spectral_reachability.agent_p3_paired_quota_physical_handoff_20261028_v1 verify --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_paired_quota_physical_handoff_20261028_v1_prepare_MAIN01.json --prepare-sha256 <ACTUAL_HANDOFF_PREPARE_SHA>
/home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python -m experiments.spectral_reachability.agent_p3_paired_quota_physical_handoff_20261028_v1 run --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_paired_quota_physical_handoff_20261028_v1_prepare_MAIN01.json --prepare-sha256 <ACTUAL_HANDOFF_PREPARE_SHA> --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_paired_quota_physical_handoff_20261028_v1_MAIN01 --authorization MAIN_APPROVED_PAIRED_QUOTA_NATIVE_PHYSICS
/home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python -m experiments.spectral_reachability.agent_p3_paired_quota_physical_handoff_20261028_v1 consume --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_paired_quota_physical_handoff_20261028_v1_MAIN01 --report-sha256 <ACTUAL_PHYSICAL_REPORT_SHA> --receipt /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_paired_quota_physical_handoff_20261028_v1_consume_MAIN01.json
```

测试使用原封存metadata与明确RAM投影，只检schema/身份/费用/异常；不把fixture统计当新quota实际结果。
MAIN新runtime已终态report绑定为
`/home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_paired_quota_runtime_20261028_v1_cuda_MAIN01/report.json`，
SHA `c8030ebc8c48f74688f4c2ab732c0b2adb184b49a30a4e5ab4bdce5f0bbfbf83`；
MAIN完整strict已真实落盘73,057,807bytes，SHA
`3b62bc4cbf669ecbea10f5ecbf9863a1d96d2432ce029b75059c4edbc2838e6f`；
两侧各634predict，36compiled/10paired-paid-refusal/2notreached。代码仍要求显式实际receipt参数，
缺失/错误立即拒绝。实际复用/待执行数量与新物理结果在MAIN完整交接前unknown。

worker最终测试：39 passed in14.36s。包含零新增环境的RAM run→完整consume，以及native callback
异常后保留48槽、attempt/return与cleanup并拒绝strict；真实旧native完整PNG/census作为saved-schema
回归，不是新paired物理证据。未执行实际prepare/verify/新physics；未重复MAIN新runtime.consume。
另有真实MAIN新report/full strict的只读ABI及全部36compiled raw位模式路由算术回归，
只检精确路由，不替代旧physical.consume、完整prepare或物理科学证据。
