# Mixedmode222：固定四项真实物理 adapter

只新增本 stem 的 source/tests/本文，旧源码/封存结果不改，不修改旧 globals，不投射成旧 schema。
本 worker 只做 RAM fake/微小临时 metadata 测试，不运行真实 prepare/consume、CUDA、神经、physics 或下载。
MAIN 审核并执行；实现/测试完成不是 P1 通过。

## 固定真实权威与完整消费

新生产者 `agent_mixedmode_prefix_20261028_v1`：

- source SHA `ec912669c678960aac0d004a2e764b4bf036194673da5d1c63d0b535bc199396`
- prepare `results/agent_mixedmode_prefix_20261028_v1_prepare_MAIN01.json` SHA `8f6d027116b76334e80f9a761688ffd33825bc25d825484012d37230f083ed6b`
- 实际 terminal report `results/agent_mixedmode_prefix_20261028_v1_cuda_MAIN01/report.json` SHA `d6166b98958af70cecb8529135396c77467b7c803bfe12a1a5de29c422c22ad2`（MAIN提供，不是未来占位）。

固定旧物理 `agent_allmode_physical_20261028_v1_MAIN01`：

- source SHA `c0842f63685718e3d1d562fbbe5e2e3fdf9103954804f4276a1186d8db91efe4`
- report SHA `3ca4891a5f0823bede431e85442513a095b1e9bc8d49c87feefbb56663dca3b4`。

外部 SHA 与固定实际 seal 均必须符合，才完整调用新 producer.consume 和旧 allmode physical.consume。
新 consume 自身验证全部222候选、4任务、5352predict/AE/pred_proj、892J等完整 ABI；这里再核计划schema/arm、
selection_digest、whole_run_report_SHA、task/blind/task_file_SHA、literal raw10及prefix/suffix5+5一致。
旧 physical.consume 完整重建原 allmode producer/source/prepare/物理 journal/PNG闭包。
双方原始 consume 返回完整保存，所有输入、任务、源码、prepare、报告、artifact路径/SHA合并，冲突/漂移拒收。
严格消费会做已有 saved 数组/SVD 算术，并非免费；prepare、verify、run前verify、consume前verify各次调用都要付费，
但没有新模型调用/神经/physics。worker没有替MAIN重复这些真实工作。

## 四分母与继承规则

顺序永远 `(slot0,off),(slot0,on),(slot1,off),(slot1,on)`，只有两个known roots，不是四个独立根。
不按winner index、stagegreedy、预测改善、成功/失败标签删除任务；不混observed4。
继承必须同时满足：完整旧consume通过；原task_id/blind_SHA/task_file_SHA/source_SHA相同；
新旧encoded slot全文（包括全部context、goal、Q及原任务定位）canonical字节一致；
源slot seed/placement、native7 root/goal/goal_pose/条件goalRGB、原warmup17帧参照全部完全一致；
新旧完整float32 raw10逐位相同；旧整行工程完成（native_stopped或horizon_completed）。
任务identity/order错乱直接拒绝；其他不满足则该项 route=execute，必须MAIN授权后整条新运行。
不凭模型greedy跳过新动作，不凭index23未变认定可复用，不根据物理成功决定是否保留根。
当前MAIN另行报告 root0 off/on raw改变、root1 off/on raw相同；这是上游消息，不是本worker实际路线验证结果。
本adapter仍逐项执行上述完整检查，绝不硬编码root1继承。

继承行保存旧row全文在 `inherited_evidence`，原selection/PNG/checkpoint/metric/早停/资源全部保留，
明确 `evidence_origin=allmode_MAIN01_historical_not_fresh`。新行顶层轨迹为空、新计数0；
不把历史步数冒充新执行，也不把顶层0计数解释为旧物理免费。
新执行行 `inherited_evidence=null`，从同根seed+placement+原非零warm10重放，独立env完整执行其新raw10。
禁止旧prefix+新suffix、旧endpoint+新标签、旧/新PNG混拼。部分失败保留4行及unknown/null，不补0，不补checkpoint。
若所有4项可继承，run仍需MAIN授权，但不构造env、所有新增资源计数为0。

## 执行与成功语义

复用冻结allmode公共 `Meter/Store`、place/bind_goal、observe/dispatch/native_hooks、source_material、
root_metadata、reference_for/check_warm_frame；独立新schema的material/路线判定/episode/execute/validate_saved/consume。
不改旧文件或globals，不用旧schema冒充新arm。Store只覆写新独占前缀初始化，其余公共I/O硬限保持一致。
native控制始终off、relative/action_scale100，不额外缩放/tanh/clip新raw；模型/Q/normalizer不变。
warm10严格float32 `(10,2)` 前5 `[.1,0]` 后5 `[0,.1]`，zero/时序/尺度/dtype/shape变更拒绝。
条件off/on history3与真实目标图均精确源绑定；默认goal_pose不是构造goal，禁止编辑目标图或goal_pose。
原source每根只有17个warmup receipt，中间8个on无原参照，仍显式unknown；不制造33帧旧匹配。
每条新轨迹保存真实off/on/off PNG/fileSHA/pixelSHA/native7 snapshot，实际prefix51桥及0/5/10可达checkpoint。
原native终止/截断即停，不padding，不补缺PNG，不借旧行填尾。

**成功与reward不能混称：** native success为联合agent+object XY（前4维）误差范数 `<20`，
并且原native角项 `min(abs(delta_angle),2*pi-abs(delta_angle)) < pi/9`；角度多圈的原native行为不修改。
reward是七维完整state距离的负值，包括最后两维velocity；不是上述成功阈值，7维距离大仍可能native成功。
consume独立复算native7 reward、jointXY、native角项、wrapped角诊断、velocity误差；wrapped仅诊断，不替换成功门。
未知endpoint的metrics=null；不把预测latentQ当真实任务指标。

新增资源轴上限沿用allmode：constructor/close4、reset8、public80、physics816、render264；
按真实dispatch intent/entry/return计账，嵌套资源不相加，失败不退款，未返回constructor的内部工作unknown。
只对新执行行收费；继承成本在旧完整receipt中另列，不声称equal_total_compute/FLOPs。
wall900s操作前/后检查，不能抢占卡住的native调用；MAIN应另行watchdog。
NEW output32MiB/final4MiB/floor200MiB；prepare64MiB保完整mixed大artifact路径hash闭包，不截断。
run默认拒绝，必须精确 `MAIN_MIXEDMODE4_RAW10_PHYSICS_ONLY`，授权检查早于读取/建目录/env导入。
P1/P3/scientific_admission/freshness/independent_confirmation/equal_total_compute/equal_FLOPs全false；fresh_roots=0。

## MAIN精确命令

仅MAIN在审读后执行；以下不会启动任何新模型。新输出必须不存在。

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
.venv/bin/python -B -m experiments.spectral_reachability.agent_mixedmode_physical_20261028_v1 prepare \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_mixedmode_physical_20261028_v1_prepare_MAIN01.json \
  --producer-output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_mixedmode_prefix_20261028_v1_cuda_MAIN01 \
  --producer-report-sha d6166b98958af70cecb8529135396c77467b7c803bfe12a1a5de29c422c22ad2 \
  --producer-source-sha ec912669c678960aac0d004a2e764b4bf036194673da5d1c63d0b535bc199396 \
  --producer-prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_mixedmode_prefix_20261028_v1_prepare_MAIN01.json \
  --producer-prepare-sha 8f6d027116b76334e80f9a761688ffd33825bc25d825484012d37230f083ed6b \
  --old-output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_allmode_physical_20261028_v1_MAIN01 \
  --old-report-sha 3ca4891a5f0823bede431e85442513a095b1e9bc8d49c87feefbb56663dca3b4 \
  --old-source-sha c0842f63685718e3d1d562fbbe5e2e3fdf9103954804f4276a1186d8db91efe4
sha256sum results/agent_mixedmode_physical_20261028_v1_prepare_MAIN01.json
```

用实际新prepare SHA替换 `ACTUAL_OWN_PREPARE_SHA`，不得预造：

```bash
.venv/bin/python -B -m experiments.spectral_reachability.agent_mixedmode_physical_20261028_v1 verify \
  --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_mixedmode_physical_20261028_v1_prepare_MAIN01.json \
  --prepare-sha ACTUAL_OWN_PREPARE_SHA
.venv/bin/python -B -m experiments.spectral_reachability.agent_mixedmode_physical_20261028_v1 run \
  --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_mixedmode_physical_20261028_v1_prepare_MAIN01.json \
  --prepare-sha ACTUAL_OWN_PREPARE_SHA \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_mixedmode_physical_20261028_v1_MAIN01 \
  --authorization MAIN_MIXEDMODE4_RAW10_PHYSICS_ONLY
sha256sum results/agent_mixedmode_physical_20261028_v1_MAIN01/report.json
```

物理report真实落盘后才取得并使用其SHA：

```bash
.venv/bin/python -B -m experiments.spectral_reachability.agent_mixedmode_physical_20261028_v1 consume \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_mixedmode_physical_20261028_v1_MAIN01 \
  --report-sha ACTUAL_NEW_PHYSICAL_REPORT_SHA
```

consume只读saved：重建双consume/prepare、严格新schema/journal嵌套receipt及完整PNG字节/像素hash、
路线与整行继承、原始raw、native早停/真实checkpoint、源goal/renderer、原7维指标和new-only资源计账。
任一失败抛错；真实partial产物保持partial/unknown，不能冒充全4物理通过。

## RAM验证与未运行项

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_mixedmode_physical_20261028_v1.py
```

测试使用复用的RAM Env/PNG/journal和微小tmp metadata；旧producer/physical只mock saved ABI边界，不触发真实consume。
覆盖5352/固定4/seal和full222范围、raw/context/source不同需整条执行、root0新/root1继承、全部继承零新增env、
native早停不padding、未知构造失败4分母、非零warmup错误拒绝、重封journal篡改/继承拼接拒绝、
真实PNG字节与schema/goal/资源检查、双consume失败传播/完整闭包、授权/NEW/floor/cap/prepare篡改。
未运行真实prepare/run/consume，未确认本批继承数量，无P1/freshness或物理结果声明。
