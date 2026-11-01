# control15 I/F真实部署器（MAIN执行）

## 范围与固定权威

独立schema；4个固定job，每job同base的I/F两条新轨迹；原4条O仅继承，合12槽，不去重、不按Q省略selected0、不替换缺失导出。
job0/1为root0 transfer/joint，job2/3为root1 transfer/joint。
继承MAIN完整model/native/RGB严格receipt，`full_upstream_rebuilt_here=False`。
prepare/verify重新读取并hash全部闭包、原始task、所有模型输出artifact、原native journal及本三件套；不回调上游fullstrict。
模型四权威固定于source.PINS（不是CLI可替换输入）：

|输入（results/下）|SHA256|
|---|---|
|agent_p2_full30_control15_feedback_20261029_v2_prepare_MAIN01.json|498b561855d7859681a18ed47e0157908a146a796145b929b5436a1e73b79630|
|agent_p2_full30_control15_feedback_20261029_v2_cuda_MAIN01/report.json|6e2c82633140e071c880aed92c3cde025635cf7dd92c6687d9e0334bc30db3a8|
|agent_p2_full30_control15_feedback_20261029_v2_cuda_MAIN01/completion.json|8622879f9c674d6acffcbc6646fbaccf25023486b0534626c3b63a5bab1d7329|
|agent_p2_full30_control15_feedback_20261029_v2_consume_MAIN01.json|4e7d5e1f6b4b03b28b68eb8a31016211f7d1e0ff3115671923b6df2e97f3d893|

原native authority由冻结planner.E.ACTUAL逐项绑定：prepare04d4a887…、report39dae04a…、receiptaabc5cce…；具体完整path/hash保存在prepare.bindings，不伪造旧receipt或旧full15 validator。
批准协议SHA de6fdc85a131d1a0a1eac3d27c2cf1450b2cbaf67563f56aa346169e50d4f0f8保持不变。

## 执行与消费

每条独立constructor/reset/place/warm10，原goal与task不变；完整执行原prefix15，连续接所选suffix15，cp15不hydrate、不reset。
81个warm/prefix观察（off/on/off）及25个返回transition必须逐字段等于原parent journal，包含native7、snapshot、像素和PNG hash，仅剔除路径及链scope字段。持久化feedback_prefix_guard后才允许control16 intent；此前已有初始native history guard。
input context/model/witness/domain/raw30/cast chart/context seam/Q均绑定，原incumbent舍入语义继承冻结planner，不重新优化。
consumer独立重建本schema全部event hashchain、嵌套intent/return、计数、状态、动作、guard时序、每张RGB224 PNG与像素hash、最终输入闭包及外部completion SHA。
所有早停按实际时间返回；缺cp为missing/null，不padding到30；异常保留first cause、partial、已付attempt和cleanup错误。超wall不可签完整成功；遗留partial仅保全，不等于严格通过。
join明确继承四份已完成own consumer receipt的path/SHA及bindings，非再次执行四份consumer；返回12槽和各资源attempt/completed总数。
P1/P2/mechanism/physical_release/science/fresh/equalcompute均false；0模型/训练；known2不是独立确认。

## 完整字段预算（非删字段估算）

每job完整两轨迹：2env、4reset、80public、808physics、252PNG。
四job合8env、16reset、320public、3232physics、1008PNG；原O成本另列继承。
硬组cap：252×256KiB=63MiB PNG；journal32MiB；其他metadata16MiB；final reserve8MiB；合119MiB，低于总128MiB，floor200MiB。
每次write前检查总量/组量/单PNG/剩余空间；不扩cap、不删IO、不缩帧。metadata单文件受16MiB约束。
真实RAM fake callbacks完整输出（包含report/执行binding/completion、所有事件、全部重复plan）实际约4.66MB，其中journal约3.31MB、metadata约1.18MB，252张低熵fixture PNG约174KB；不能把此压缩图大小当真实PNG保证。
`shape_budget`改将所有JSON浮点按32字符展开，全部PNG按256KiB计，并另加8MiB reserve：完整fixture上80,945,303 bytes（journal4,722,736、metadata1,773,671、PNG66,060,288、reserve8,388,608）。后续完整磁盘build测试还覆盖prepare/source closure实际运输。
逐job whole900s从verify前开始，包含verify/hash/运输、native、report fsync与hash；外部completion绑定report/journal SHA和该完成边界。四job wall相加单列，不冒充含排队/idle的orchestration wall。

完整临时磁盘build实际夹具prepare（job0..3）为2,463,809 / 2,479,384 / 2,463,809 / 2,479,384 bytes；这是RAM合成输入通过真实material/prepare/verify后的运输量，不冒称已对生产prepare执行验收。保存真实资产的只读assemble接线已覆盖全部4job/8plan，生产material闭包由MAIN执行。

## MAIN精确CLI

在`/home/gbk/code/LeWorldModel/le-wm-main`运行。以下循环固定执行job0,1,2,3，无Q/结果筛选；所有输出exclusive，失败保留原目录，勿重跑覆盖。worker未执行这些命令。

```bash
module=experiments.spectral_reachability.agent_p2_full30_control15_feedback_physical_20261029_v1
stem=agent_p2_full30_control15_feedback_physical_20261029_v1
for job in 0 1 2 3; do
  prepared="results/${stem}_prepare_J${job}_MAIN01.json"
  output="results/${stem}_J${job}_MAIN01"
  .venv/bin/python -m "$module" prepare --job "$job" --output "$prepared" || break
  prepare_sha=$(sha256sum "$prepared" | cut -d' ' -f1)
  .venv/bin/python -m "$module" verify --prepared "$prepared" --prepare-sha "$prepare_sha" || break
  .venv/bin/python -m "$module" run --prepared "$prepared" --prepare-sha "$prepare_sha" --output "$output" --authorization MAIN_FULL30_CONTROL15_FEEDBACK_PHYSICS_ONLY || break
  report_sha=$(sha256sum "$output/report.json" | cut -d' ' -f1)
  completion_sha=$(sha256sum "$output/completion.json" | cut -d' ' -f1)
  .venv/bin/python -m "$module" consume --output "$output" --report-sha "$report_sha" --completion-sha "$completion_sha" --receipt "results/${stem}_consume_J${job}_MAIN01.json" || break
done
# 全四own receipt实际存在后：不允许缺job/重复job。
.venv/bin/python -m "$module" join --job-receipts \
  "results/${stem}_consume_J0_MAIN01.json" "results/${stem}_consume_J1_MAIN01.json" \
  "results/${stem}_consume_J2_MAIN01.json" "results/${stem}_consume_J3_MAIN01.json" \
  --job-receipt-shas \
  "$(sha256sum "results/${stem}_consume_J0_MAIN01.json" | cut -d' ' -f1)" \
  "$(sha256sum "results/${stem}_consume_J1_MAIN01.json" | cut -d' ' -f1)" \
  "$(sha256sum "results/${stem}_consume_J2_MAIN01.json" | cut -d' ' -f1)" \
  "$(sha256sum "results/${stem}_consume_J3_MAIN01.json" | cut -d' ' -f1)" \
  --receipt "results/${stem}_joined_MAIN01.json"
```

## RAM/临时磁盘验收与边界

```bash
.venv/bin/python -m pytest -q tests/test_agent_p2_full30_control15_feedback_physical_20261029_v1.py
```

无worker生产prepare/fullstrict/NN/physics。fake Env仅测试计账、原生保存ABI、早停及cleanup；不是物理正确性证据。
最终回归修复consumer作用域：Meter的operation branch位于receipt内，必须加入scope才能严格拒绝guard晚于suffix intent；负例重封hashchain仍拒绝，不放宽gate。
完整build临时磁盘夹具经真实pin reader/原始journal/artifact inventory/源码闭包/own prepare读写与verify，不用material smoke替代；夹具替换仅本新模块依赖seams，不修改冻结模块globals。
包含完整两轨迹/252图消费、早停/失败/cleanup、写前硬cap、过期verify零dispatch、guard重封移位、model export篡改、磁盘build的artifact增删/改与task/prepare SHA篡改、四job join总计及缺项/错序拒绝。
