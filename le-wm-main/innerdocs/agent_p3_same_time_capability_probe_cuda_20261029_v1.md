# 阶段B：7同job H1信号 / 5固定U 的 HOSTCUDA runner

新增独占 source/tests/本文。沿已批准 `agent_p3_secant_same_time_capability_protocol_20261029_v1.md`
（SHA b35af390fc74e6ebd44a9879784308ec74947fb28f640ce5e795cc263e65dccf），
不改原 solver / DAMPING=.002 / CAP=sqrt(5)*.75 / suffix chart / 原Q。
遵循 experiment-code 的固定计划、持久日志及RAM回归；不采用训练、阈值、候选或择优扩展。
worker只实现/RAM/保存ABI检查，不生产prepare、NN、physics或fullstrict；MAIN审读后执行。

## 实际上游与继承范围

Jason bridge source已冻结：40d51e4d5ad5fce5db9417fecc639e0288f286b678a5e7b0eb4e22adea392b77。
只读取其完整 prepare/report/completion/consume、cp10_tasks.json 与bindings，
不调用旧bridge/physical生产consumer、不重编码、不假造旧schema。
四authority逐字节外部SHA并交叉join；完整5 tasks与已认证receipt精确相等。
从prepare.rows[i].cp5_task精确去重出同根两份cp5参考；同根不同role的旧task必须完全相等。
新task严格重建Jason namespace及完整binding，保留base/cp5/endpoint三种身份、
原goal/model/projection/normalizer，核control0/5/10对应abs10/15/20三帧history、
prefix5+role5的raw10与恰一次原f32 subtract/divide；state只作来源/guard证据，不进入模型任务。
bp.bindings + receipt.bindings完整继承并逐文件SHA，加入本三件套、协议、原native/solver/loader源。
不依赖进程当前已加载模块集合生成不稳定sourceclosure，不覆盖继承SHA。

实际来源（共同绝对根 `/home/gbk/code/LeWorldModel/le-wm-main/results/`）：

|输入|路径|SHA256|
|---|---|---|
|prepare|agent_p3_same_time_cp10_rgb_bridge_20261029_v1_prepare_MAIN01.json|f2fac71516c36bdb654a9a84bbc0115d6155c04ed10231bc6b87b99ed0bda846|
|report|agent_p3_same_time_cp10_rgb_bridge_20261029_v1_cuda_MAIN01/report.json|b5744f311d3f412cfc5016bf37c8854cc7b5272a6c37e62055c7f3aa68f2c657|
|completion|同目录/completion.json|37588f050ed9381b191673e27c0b6fcab3ddf0c726e63a54bb75d9579963facf|
|full receipt|agent_p3_same_time_cp10_rgb_bridge_20261029_v1_consume_MAIN01.json|af67caf8536a9637d709fcfb11c947d993e566bd4de5914725f4640a75de4d6f|
|tasks|report目录/cp10_tasks.json|1ff29caa35acc862ce63d64f157024cb61d3a7e1ad76ad1d43f7ba11a09d8e7d|

已对上述保存文件完成只读assemble ABI smoke：7 tasks/5 endpoints；无新J/solve/编码或生产prepare。
完整上游fullstrict是MAIN已完成且本版显式继承的信任起点，不宣称本版重新认证历史物理/encoder。

## 固定执行顺序与真实账

signals顺序：cp5/root0、cp5/root1、cp10 `(r0,C),(r0,Z),(r0,P),(r1,Z),(r1,P)`。
每项原native.nominal(rawzero) + AD primal + 完整192×10 J / 192 scalar VJP；
原bounded_ridge每项恰一次solve/arraySVD，保存完整10 singular_values、params、R和ridge分项。
两cp5只参考，不编译/执行其动作。

全部7完成后，对每cp10仅一个U：原components.suffix_raw(params)，
float64 tanh→float32 raw5；原NativeTree按原mean/std恰一次normalize。
各U独立whole/segmented H1各一次，保存所有native IO/完整shifted context并exact相等。
全7J/solve和全5U双重放durable native_seal.json，再发native_sealed事件，之后才算22原Q：
5 actual cp10 +7 nominal +10 replay endpoint。无排序、阈值或第二候选，无结果筛选。

|新付轴|固定完整值|
|---|---:|
|predict API/samples；AE/predictor/pred_proj calls各|24|
|predict/AE/predictor context tokens；pred_proj rows各|72|
|J API / AD primal predict各|7|
|VJP / grad API / raw_backward各|1344|
|Jacobian elements|13440|
|signal_compile / solve / arraySVD各|7|
|program_compile|5|
|replay|10|
|metric|22|
|encoder/projector/physics/training|0|

资源账实际entry/return/failure hook计量，失败不退款；不以caps冒充actual。
FLOPs/backward_engine_calls/cuda_ns保持null。同步guard及实际CPU monotonic事件时间保留，
嵌套operation wall不可相加当总墙时。CUDA device0、原frozen eval模型/全部buffers前后hash、
installed class源inventory、TF32off/noautocast/MATH开关before/after；不称kernel dispatch证书。
只有run入口导入torch，绝不CPU神经fallback、训练、模型改写或旧global patch。

## Shape、失败运输和最终出版证明

shape_census()列全部原生重复数组，不按unique tensor打折：每signal 378,928B、每H1 replay25,840B，
7+10共2,910,896B原始f32；含192方向每项output192/raw10/basis192/gradreturn10/rawbackward10。
完整4193资源operations /8386 attempt-return records，完整native和应用事件少于16000。
默认hashchain记录上限64KiB；17 call_attempt各≤1MiB，17 call_return≤128KiB，
7 signal_return及5 probe_return各≤256KiB；特别记录作为保守额外项再算一次，不凭压缩率。
binding≤32MiB、classes≤16MiB。常规出版上界1,122,107,392B。
加256MiB reserve，全目录事前上界**1,390,542,848B < 2GiB**。
reserve内seal8MiB+partial8MiB+report96MiB+completion8MiB+failure1MiB+最多64×1MiB恢复记录
合计193,986,560B <268,435,456B。真实每record/file/全目录、free floor200MiB均硬核。
wall1800s、GPU allocated3GiB/reserved4GiB；不缩7/5/J精度或省任何VJP来过预算。

own Store/reader使用本版2GiB与1MiB单行上限，不沿旧8/64MiB reader cap。
hashchain完整顺序、未知kind（包括seal之后）、孤立nativeIO、资源未闭合均拒。
mid-AD/J未返回、第三module已返回但外层predict未返回：只保存实际event prefix并正确失败unwind，
不会根据module数合成API return。native failure可作saved_partial闭合核验，绝不出可用probe。
若普通记录额度/磁盘/写入中断导致intent未持久化，store.broken=true，保reserve failure记录与partial report；
这是metadata-only运输不完整，consumer拒绝完整strict，不伪造缺失intent。
未知异常类型、无法独立复现的solve/源/加载失败、字节损坏/无model闭包等可能仅partial metadata可保存，
不保证任意底层I/O故障都能完整出版，不重试挑成功。

report fsync+目录fsync+hash后写completion，其inclusive_wall_seconds明确排除completion自身写入。
实际run在completion自身fsync后再查1800s；越界写publication_failure.json，所有consumer拒绝。
不能把已记录before-self-write时间冒称after-fsync精确时间；stdout另回实际post-fsync elapsed。
最终partial/report同样受2GiB硬cap。失败不换目录重跑或放松科学门。

## Own saved consumer / 阶段C witness ABI

consume校验全文件SHA/census、binding、source/model/class/precision、journal与全部原生IO。
复用冻结native.audit_native校验native buffers/192逐direction rows，
用inert SavedNative重放固定driver/resource树并独立重算7原solve和22Q（saved CPU NumPy，不是NN），
逐事件比较exact固定顺序；7 saved SVD明确另栏，不冒原run免费计算。
本版严格7 signals/5 probes/17 native invocations/22Q，无额外候选或择优。

完整own receipt含`fixed_signals=7,fixed_probes=5,signals[7],probes[5],plans[5]`。
plans固定0..4顺序与ORDER一致，每项自有 `.probe_plan` schema：
`root_slot/role/checkpoint=10/probe_checkpoint=15/task/task_sha256/endpoint_identity/source/base_task_id`；
`causal_raw10/causal_raw10_bits`；`raw5/raw_float32_bits/raw_float32_sha256/params_float64`；
完整`signal/solve/nominal_context/whole/segmented/Q`；
`native_call_ids(signal=i+2,whole=7+2i,segmented=8+2i)`；
`authority(prepared/prepare_sha256/report/report_sha256/completion/completion_sha256)`、
`bridge_authority/native_seal_sha256/witness_sha256`。witness digest使用原core compact JSON**无newline**。

`witness --receipt --receipt-sha --index --output`只从外部实际SHA绑定的完整own receipt提取：
全7/5/17/资源轴核验、plan digest/fulltask/rawchart/fullcontext与实际report/completion/prepare SHA；
不再次跑fullstrict或NN。输出`.witness`含完整plan与receipt path/SHA，供Pauli按真实schema接线。
partial与RAM receipt禁止该出口；schema/fixture本身不是生产witness。物理工程执行仍须MAIN另授权。
原C另三no_witness、五角色及两known根不因新probe变fresh或独立根。
所有P1/P2/P3/C3/scientific/physical_release/freshness/independent_confirmation/equal_FLOPs flags false。

## MAIN CLI（所有文件参数canonical absolute；worker不执行）

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
.venv/bin/python -m pytest -q tests/test_agent_p3_same_time_capability_probe_cuda_20261029_v1.py
M=experiments.spectral_reachability.agent_p3_same_time_capability_probe_cuda_20261029_v1
P=/home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_same_time_capability_probe_cuda_20261029_v1_prepare_MAIN01.json
O=/home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_same_time_capability_probe_cuda_20261029_v1_cuda_MAIN01
R=/home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_same_time_capability_probe_cuda_20261029_v1_consume_MAIN01.json
.venv/bin/python -m "$M" prepare --output "$P" \
 --bridge-prepare /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_same_time_cp10_rgb_bridge_20261029_v1_prepare_MAIN01.json \
 --bridge-prepare-sha f2fac71516c36bdb654a9a84bbc0115d6155c04ed10231bc6b87b99ed0bda846 \
 --bridge-report /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_same_time_cp10_rgb_bridge_20261029_v1_cuda_MAIN01/report.json \
 --bridge-report-sha b5744f311d3f412cfc5016bf37c8854cc7b5272a6c37e62055c7f3aa68f2c657 \
 --bridge-completion /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_same_time_cp10_rgb_bridge_20261029_v1_cuda_MAIN01/completion.json \
 --bridge-completion-sha 37588f050ed9381b191673e27c0b6fcab3ddf0c726e63a54bb75d9579963facf \
 --bridge-receipt /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_same_time_cp10_rgb_bridge_20261029_v1_consume_MAIN01.json \
 --bridge-receipt-sha af67caf8536a9637d709fcfb11c947d993e566bd4de5914725f4640a75de4d6f \
 --bridge-source /home/gbk/code/LeWorldModel/le-wm-main/experiments/spectral_reachability/agent_p3_same_time_cp10_rgb_bridge_20261029_v1.py \
 --bridge-source-sha 40d51e4d5ad5fce5db9417fecc639e0288f286b678a5e7b0eb4e22adea392b77
PREP_SHA=$(sha256sum "$P" | cut -d' ' -f1)
.venv/bin/python -m "$M" verify --prepared "$P" --prepare-sha "$PREP_SHA"
.venv/bin/python -m "$M" run --prepared "$P" --prepare-sha "$PREP_SHA" --output "$O" --authorization MAIN_APPROVED_SAME_TIME_CAPABILITY_HOST_CUDA
REPORT_SHA=$(sha256sum "$O/report.json" | cut -d' ' -f1)
COMPLETION_SHA=$(sha256sum "$O/completion.json" | cut -d' ' -f1)
.venv/bin/python -m "$M" consume --prepared "$P" --prepare-sha "$PREP_SHA" --output "$O" --report-sha "$REPORT_SHA" --completion-sha "$COMPLETION_SHA" --receipt "$R"
RECEIPT_SHA=$(sha256sum "$R" | cut -d' ' -f1)
for i in 0 1 2 3 4; do
 .venv/bin/python -m "$M" witness --receipt "$R" --receipt-sha "$RECEIPT_SHA" --index "$i" --output "/home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_same_time_capability_probe_cuda_20261029_v1_witness${i}_MAIN01.json"
done
```

CLI只打印compact路径/SHA/status；所有输出exclusive-create，不覆盖。MAIN须逐条核exitcode，
不得在run partial后仍导出witness。worker未实际生成本版prepare/report/receipt或任何生产witness。

## 测试与当前交付边界

测试覆盖明确RAM全7/5/1344/24/22 native transcripts、J/grad/raw/older context/unknown事件篡改、
midAD与第三module返回后错误的非退款partial、ordinary allowance耗尽intent丢失拒绝认证、
真实临时文件build→verify→Store→publication→own reader/consumer和完整来源SHA、最终fsync超时，
源码冷import无torch/pymunk及run授权前拒绝。RAM不能作神经/物理证据；实际预算/精度/模型调用仍待MAIN实跑。
首轮28 passed /1 failed（21.34s）：仅chart单测把ndarray传入原literal-list ABI；
已仅将fixture调用改为params.tolist()，不改冻结组件。随后29 passed /21.75s；
补齐完整witness正向RAM出口、内容hash篡改与公开出口拒RAM后，最终 **29 passed /22.40s**。
临时磁盘测试未patch build/verify/consumer/source pins或原算术，只用显式NumpyNative替代真实模型；
内部`_consume/_witness(...permit_fixture=True)`产物强制标RAM，公开CLI无该开关，不能升级为HOST证据。

三件套至此冻结，尚无本版真实prepare、CUDA report、完整HOST receipt或生产witness。
已知输入ABI闭合；MAIN审读/独立回归/实际prepare→verify→HOSTCUDA→consume→witness是剩余前置。
本版不运行阶段C物理；schema/RAM通过绝非production witness或科学准入。
