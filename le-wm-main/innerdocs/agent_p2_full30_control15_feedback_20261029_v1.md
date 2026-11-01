# full30 control15：O/I/F固定因果输入对照实现

## 合同与边界

四个独占新文件：同stem runner、`_kernel.py`、tests与本文；未修改旧源码、模型、结果或批准协议。
遵循协议 `agent_p2_two_program_full30_control15_feedback_protocol_20261029_v1.md`
SHA `75e8604604bdd8f6a12eda24367747640a7318ca4525c6dba19c685fd85060d4`，及实际权威补记
`agent_p2_two_program_full30_control15_feedback_authority_20261029_v1.md`
SHA `636c87f88d83c9a69134b1d1949e9f638c325932f295ac7edafda558c3f63895`。
算法域、alpha、单轮次数和原生成功标准未变。不是新物理runner，不声称已闭环部署。
两个已知根、非独立确认；P1/P2/C3/scientific/physical/fresh门全false，FLOPs与engine backward unknown。

固定root0 transfer/joint、root1 transfer/joint四base，各O/I/F，总12策略槽。
O完整引用原openloop raw30，明确继承、不新调用模型；I/F各一独立模型job，合8，绝不去重。
I用原完整imagined cp15 context，F用本轮真实RGB链cp15 context；同base prefix15逐位固定。
各自新付费保留原后缀incumbent，四候选稳定选优，只保证自身输入context上的模型Q非劣。
O对F混合信息与额外优化；I对F才隔离固定优化过程的context信息输入，仍不是跨根科学因果认证。
transfer/joint域与原始付费不同，不能称跨臂纯组合干预或等FLOPs。

## 实际来源、继承与白名单

目录前缀 `/home/gbk/code/LeWorldModel/le-wm-main/`。

|RGB输入|相对path|实际SHA256|
|---|---|---|
|prepare|results/agent_p2_two_program_full30_rgb_encode_20261029_v1_prepare_MAIN01.json|20a248dc940188ff3858fe71769cdc2d726ec23d2e77e8e50783ff777db63527|
|report|results/agent_p2_two_program_full30_rgb_encode_20261029_v1_cuda_MAIN01/report.json|915b6760db8bf30abe16e4bfba347604e6359c17d08c1ff3f2a7a076b7714df4|
|MAIN完整receipt|results/agent_p2_two_program_full30_rgb_encode_20261029_v1_consume_MAIN01.json|42164a225310c39ff83e398f55501e36e8db9876d5b52dbc697d031e31468166|

`build`只接受这些固定pins，继承MAIN完整RGB/model/native证据；不调用上游build/verify/consume/fullstrict。
`full_upstream_rebuilt_here=False`。独立重hash原全部bindings、编码report所有artifacts/completion、
原模型源/权重、当前四文件、批准协议及8个纯metadata解析的torch/transformers installed源；冲突拒绝。
真实加载后每个module类源都必须在此闭包，不skip内建类，权重同buffer loader与before/after frozen hash。
不是因为指针标记“已付”就认证任意新文件；明确继承外部完整receipt，不重新认证旧神经数值。

`material(ep, receipt)`严格join四base任务/原witness/logit/rawbits/normalizer/goal/真实cp5/10/15。
cp15三latent对应control5/10/15、native绝对15/20/25；两个action token对应control6..15，原normalize一次。
cp5的causal raw10包含最后5warm+前5plan，不能使用负数切片只取plan；保存ABI smoke已实际检查此接缝。
只读实际metadata smoke返回4bases/8jobs，未调用完整build或生产prepare，也未写results。
该检查不是模型执行或上游fullstrict。四个原suffix cap按base分别为
2.904737509655563、3.642957079593698、2.904737509655563、2.9047375096555625，I/F同base一致。

传给kernel的job严格只允许14字段：base_index/root_slot/arm/policy/input_context/goal/parent_logits/
incumbent_logits/suffix_cap/prefix_raw/parent_raw_bits/task_id/task_sha256/parent_witness_sha256。
未来实际cp20..30向量、Q、native labels不写入job或own prepare；原未来计划动作是合法冻结incumbent，不是未来实际标签。
来源闭包含旧文件路径/hash，不意味优化器读取其中的终态。测试将未来向量改为非法字符串，生成job不变；
若在job额外注入actual_cp30则拒绝。输入缺失/损坏在固定pins下failclosed，不换根/补帧/替补；
runtime失败保全部8-job/12-policy roster，不跳槽或退款。

## 数值与原生接口

kernel公开`validate/proposals/execute/resources/contract`；backend接口为
`rollout(full_context, packets_float32, tag)`、`jacobian(context, logits30, tag)`、`last_ad_contexts/event`。
transfer suffix球`.75sqrt15`；joint冻结prefix后按原60维球的剩余平方半径，保持原父witness合法语义。
iterate0认证原float64 full witness及原tanh64→cast32 rawbits，不逆atanh、不clip、不修复postcast边界。
新候选核concat full60原域；数值越界failclosed，不能epsilon放宽。
J为literal float32 raw leaf的192×30，3packet primal完整保存；不是60维旧J、不是logit/cast导数。
连续float64 chain为`(2/192)*(J*(1-tanh(u)**2)).T@(z-goal)`。
target=−.5R单位梯度，零梯度保u0；非下降方向置零；alpha固定1/.5/.25/.125，保留全部重复候选。
以真实非线性模型末节点meanQ稳定选(Q,index)，index0 incumbent优先，后续selected/whole/5+10分段逐位重放。

复用未修改native单包transition/HostBackend.observe/validate_packet/loader及hash工具，
本模块自有H3 AD、27-call上限、Store、逐事件strict和新schema；无globals patch或旧receipt impersonation。
逐forward intent/input/三module intent/input/return/predict API/return完整保存；
每J intent/3primal/NPY/return、selection与replay事件精确全时序重建，未知event不滤。
CUDA only、frozen eval、TF32/autocast off/MATH；视觉encoder/projector入口禁止。
实际precision前后均保存并严格消费；加载与逐job peak4GiB allocated/6GiB reserved，未知值不填零。

## 固定计账、运输预算与whole wall

每job：incumbent3 + AD primal3 + 四候选12 + selected3 + whole3 + segment(1+2)3 =27predict/1J。
全8job=216predict/8J，216次各AE/predictor/pred_proj，1536 mathematical requested VJP，46080 J元素，
24 AD primal packet已计入216。0新增encode/projector/SVD/solve/physics/train；engine backward/FLOPs=null。
I/F中incumbent三调用同时提供O suffix在对应context的新评分，不另加费或假称免费旧预测复用。
原228predict/4J、native、28encode分别列inherited_resources，不抵扣本轮新增。

全8job真实Store→gzip单LF hashchain→reader→完整saved consumer的RAM合成回调夹具预算：

|项|bytes|
|---|---:|
|fixture完整未压缩journal|16,021,396|
|fixture完整未压缩8rows|14,705,908|
|逐float按32字符的journal上界|19,371,390|
|逐float按32字符的8rows上界|19,621,560|
|8个完整NPY含header|185,344|
|强制全部runtime JSON metadata额度|4,194,304|
|gzip扩张/逐record flush上界额外量|305,128|
|保守合计|43,677,726|

3068条event每条单独计算compressBound，再显式加64-byte Z_SYNC_FLUSH余量；另留128-byte流封装尾部。
不能仅用整流compressBound忽略每条flush。journal最终压缩上界19,670,286 bytes，另有真实熵运输夹具验证。
合计低于128MiB−2MiB，不依赖模型值可压缩性，不减少IO、候选、J精度或分母。
runtime所有JSON累计4MiB硬限；增量物理输出总128MiB、失败reserve2MiB、floor200MiB。
NPY消费先校验shape/header/dtype/长度再分配，拒绝缩维、尾部/重复gzip member/未知文件。
write在with内flush/fsync，再目录fsync；xb排他，真实磁盘回归验证未关闭fd与singleLF。

whole1800秒从verify前起，单job450秒；包括源/权重读取、模型加载、优化、IO与最终出版。
report.wall_seconds明确为report出版前采样值，**不是**报告fsync/hash后的inclusive完成时间。
run在report fsync/hash后写独立completion（绑定report SHA及该边界elapsed）；返回report SHA和completion SHA。
saved consumer要求两项外部SHA，验证completion边界、elapsed>=report.wall且<=1800，才签完整工程receipt。
completion自身fsync/hash后再runtime deadline检查，越界写terminal_failure使完整inventory拒签。
completion的elapsed不假称包含它自己的后续出版；该后续操作仅由runtime最终deadline硬守卫。
partial绕过普通dispatch时钟门保留首因/所有已发生intent/返还，失败后剩余槽not_started，不虚报complete。

## RAM验收与MAIN命令

本worker仅RAM、临时磁盘、真实保存metadata/hash读取；未生产prepare/NN/physics/fullstrict。
回归覆盖完整216/8J/12slots、原schema/warm cp5、未来标签隔离、float64chart/非零chain/稳定tie、
module/J completed与shape篡改、全journal重封hash后的event移位/未知event/双LF、source漂移、
precision/模型/科学门、外部completion SHA/绑定/超wall/bool/nonfinite、verify耗时耗尽前零dispatch、
逐event sync-flush运输与真实fsync/exclusive writer。合成J=0只作ABI夹具，不是神经导数证据。

MAIN工作目录 `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
.venv/bin/python -m pytest -q -s tests/test_agent_p2_full30_control15_feedback_20261029_v1.py
.venv/bin/python -m experiments.spectral_reachability.agent_p2_full30_control15_feedback_20261029_v1 prepare \
  --output results/agent_p2_full30_control15_feedback_20261029_v1_prepare_MAIN01.json
```

取实际prepare返回SHA，不填未来值：

```bash
PREP_SHA=实际prepare_SHA
.venv/bin/python -m experiments.spectral_reachability.agent_p2_full30_control15_feedback_20261029_v1 verify \
  --prepared results/agent_p2_full30_control15_feedback_20261029_v1_prepare_MAIN01.json --prepare-sha "$PREP_SHA"
.venv/bin/python -m experiments.spectral_reachability.agent_p2_full30_control15_feedback_20261029_v1 run \
  --prepared results/agent_p2_full30_control15_feedback_20261029_v1_prepare_MAIN01.json --prepare-sha "$PREP_SHA" \
  --output results/agent_p2_full30_control15_feedback_20261029_v1_cuda_MAIN01 --authorize-host-cuda
```

从run实际返回分别取report sha256与completion_sha256：

```bash
REPORT_SHA=实际report_SHA
COMPLETION_SHA=实际completion_SHA
.venv/bin/python -m experiments.spectral_reachability.agent_p2_full30_control15_feedback_20261029_v1 consume \
  --output results/agent_p2_full30_control15_feedback_20261029_v1_cuda_MAIN01 \
  --report-sha "$REPORT_SHA" --completion-sha "$COMPLETION_SHA" \
  --receipt results/agent_p2_full30_control15_feedback_20261029_v1_consume_MAIN01.json
```

输出own schema `control15_feedback_verified_NOT_admitted`，8个新full30 plan与完整12 policy_slots。
plan保留parent字段，新增new_suffix_Q/new_contexts/new_witness_sha256，旧预测不伪装新claim。
未来物理须先连续复现prefix15并核真实native/RGB guard，不能control15 reset或默增encoder；
本实现不含物理执行授权。原任务success、同时间指标、早停null合同沿批准协议，未看结果改门槛。
尚待MAIN最终审读→实际prepare/verify/HOSTCUDA/own consume；无新模型结果或未来SHA。
