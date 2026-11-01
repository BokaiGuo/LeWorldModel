# 新matched-domain beam两根真实RGB桥与CUDA evaluator

独占本stem source/tests/本文；MAIN审读并运行HOSTCUDA。不是再运行beam搜索或physics，不把旧beam16x4产物重标。固定原moving两known roots，每根新beam own raw10；不挑根/臂、无训练/拟合/Qfit/阈值、无CPU神经fallback、无P1准入。

## 输入与权威

- 固定actual physical report SHA `01d5485208fe40a71a53ae20bb2920740894d6ef7c114c139a991f4707e5ebe7`，原prepare SHA `2430db7624f3d24ba80af78313b32ac2be168d45383a484f3aed91d5270fab43`。
- 固定beam planner report SHA `d4976aeca357b5324c74be8b2b7e3bfc254e82b834b3acfe1b0ab90a1ef4614c`，继承288predict/API/samples/AE/proj，不重跑规划。
- 输入桥调用已冻结独立physical saved consumer：核journal hashchain、全部44PNG字节/像素、两row、native完整7维/动作/native reward/实际operation receipts、全源闭包。它只读saved，无模拟。再调用planner ownconsume复核280节点与实际whole/segmented证据；读取两个result.json.gz中whole_contexts与segmented_contexts，必须逐字相等并与原selected plan、raw10和任务精确join。不是将latent近邻当动作。
- 每根使用原moving encoder的`load_task(path,external_sha)`。原task两份SHA仍为`a42bfbb1ea692782355e2572a5df2cd048b72cf3e6c41fee40eb12503674959b` / `101add21a7a40591167a6ab8e8b22c3c62cb1fd700c741fd36cd2073e2d5df74`。原非零warmup前5[.1,0]/后5[0,.1]、完整H3 states/two normalized tokens、identity192、task goal和normalizer原样保留；training provenance仍未认证，不再/100、不fit。
- physical自己的root/goal/state身份和plan selection由原sealed consumer检查；本桥再次检查实际plan0 PNG=原task observed10 pixel SHA、plan10 state=physical final_state、全部0/5/10存在。不补图，不以失败减少两根分母。
- 不导入/调用任何source execute或physical execute；不读证书作控制候选。

## 图像顺序、去重与预算

每根固定goal、plan0/root、plan5/prefix、plan10/terminal；当前两根合计**8唯一图像、8encoder API/8samples/8frames、8projector API/8samples/8frames**。一个调用batch1×frame1。原source RGB0/5不再编码；plan0即source RGB10，只编码这一个root实例，不重复编码相同root供whole/segmented各用一次。goal按同来源path+fileSHA+pixelSHA去重，不仅按像素跨来源合并；checkpoints按root/selection/step保留。合同固定两根、六checkpoint，实际图像数在prepare中冻结，禁止事后调整。

原source8encoder/proj继承一次单列；本轮8是新的真实评价成本，不把旧编码说成免费。新增predict/AE/predproj/AD/physics全部0；编码器的projector不是predictor的pred_proj。API与内部module不相加为16次encode，更不声称等FLOPs。

模型字节沿原policy/config/tensorSHA闭包；224RGB/patch14从实际config导出257hidden tokens、192CLS，不硬抄197。HOST CUDA0/MATH、TF32off、autocastoff、eval/frozen；原image transform、`model.encode`与native encoder/projector真实hook。guard禁止predict/actionencoder/predproj与AD；不改变其他模块global或冒用旧runner的execute。

输出32MiB硬cap含2MiB final reserve，free floor200MiB；pre-CUDA要求32+200MiB。prepare独占新文件<=2MiB，输出必须NEW目录。600s自Budget创建（verify/load之前）开始，在dispatch/write/finalize检查，不是可抢占整个blocking native call的超时。没有重试/替补/额外replay。

## 封存与指标

每图保存原NPZ或PNG字节、原uint8 RGB NPY、实际预处理tensor、encoder input/fullhidden、CLS projector input/output和最终192latent，无有损压缩。NPY以gzip无损压缩。每module有独立intent/returned JSON，失败有failed receipt；attempts/returns按真实entry区分，API/sample/frame各列，本批均1。没有臆造render或模型receipt。

所有8图encoding与source/model最终核验完成后，先durable写`encoding_seal.json`，再计算Q、写metrics/report。`tasks.json`保完整原task（含3历史latent/2incoming tokens/原normalizer/原goal）、whole与segmented两个完整successor contexts。评价：

- `Q`：实际checkpoint latent对**原task goal**，float64 mean192 identity。
- `Q_encoded_goal`：实际checkpoint对本轮新编码goal，另列，不悄悄替换原目标。
- `model_Q`：原root或对应whole预测prefix/terminal对原task goal；不再调用预测器。
- `prediction_MSE/maxabs/bitwise`：该预测与同时间实际RGB编码的差；root项是原root再编码一致性，不是新增预测。
- `goal_reencode`：新编码goal对原goal一致性。差异只报告，无事后tolerance、无阈值调参。

native initial/final metrics原样保留，原生成功不由RGB Q决定。native完整状态7维、位置含agent与object；object-only误差不冒充完整任务success。已知两根不是独立样本，P1保持关闭。

consumer不调用`evaluate`/`score_rows`，独立以saved NPY复算sum192/192与模型残差；核输入prepare重建、所有artifact SHA/准确文件分母、module输入→fullhidden CLS→projector→latent链、intent/return、encoding seal及实际计账，最后再核源/模型SHA。成功receipt只证明保存合同闭合，不是理论成功。

partial仍保两rows/8图状态、null Q、已实际module计账与first/secondary errors；不把未执行module报成返回。初始化失败也尽力保存partial。磁盘/IO彻底失败时无法强保证写出最终report，但保留已写文件并传播异常，绝不覆盖原目录或退款重跑。

## MAIN命令

cwd `/home/gbk/code/LeWorldModel/le-wm-main`。本worker不执行prepare/run或神经/物理。

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_kinodynamic_matched_rgb_20261028_v1.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_kinodynamic_matched_rgb_20261028_v1 prepare --physical-report /home/gbk/code/LeWorldModel/le-wm-main/results/agent_kinodynamic_matched_physical_20261028_v1_MAIN01/report.json --physical-sha 01d5485208fe40a71a53ae20bb2920740894d6ef7c114c139a991f4707e5ebe7 --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_kinodynamic_matched_rgb_20261028_v1_prepare_MAIN01.json
```

以下`$PREPARE_SHA`为上述实际prepare stdout值，不能预填虚假SHA；`$RGB_REPORT_SHA`为实际terminal后report外部SHA。

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_kinodynamic_matched_rgb_20261028_v1 verify --prepared results/agent_kinodynamic_matched_rgb_20261028_v1_prepare_MAIN01.json --prepare-sha "$PREPARE_SHA"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_kinodynamic_matched_rgb_20261028_v1 run --prepared results/agent_kinodynamic_matched_rgb_20261028_v1_prepare_MAIN01.json --prepare-sha "$PREPARE_SHA" --output results/agent_kinodynamic_matched_rgb_20261028_v1_cuda_MAIN01 --authorization MAIN_MATCHED_DOMAIN_BEAM_RGB_CUDA_ONLY
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_kinodynamic_matched_rgb_20261028_v1 consume --output results/agent_kinodynamic_matched_rgb_20261028_v1_cuda_MAIN01 --report-sha "$RGB_REPORT_SHA"
```

源码/tests/doc一起绑定prepare，交付冻结后不再修改。MAIN实际GPU资源准入、代码审读和结果consume仍待执行；RAM fixtures不是CUDA科学证据。无需任何新physics或beam计划。

## Worker实际验证范围

实际saved-only `build()` smoke已成功：两rows、八图、223bindings；config导出hidden=[1,257,192]。当时canonical prepare大小526884B（未落盘，后续本三文件的hash值更新不改变字节长度）；低于2MiB上限。它消费的是原actual44PNG/journal与whole/segmented保存证据，不运行模型或物理。没有创建prepare或新results目录。RAM测试覆盖缺checkpoint/错误臂/动作/normalizer/goal/源SHA、whole-segmented差异、独立指标复算、module数组篡改、全artifact分母、计账、CUDA数值guard、interrupt/最终核验失败，以及wall/disk cap。实际测试数和最终三SHA见冻结交付；不能把mock里标注的CUDA receipt当主机CUDA证据。
