# NEW equal144 beam / spectral runner — MAIN-approved prospective design

本stem三文件独占。旧144/88合同、所有旧代码与结果不改。本设计由MAIN在新运行前明确批准；不是事后改旧配额或回证原加载。仅同WM predict API/batch1 predictor samples匹配，不是等FLOPs/AD/候选分布或P2科学准入。worker仅实现、RAM/saved metadata tests；MAIN完整审读后prepare/运行HOST CUDA。

## 固定四臂与预算

两份原prepare绑定同样两moving任务，完整source_slot0/1、task_id、goal、native192 identity Q、非零incoming raw10（5×[float32 .1,0]然后5×[0,float32 .1]）、normalizer及packet logit ball全部继承。root-major各执行旧beam与新谱臂，四行预置，全部结果/负结果/partial保留。

beam原函数beam.search与bank/seeds不改：首20→beam6→120，独立whole2+segmented2，总144predict；不是RRT或论文全kinodynamic实现。新谱臂 joint_H2_spectral_top16_packet_mapped_equal144：zero1+DLS(.25,.5,1)3+top16×2radii×2sign64=68。原192×20 full-H2 zero-logit J、damping.002、exact bounded ridge与原SVD符号约定（最大绝对坐标为正，首argmax tie）、joint radii .25√10/.75√10保持。joint bank后按原screen.projected_logits投影逐packet .75√5球，再float32 tanh，不重新定义动作域或normalizer。零奇异模式仍保LAPACK基底，不按结果删候选/去重或redraw。

每谱臂 nominal2+AD-forward2+68candidate×2+winner whole2+segmented2=144。全库真实原Q、stable first argmin，选择前durable冻结bank、全部candidate评估后冻结winner，之后两个真实独立H2 replay。AD-forward已计144内；另列每谱臂1J/192VJP/2SVD/1boundedsolve，两根合2J/384VJP/4SVD/2solve。全job576predict API/samples、各576AE/predictor/pred_proj；beam0AD，不能将API与内部模块相加当预测数或FLOPs。CPU saved consume再次重建bank/SVD属于验证成本，非CUDA planner的新调用，不隐瞒也不混入144。

继承原report真实8encoder/8projector共享一次，两方法不重复计为16；其他上游unknown保null。新encoder/projector/physics=0，由真实guard阻止新图像神经调用；无训练、Qfit、重试、padding、科学阈值或筛根。

## 实际执行接线与同buffer加载

旧beam/spectral/loader源码SHA固定到原prepare。新prepare合并原完整bindings（含模型权重/config、官方模型类、各数学helper、task/encoder产物与源闭包），再绑定新source/tests/doc。verify全文件重hash并全量重建；prepare小于256KiB、不复制大模型。

新NativeCuda继承封存sampling.NativeCuda的真实model.predict调用与action_encoder/predictor/pred_proj pre/post hooks。没有全局或实例method monkeypatch。每packet完整模型模块输入输出shape/dtype/nativefloat32值保留，API独立预dispatch，batch/sample单列，失败不退款。谱的联合导数复用原NativeCuda.jacobian逐192标量autograd.grad；采样native_step没有detach计算图，receipt的detach仅用于保存。所有模型参数冻结，输入raw需要梯度仅用于联合J。

新loaded_model显式单次读取config/weights为immutable bytes；在调用前journal load_intent，hash该同一bytes，config由该buffer json.loads，weights由同buffer BytesIO交torch.load(weights_only=True,map_location=cpu)。复用原checkpoint_compat.remap_legacy_vit_keys、原config instantiate、strict=False后missing/unexpected必须空、插值设置；随后CUDA/frozen/eval及原tensor hash。CPU state-dict staging是原加载机制，**不执行CPU神经推理**。保loader.json中两文件路径/长度/hash-buffer/deserialize-buffer/SHA/same_buffer，以及新adapter源码SHA、旧兼容loader语义SHA、prepareSHA、remap数量、tensor hash；执行完再核模型/源/权重文件。新receipt仅证新invocation，不回证旧运行。adapter自有加载不是给旧loader冒造receipt。

## 保存、消费与工程cap

新prepare→verify→HOST CUDA run→consume均可执行。64MiB整个四臂job、200MiBfloor、900s工程wall，沿已有Store真实cap/fsync/gzip协议；起点为output creation，涵盖load/search/dispatch/write，不涵盖此前prepare verify，且无法抢占阻塞native call。非final写受wall gate，失败保部分文件；footer/report尽力保存。固定四行不因failure减分母，无成功筛选。8MiB展开单result、256MiB完整lossless journal上限沿现有reader，输出64MiB不放宽为磁盘无限。无需物理PNG或新encode。

strict consume全量：外部reportSHA→prepare/source/model/task闭包、完整4row/11artifacts+report、同buffer loader receipts与journal、原commands/goal/Q、所有API/module/math intent-return、逐packet normalized history与实际输入输出/native successor、192J rows、ALL68重建与argmin/replay、原beam搜索树/edge/witness/selection重建。仅saved CPU数组/验证SVD，不运行神经。无report就不能声称actual。输出plans含新selection_sha256、rawcommands、task_export/SHA、blindSHA、prediction/cost和whole_run_report_SHA，供未来另物理adapter显式接线，不能把旧physical arm重标成新谱结果。

## MAIN CLI（cwd=/home/gbk/code/LeWorldModel/le-wm-main）

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p2_equal144_baseline_cuda_20261028_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_equal144_baseline_cuda_20261028_v1 prepare --output results/agent_p2_equal144_baseline_cuda_20261028_v1_prepare_MAIN01.json
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_equal144_baseline_cuda_20261028_v1 verify --prepared results/agent_p2_equal144_baseline_cuda_20261028_v1_prepare_MAIN01.json --prepare-sha ACTUAL_NEW_PREPARE_SHA
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_equal144_baseline_cuda_20261028_v1 run --prepared results/agent_p2_equal144_baseline_cuda_20261028_v1_prepare_MAIN01.json --prepare-sha ACTUAL_NEW_PREPARE_SHA --output results/agent_p2_equal144_baseline_cuda_20261028_v1_cuda_MAIN01
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_equal144_baseline_cuda_20261028_v1 consume --output results/agent_p2_equal144_baseline_cuda_20261028_v1_cuda_MAIN01 --report-sha ACTUAL_NEW_REPORT_SHA
```

ACTUAL_*由MAIN新执行后回填；不预填未来SHA/成功。RAMmock测试假模型收据只存在于内存，明确不是实际neural结果。已知两根仅calibration，P1/P2 false，无正式composition收益结论。
