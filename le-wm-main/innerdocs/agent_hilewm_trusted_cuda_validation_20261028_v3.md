# HiLeWM 四个原用例：source-layout 独立字典 reference v3

只新增本stem源码/测试/本文；不改v1/v2、loader、export或其结果。MAIN审读并运行HOST CUDA，本worker不做生产prepare、神经/物理/网络。experiment-code技能用于固定用例、保存/严格消费和RAM回归，不引入训练、调参或科学发布。

## 改正边界与真实authority

v1两个direct已精确，但旧reference rollout用整个slice+reshape，数值相同却storage stride不同。实际v2验证了source append+stack与native精确、legacy第一步macro_to_condition开始发散、contiguous恢复、source restride再现、旧失败逐位复现。v3不覆盖或声称旧v1消费成功。

prepare/verify调用完整`v2.consume()`重建全部strict，再与持久化receipt整对象比较；`v2.source_evidence()`重新认证官方文本、AST append/stack以及quantize_latents原样return latents。原bytecode身份仍未认证。绑定实际v2 manifest `eadbb1d11383a46b02866a1966588b482dccbb45d6684a55c812d81925782c72`、report `8f19346c99aba5c3cc5dd6efec878356938c24bcb53023444c37791f94ff6771`、receipt `0c2b3055c91cd01101774511e8dff24693ec46caa3a91a7ef43b495eeba0a4d9`。原v2/v1/loader/export source/tests/doc pins、source texts/IR/export数组及本三件套闭包重建，不是status-only。

## 实现与固定费用

原四用例完整保留：direct_B1_T1、direct_B2_T4、rollout_B2_S2_T5、cost_B2_S2_H2_block2；原arange/dyadic输入seed/shape/goal不变。8次顶层dispatch，owned/reference各11 high predictor、各39 batch sequences/105 context tokens，总22次；无pixel/macro encode、AD、物理或训练。

owned直接调用未改loader的predict_high/rollout_high/get_cost。reference_high用冻结`old.reference_high`字典公式，不调用owned节点；v3 reference_rollout独立维护history和act_hist，每步当前packet reshape→append→stack→原context尾slice→字典high→只接本路末行。identity quantizer由source evidence认证，不monkeypatch。cost仍按width64展开4个width32宏步，最后terminal-goal的square.sum(dim=-1)，不是mean，未改goal。

四个实例hook观察owned macro输入、high输入/输出和proj输入/输出，不替换模型方法或旧global。保存所有22次完整z/actions/output/layout，另存owned condition/predictor/projector数组；reference独立滚本路history。四个完整输出必须float32字节一致，无tolerance/fallback。consumer重建append-stack动作值、stride/storage_offset/contiguity（第五步tail offset32）、本路history、native输出join、原输入recipe、完整数组inventory。legacy第一步stride160不得冒source第一步stride32。

cost保存reference terminal/expanded goal/residual/square/CUDA sum结果，与owned完整cost逐位相等；consumer复核逐元素公式和terminal join。CUDA sum由source绑定的实际调用和两路精确结果认证，**不伪称NumPy reduction重现CUDA sum字节**，不引入tolerance接受不同输出。

另独立核验sum(square)算术：对192个非负float32 square，用float64求和，float32标准unit roundoff u=2^-24、保守gamma192=192u/(1-192u)，加192×2^-149次正规项；同时计入float64审计求和的gamma192误差，上界向+inf舍入。该界仅验证原cost表达式，不放宽owned/reference逐位相等门。回归包含正常/零/次正规数与三个cost输出文件、SHA及comparison一致篡改但square未变的拒绝。

每次dispatch/high entry/return立即fsync journal；consumer重建全部8/22事件精确时序，拒绝未知/移位，不只验hash。错误保留四行分母、已付meters/ledger、pending tensor及首个cause，停止后续case、不重试；strict只接受全部完成。模型参数/state buffers前后逐值hash必须export-exact；425copy attempts/returns、491entries/531modules及loader source/device/IR逐项验收。

原caps：output32MiB/floor200MiB/wall600s（含run入口verify，边界watchdog非抢占）。journal额外留1MiB final reserve，不扩大总cap。torch import前CUBLAS_WORKSPACE_CONFIG=:4096:8，deterministic on、TF32/autocast off、CUDA-only；沿原SDP行为，不换kernel制造parity。

official_original_model_parity/official_parity/baseline_runnable/scientific_admission/legacy_v1_released全部false。仅owned高层子集与经认证source转写的四个固定输入；非官方类/完整pixel-lowlevel/planner或HiLeWM baseline运行准入。

## MAIN CLI

冻结后生产prepare。输出必须专属目录，old.write_new累计同目录全部文件，不可平写到results/根目录。

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
.venv/bin/python -m pytest -q tests/test_agent_hilewm_trusted_cuda_validation_20261028_v3.py
.venv/bin/python -m experiments.spectral_reachability.agent_hilewm_trusted_cuda_validation_20261028_v3 prepare --output results/agent_hilewm_trusted_cuda_validation_20261028_v3_prepare_MAIN01 --main-authorized
.venv/bin/python -m experiments.spectral_reachability.agent_hilewm_trusted_cuda_validation_20261028_v3 verify --manifest results/agent_hilewm_trusted_cuda_validation_20261028_v3_prepare_MAIN01/manifest.json --manifest-sha256 ACTUAL_MANIFEST_SHA
CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python -m experiments.spectral_reachability.agent_hilewm_trusted_cuda_validation_20261028_v3 run --manifest results/agent_hilewm_trusted_cuda_validation_20261028_v3_prepare_MAIN01/manifest.json --manifest-sha256 ACTUAL_MANIFEST_SHA --output results/agent_hilewm_trusted_cuda_validation_20261028_v3_cuda_MAIN01 --main-authorized
.venv/bin/python -m experiments.spectral_reachability.agent_hilewm_trusted_cuda_validation_20261028_v3 consume --manifest results/agent_hilewm_trusted_cuda_validation_20261028_v3_prepare_MAIN01/manifest.json --manifest-sha256 ACTUAL_MANIFEST_SHA --report results/agent_hilewm_trusted_cuda_validation_20261028_v3_cuda_MAIN01/report.json --report-sha256 ACTUAL_REPORT_SHA --receipt-output results/agent_hilewm_trusted_cuda_validation_20261028_v3_consume_MAIN01
```

ACTUAL_*替换为实际SHA，无未来SHA预填。consume可省略receipt-output只打印小receipt；指定时新建v3专属receipt目录，旧报告不写。

测试仅NumPy/saved metadata/显式非神经callback与temp目录，不是实际模型证据。真实CUDA与四用例strict仍待MAIN；不得因fixture一致宣称parity或科学通过。
