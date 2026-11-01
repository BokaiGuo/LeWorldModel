# Six target controller：仅saved预测双距离比较

只新增本stem source/tests/doc，不改任何冻结runner/目标/成本/结果。固定实际MAIN01三元组：

| role | results路径 | SHA |
|---|---|---|
| manifest | agent_p2_six_target_controller_cuda_20261029_v1_prepare_MAIN01.json | cccfadc20911dc1be008a047c57bca7400272f6602deb435fb0341a0c8bbc8f9 |
| report | agent_p2_six_target_controller_cuda_20261029_v1_cuda_MAIN01/report.json | b81f12ee3602835bc14fd2e22d7e3644b8376f7ba8b0b265948bfb18429b6934 |
| full receipt | agent_p2_six_target_controller_cuda_20261029_v1_consume_MAIN01.json | 592062bded7d6eb1716594b21fac4b5ef6255cf93d5285c2361f0ec2686c0103 |

`build`在MAIN执行时重建runtime.build(manifest.upstream_authentication)并全等原manifest，调用冻结runtime.analyze_saved验证所有6条完整candidate/mean/replay/27轴及数组链，原消费完整附加字段一并重建并全等外部receipt。核completion/report/manifest精确SHA、inclusive wall、saved_consume_authentication及其原完整Nash receipt；不只status。直接复用纯saved函数，不调用consume/verify的新增writer，不NN、不创建新认证wrapper/proof。认证不冒称独立CUDA归约/梯度。

原task goal由manifest.authorities.arbitrary_prepare精确绑定的原prepare（SHA3aaffe495a3cfddda0d88403a39b1c937121942c64f79be5a574499579384ad0）slots[root].goal读取，并逐位核对原parity goal图编码NPY，即slots[root].task.coordinate_refs.task_goal → rows[root].pixels[3].paths.hi.output。同时检查source_arrays.z/a/goal、完整history3/真实between-action2、task_id/task_sha/blindSHA、425 state坐标digest、原task.context/goal配方与每根三臂相同历史。绝不取prior_FIRST/program postcondition充原goal，goal不再编码。

固定6slots×cp0/5/10/15=24行：cp0原observed latent_history末帧；cp5/10/15为同槽实际native returned mean whole-H3 prediction三槽。各cp同时绑定segmented[1,2]对应数组，完整z/a histories和final history均byte-exact；没有mean/whole/seg一致即拒绝。不拿elite/candidate/witness代mean。cp为模型预测动作步索引，不是已执行物理观测时间。

每行分栏：本槽tracking_target_mse64、原task_goal_mse64，两者各自相对cp0的减小量。float32输入先转float64，再subtract/square，以math.fsum归约/192，identity metric，不拟合Q、不重用producer已存cost。原保存tokens/目标及所有新计算均不回写。固定每根全部三对×四cp=24个pair，只给right-minus-left **共同task_goal** MSE差值；不同tracking target的loss差值为null，不排名、不输出winner。goal/source数据SHA与原array pointer完整保留。

仅完整6returned、72API/324samples和完整receipt允许build；missing/partial/schema错/未知原goal/回放不一致均failclosed，不写伪分数、不丢分母或padding。所有结果是saved **model prediction**，native_metrics=null，无物理比较/任务成功/新NN。known2不是fresh/独立确认；prior source_time=null、program来源cp5而控制预测H3、arbitrary时间语义unknown逐行披露。generation预算不同，不称equal-FLOPs/fairbaseline/time-matched，scientific gate及raw部署许可均false。

source/metadata单文件reader沿冻结128MiB/32MiB限；输出16MiB、floor200MiB，排他xb+fsync。summary保留整份strict receipt、27轴、来源和source/NPY hash闭包。`verify`全量重建同一summary。生产build/verify会重读上游saved链并复算，认证CPU/IO不是零成本；worker仅RAM/小ABI，从未调用实际build/verify/fullstrict或模型/NN/physics。

## MAIN命令

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p2_six_target_prediction_comparison_20261029_v1.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p2_six_target_prediction_comparison_20261029_v1 build \
 --manifest results/agent_p2_six_target_controller_cuda_20261029_v1_prepare_MAIN01.json --manifest-sha256 cccfadc20911dc1be008a047c57bca7400272f6602deb435fb0341a0c8bbc8f9 \
 --report results/agent_p2_six_target_controller_cuda_20261029_v1_cuda_MAIN01/report.json --report-sha256 b81f12ee3602835bc14fd2e22d7e3644b8376f7ba8b0b265948bfb18429b6934 \
 --receipt results/agent_p2_six_target_controller_cuda_20261029_v1_consume_MAIN01.json --receipt-sha256 592062bded7d6eb1716594b21fac4b5ef6255cf93d5285c2361f0ec2686c0103 \
 --output results/agent_p2_six_target_prediction_comparison_20261029_v1_MAIN01.json
# SH用上述实际返回的summary SHA。
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p2_six_target_prediction_comparison_20261029_v1 verify \
 --summary results/agent_p2_six_target_prediction_comparison_20261029_v1_MAIN01.json --summary-sha256 "$SH"
```

experiment-code技能仅用于固定检查与mutation回归，不采用训练/搜索/效果择优建议。未执行生产比较；由MAIN实际build/verify后才可报告具体数值。

## 冻结交付测试边界

上述指定venv命令实际46passed。包括RAM六槽24行/24对双距离、完整whole/seg checkpoint/history/mean逐字节门与signed-zero mutation、独立f64算术、真实MAIN三元组小metadata与原goal/history小NPY ABI、整份receipt重建控制流的纯fixture额外字段篡改拒绝、排他写入/verify/CLI及无torch冷导入。fixture不冒称HOST结果；真实元数据检查未调用runtime.build/analyze_saved/consume，未计算实际六槽比较。实际build、全链strict重建与verify均未运行，留MAIN执行；无新增NN/physics或部署许可。source/tests/doc此轮冻结后不再原地修改，后续修复须新版本。
