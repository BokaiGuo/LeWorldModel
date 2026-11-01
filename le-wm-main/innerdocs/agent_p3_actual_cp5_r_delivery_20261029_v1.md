# Actual-cp5 R：单 parent 模型首包 delivery

本独占 source/tests/doc 按 MAIN 事前协议 `main_p3_actual_cp5_R_delivery_protocol_20261029_v1.md` 实现，其 SHA 为 `deed88fd70073c1aaca96d32c81968b7f02a7a3841e3d8a7be00d4e4d573275f`。不修改旧 tree/group/bank、caps、协议或结果；不根据 control0 no_witness 或其它输出调参数。仅 MAIN 审读后可授权主机 CUDA。worker 仅 NumPy/RAM 与临时保存文件 ABI 测试，不 production prepare/run/fullstrict，不执行神经、物理、训练或下载。

## Actual-cp5 与来源闭包

固定输入 `/home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_actual_cp5_task_builder_20261029_v1_prepare_MAIN01.json`，SHA `ce6670d84f6f0f712bdfe5e1b7471be65b0441338c18010f79b3967ef28397c7`。`build()`不只接受状态：用冻结 Nash `resolve_actual_cp5_tasks(cp.authorities)`完整重建并比较整个输出，同时绑定 resolver/source/tests/doc、三个实际上游 authority、两原始 gzip task export、模型/loader/installed classes 与自身三件套。保留原 MAIN strict 的来源，不声称重跑上游 fullstrict。

输入固定两 known roots；同根原四 b5 分支一致性由 resolver 完整认证，无结果选枝。本版 root 是实际观察 absolute5/10/15，control−5/0/5，raw10=warmup尾5+真实prefix5。原 task 与 rebased local task 双身份保留；goal192、identity Q、normalizer/model/root/domain不换，旧 root context 不冒称 cp5。输入较老两 latent 必须精确对应原 root 后两帧；两 incoming token 按原 f32减/除一次且逐位核验。不新增 encoder/projector。

`material(prior,cp)`与`input_pair(task,provenance)`为 RAM 可测的纯 ABI 接缝，生产身份只能由 `build/verify` 的实际 resolver+字节/SHA闭包认证。它们不授予任意注入 fixture 为实际 observation 的权限。

## 固定四槽与执行

顺序 root0H1/root0H3/root1H1/root1H3；各槽独立 parent signal 与九项费用，不跨槽复用。shared Ledger 接冻结 Session/NativeTree/evaluate_R；真正 HOST factory 只在授权 run 内构造本版 `NativeDelivery`，RAM 必须显式许可，生产 consume 无 RAM开关。原 raw-zero J 已含 normalizer，不二次缩放；bank 为冻结 spectral8，一次 float64 SVD、固定半径/方向/规范符号、float64 tanh→一次f32。H3是明确30维推广，不签原 proposal 等价。

完整 zero+8 edge，各自原生 postcontext H1 nominal/J、原 ridge 与 H1 bounded suffix 全付并保存。资格严格 immediate>zero && reach<zero，无 epsilon。`eligible_order`保持全部 eligible，按 `(immediate,candidate_index)` 排序取首项，不按 reach winner、不读物理标签。suffix 仅诊断，永不拼计划。

若无 eligible：`status=no_witness`、`delivery=null`、`model_delivery_verified=false`；两验证槽 not_requested，实际只付九项，不 fallback、不阻断其它固定槽。有 eligible：完整原 edge 独立 whole 与 segmented 各重放一次，完整 path 逐位（canonical包含负零）核对原候选。H1=5raw；H3 segmented 明确 **5+10**：只在第5步后重建完整context，然后连续预测余10步，不冒称旧逐5步重建为5+10。`segmented_packets()`是 HOST 实际使用且 RAM 可测的调度。每次原生调用保存实际 full IO/完整3latent/2token/raw/normalize/路径/AD/192VJP/J；不以 RAM 数值假冒模型正确性。

仅全部验证通过后创建 own-schema model delivery。实际只交首5raw；其余 H3 raw10显式 lookahead，suffix另属诊断。H1/H3共同首包终点control10（absolute20），预测终点分别control10/20（absolute20/30），不能称同预测终点或已实现控制收益。不是完整策略树、原等预算实验或P1/P3/C3准入。

## 精确费用、失败和保存

cap=`base.caps(H)+verification_cost(H)`，完整27轴，不改旧树 replay2。

| 每完整槽 | H1 | H3 |
| --- | ---: | ---: |
| 有 witness inclusive predict API/samples | 40 | 66 |
| 无 witness实际 inclusive predict | 38 | 60 |
| J / signal_compile | 10 | 10 |
| VJP / grad / raw callback（各） | 1920 | 1920 |
| array SVD / solve | 10 / 9 | 10 / 9 |
| program_compile / metric | 10 / 18 | 10 / 18 |
| 有 witness replay / native calls | 20 / 30 | 20 / 30 |
| 无 witness replay / native calls | 18 / 28 | 18 / 28 |

AE/predictor/pred_proj/sample/token/row 分轴按原语展开。parent signal后、bank之前，原子预检完整九项+双验证reserve，不运行有利子集；无witness reserve不伪造实际费用。AD primal子轴为H+9、J元素192×10×(H+9)；失败未闭合时null。未知 scalar backward/FLOPs保持null，不把预测次数称FLOPs。

native/runtime异常保留已付前缀、control+8所有槽及验证返回/失败/未请求状态，交付清空，不退款/重试，后续固定槽标blocked；四分母不删除。no_witness不是异常。partial report持久化但run非零；已知失败prefix可由RAM inert verifier核验，生产完整consume不签partial。

工程预算固定每槽512MiB、整作业2GiB含256MiB收尾reserve、floor200MiB、wall1800s含verify/加载/hash、allocated3GiB/reserved4GiB。继承排他Store、全journal hashchain/fsync、精确文件census、immutable parent signal capsule及真实SHA/数组索引。模型/source/installed class前后逐项hash、eval/frozen、MATH-only、matmul/cudnn TF32=false、CUDA/CPU autocast=false、实例hook恢复。无需全局monkeypatch，无CPU神经回退。

## Pasteur physical driver 稳定消费 ABI

Python模块 `experiments.spectral_reachability.agent_p3_actual_cp5_r_delivery_20261029_v1`。
`consume(output, report_sha, prepared, prepare_sha)`使用真实参数，返回自身`.consume` schema，状态 `actual_cp5_R_model_deliveries_verified_NOT_physical`。下游应完整调用并与实际持久 receipt相等，不仅信status；所有实际prepare/report/receipt SHA由MAIN运行后提供，不填未来值。

| 位置 | 真实语义 |
| --- | --- |
| prepare.tasks[root] / prepare.provenance[root] | rebased cp5 task；完整original_task、raw10/prefix5、4branch witnesses、export与时间身份 |
| report.rows[i].slot/root/horizon/status | 固定4槽身份与真实witness/无witness/失败状态；不是fixed/adaptive标签 |
| report.rows[i].result.R_groups[0] | 全zero+8、严格资格、诊断suffix/reach与原始context/native见证 |
| result.eligible_order | 所有eligible按immediate/index排序，不只reach winner |
| result.delivery | 无witness/失败为null；仅双验证通过才非空 |
| delivery.task / base_task 与各自task_sha256 | cp5 rebased与原始task双身份；模型输入仅task实际context/goal/Q/normalizer |
| delivery.provenance_sha256 / source_closure_sha256 | 精确绑定prepare对应provenance及完整sources |
| delivery.raw_edge / raw_edge_float32_sha256 | 完整5/15 raw；canonical字段名为 **raw_edge**，不是另造edge_raw |
| delivery.commit_raw5 / commit_float32_sha256 | 仅第一5步，控制6..10；下游只能提交这一包 |
| delivery.lookahead_raw / lookahead_float32_sha256 | H1空/H3余10raw；不是额外物理提交或诊断suffix |
| result.verification / delivery.verification | whole及segmented的status、segments、native call_id、完整path；H3为[15]与[5,10] |
| result.resource_receipt | 全27轴caps/attempted/completed/events；所有真实收费含双验证；不能只取理论cap冒实际值 |
| result.native_calls | 按call_id连接全部原生request/IO/返回/失败；whole/segmented为末两独立调用 |
| receipt.arms[i] | 显式slot/root/horizon/status/complete/model_delivery_verified、delivery及完整resources；固定顺序保留no_witness |

`model_delivery_verified=true`仅表示模型见证，不是执行许可。physical_release、execution_authorized、scientific_admission始终false。Pasteur新own-schema driver需独立reset/warm10/prefix5，并在control6前核原cp5 state/snapshot/三条件RGB和history身份；本模块不签runtime restore、不写其文件、不接物理结果选候选。下游缺delivery必须明确不dispatch并保留槽，不能借旧计划代替。

## MAIN CLI

cwd `/home/gbk/code/LeWorldModel/le-wm-main`；替换运行后实际SHA：

```bash
.venv/bin/python -m pytest -q tests/test_agent_p3_actual_cp5_r_delivery_20261029_v1.py
.venv/bin/python -m experiments.spectral_reachability.agent_p3_actual_cp5_r_delivery_20261029_v1 --help
.venv/bin/python -m experiments.spectral_reachability.agent_p3_actual_cp5_r_delivery_20261029_v1 prepare --output results/agent_p3_actual_cp5_r_delivery_20261029_v1_prepare_MAIN01.json
.venv/bin/python -m experiments.spectral_reachability.agent_p3_actual_cp5_r_delivery_20261029_v1 verify --prepared results/agent_p3_actual_cp5_r_delivery_20261029_v1_prepare_MAIN01.json --prepare-sha ACTUAL_PREPARE_SHA
.venv/bin/python -m experiments.spectral_reachability.agent_p3_actual_cp5_r_delivery_20261029_v1 run --prepared results/agent_p3_actual_cp5_r_delivery_20261029_v1_prepare_MAIN01.json --prepare-sha ACTUAL_PREPARE_SHA --output results/agent_p3_actual_cp5_r_delivery_20261029_v1_cuda_MAIN01 --authorization MAIN_APPROVED_HOST_CUDA
.venv/bin/python -m experiments.spectral_reachability.agent_p3_actual_cp5_r_delivery_20261029_v1 consume --prepared results/agent_p3_actual_cp5_r_delivery_20261029_v1_prepare_MAIN01.json --prepare-sha ACTUAL_PREPARE_SHA --output results/agent_p3_actual_cp5_r_delivery_20261029_v1_cuda_MAIN01 --report-sha ACTUAL_REPORT_SHA --receipt results/agent_p3_actual_cp5_r_delivery_20261029_v1_consume_MAIN01.json
```

Saved strict重建全部own prepare/cp5 resolver、SHA/census/hashchain/model/precision闭包，逐native IO/资源核验；保存返回值驱动inert bank/ridge/资格/排序/双验证/commit全部代数重建，同时逐项比较非native调度事件，拒未知/孤立/重复/漏项。不导入torch、不重跑神经，也不声称独立模型输出parity。实际测试结果与冻结三SHA由交付消息记录；冻结后不追加。实际模型delivery/物理结果未知，待MAIN验收运行。
