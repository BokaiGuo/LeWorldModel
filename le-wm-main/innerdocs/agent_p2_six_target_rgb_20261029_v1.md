# 六槽真实 RGB / actual-token prediction / native 三层接线

本新stem实现 prepare/verify/MAIN-hostCUDA run/整份saved consume；另复用本轮独占新 `agent_p2_six_target_rgb_roster_20261029_v1` 纯RAM核心及其tests/doc。旧协议、native adapter、paid/controller、Hi pairing及所有结果不改；无训练/fit/新physics/新predict/goal编码。不重复已完成H3编码或Hi parity。

批准协议为 `agent_p2_six_target_physical_comparison_protocol_20261029_v1.md` SHA `6e45721309e9a1fe1107918c988c170daf65ebe032163d0beace73d2bcb118f0`；ABI附录与paid实际authority补充继续保留。experiment-code技能只用于固定循环/失败账/RAM mutation，不采用训练或效果择优建议。

## 真实来源与权限

路径相对 `/home/gbk/code/LeWorldModel/le-wm-main/`：

| authority | 实际路径 | SHA256 |
|---|---|---|
| native report | results/agent_p2_six_target_physical_20261029_v1_MAIN01/report.json | ddbb339f87e0f49b1d99d32ae4f41340d72e6c19f8199fe69e6d0261f78bca2b |
| native full receipt | results/agent_p2_six_target_physical_20261029_v1_consume_MAIN01.json | 7579693032046b2427a7d5b492a90131029fe87d39147d06fe49123a3cfead13 |
| native source | experiments/spectral_reachability/agent_p2_six_target_physical_20261029_v1.py | 30990cedd608578679e255b867b2e1a28f12724574049ba7f415de67b90e7a32 |
| paid report | results/agent_p2_six_target_paid_replay_20261029_v1_cuda_MAIN01/report.json | c3733a752d1e9b80929edf9726bc0eba2ad1c2ae39b6f05550652ff7b3b74dbd |
| paid full receipt | results/agent_p2_six_target_paid_replay_20261029_v1_consume_MAIN01.json | 6e5fa7c57721cd4765430c7379d501d19978bdb44eebe96dce94f7bc68d9a3db |

MAIN生产build调用原native.consume(output,reportSHA)，整份强等实际持久receipt，不只status；再经execution_binding取得真实prepare及其SHA，原physical.verify认证。本原生consumer内部已完整复算paid receipt；新模块仍核paid.verify→(manifest,domain)、实际report/receipt和native.planner_authority.receipt强等，从真实paid trace/arrays提取预测，不复制旧normalized job分数。source/全PNG/数组/model/export/runtime闭包原样继承，加新三件套、roster三件套、旧RGB/comparator、loader/preprocess与config/weights的来源；最后流式hash再核，weights不走小metadata cap。

这是saved认证而非重跑NN。prepare/verify/run前置/consume均可能多次重读完整上游saved链，CPU/IO非零；本版不加入缓存或跳检。worker未调用这些真实生产入口。900秒run预算包含verify及其全部重读，不能以准备耗时为由偷偷延长。

## 固定24位置与actual-token来源

slot0..5按两根各 prior_FIRST / program_first_postcondition / arbitrary；每slot cp0/5/10/15，index=4*slot+cp_index。原native receipt的rows[i].plan全等prepare.plans[i]，task/selection/raw120bytes、same root/history/goal等保留。rows[i].trajectory只取phase=plan、overlay_index=0/condition=off的真正frame；checkpoints须精确等完整对应PNG/state/snapshot。cp0没有own_response，使用真实plan0及原native goal；cp>0精确对应responses[cp−1]，不拿warm10总步号或terminal代cp。

paid trace `inputs.z/a/target`核原controller_payload及dataSHA；arithmetic必须mul/add/sub/div，add实际输出raw15与plan.commands/raw_bits/sealed_raw完全相同，包含signed zero，不执行新inverse/clip/tanh。actual_tokens=[1,1,3,10]进入whole和segmented[1,2]完整预测；逐checkpoint/fullfinal histories与actual tokens追加顺序逐byte核，whole/seg全部预测一致。cp5/10/15取prediction[0,0,j,:]，cp0取真实z[0,2,:]。不依赖当前array_41/71等偶然编号，均从trace指针及原descriptor读取。

原task goal由domain.controller_prepare.authorities.arbitrary_prepare→原parity goal图output NPY核验，调用已封comparison.resolve_goals，核原history3/action2/task/blind/model坐标及每根三臂共用context。goal不再编码，tracking target与原goal绝不混用。prepare封存所有24输入引用/目标/预测向量与原文件hash，之后encoder不能看到用于选择的native标签；模型只接受RGB。

真正native stop仍完整保留24位置/24pair；未达到的frame为missing且不编码、不padding，实际encoding计数按available计算。缺失位置即使已有独立模型预测也只作为明确的预测栏保留，实际编码/native与pair差值为null，不冒同时间物理观测。工程错误不能伪装missing：整份native clean strict及原row无engineering error是门。当前MAIN实际六槽full15，因此预期24可用，但以认证结果而非预计补数。

## 原encoder / projector HOST接线

精确复用已封旧RGB加载路径：P2 policy `/home/gbk/.stable_worldmodel/checkpoints/pusht/lewm`，config/weights由旧bridge.rgb.MODELS固定SHA、模型tensor hash由bridge.native.MODEL_TENSOR_SHA固定并与已封pairing catalog MODEL_HASH强等。Hi425原state仍从paid.expected_state完整保留并由全部原NPY/source bytes封存守卫；本次只构造P2 encoder模型，不虚称又构造/前向了Hi425。Hi/P2逐key目录和786数组实际parity已由上游认证，不重新forward充证据，也不认证未知Hi raw scaler。

MAIN run显式 `--main-authorized` 后才可进入verify/CUDA；实际allocation preflight，原load_compatible_pretrained(...cuda:0).eval().requires_grad_(False)。原image_transform/_image_tensor与纯saved preprocess逐byte相等；model.encode({'pixels':x})['emb']为[1,1,192]。全257hidden、CLS/projector输入输出、preprocessed、pixels/latent全部独立保存，不只末向量，不去重，即使同根cp0图相同也各encode一次。

新Trace仅安装可恢复的只读module hooks，不patch任何旧module/global方法；encoder→projector每scope各一次，action_encoder/predictor/pred_proj任何forward在prehook拒绝。所有代码位于inference_mode、autocastoff、MATH SDPA、TF32off、deterministic；不调用AD/新predict/physics，禁止CPU neural fallback。TorchDispatchMode每op前后检查wall/GPU峰值，hooks/transfer也检查；构造后核frozen CUDA eval与模型hash，结束模型hash强等。没有独立梯度或模型公式重算声明，consumer只是封存接线复核。

## saved三层与24pair

全部模块/输出先durable native_seal，再由saved consumer算诊断，不将分数反馈给任何目标/计划。独立 `mse64`采用原f32→f64 subtract/square→math.fsum/192：model_tracking_Q/model_task_Q、actual_tracking_Q/actual_task_Q、prediction_vs_encoded_MSE、signed model_minus_actual_task_Q。cp0真实新编码必须与原观测latent逐byte一致，否则consumer拒绝，保留失败，不改容差/重编码/校正。

native逐位置直接用原state/goal独立 `physical.independent.scalars` 算native7/reward、agent/block/joint XY、agent velocity、raw/native/wrapped angle及原success，保留原response；单位不混，不把agent惯性当block机制。每root三对×四cp=24pair固定right−left，只比较共同原taskgoal actual-Q及native分项，cross_target_tracking_loss_difference=null，无winner。目标质量/控制器误差/模型误差不能由这一个断点唯一归因；不称公平生成预算、equal FLOPs、fresh或P2科学机制。

## 资源/失败/文件级消费

24张全图时encoder24 API/24samples、projector24 API/24samples，AE/predict/AD/physics/goal_encode均0；expected_calls与事件独立复算。resources另列encoded/missing位置、persisted tensor captures，all_cpu_transfer_calls=null（模型hash和附加CPU拷贝不被该capture数量完整覆盖）。原controller整份receipt和27轴、paid整份36/36及模块账、native counts均原样另列，不把nested工作相加冒总算力。

256MiB包括32MiB末尾reserve，floor200MiB；metadata128MiB；GPU allocated3GiB/reserved4GiB。whole900秒起于verify之前，至report fsync/hash；completion单独绑定report SHA和inclusive时间，ack自身不计wall但计bytes。合作式guard不能抢占kernel；final失败footer允许越时写，但完整consumer拒绝。失败不退款、不重试、不选幸存槽；磁盘耗尽/被杀不保证完备footer。实际native/paid运行不包含在本RGB900scope内。

Store独占xb/fsync，新目录不覆盖；binding→全initial_roster→事件/arrays→native_seal→report→completion。artifact_order/hash inventory完整保存，consumer核最初/末尾边界、每一module attempt→inputs/output→return和image attempt→pixels/pre/latent→return的顺序、所有事件/模块、全部数组引用无额外文件，NPY/gzip/header/EOF/dtype/shape/file/dataSHA及预处理/CLS/projector byte桥。末次源/结果hash再核。新RGB full consume仍会重建prepare及上游strict，只由MAIN实际执行。

## MAIN可执行命令

三件套及roster三件套冻结后、MAIN完整审读/批准生产再运行；worker不启动以下生产命令。

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider \
 tests/test_agent_p2_six_target_rgb_roster_20261029_v1.py tests/test_agent_p2_six_target_rgb_20261029_v1.py
MOD=experiments.spectral_reachability.agent_p2_six_target_rgb_20261029_v1
P=results/agent_p2_six_target_rgb_20261029_v1_prepare_MAIN01.json
O=results/agent_p2_six_target_rgb_20261029_v1_cuda_MAIN01
.venv/bin/python -B -m "$MOD" prepare \
 --native-output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_six_target_physical_20261029_v1_MAIN01 \
 --native-report-sha ddbb339f87e0f49b1d99d32ae4f41340d72e6c19f8199fe69e6d0261f78bca2b \
 --native-receipt /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_six_target_physical_20261029_v1_consume_MAIN01.json \
 --native-receipt-sha 7579693032046b2427a7d5b492a90131029fe87d39147d06fe49123a3cfead13 \
 --output "$P"
# PH取上一命令实际SHA；RH取run实际返回sha256，不填未来hash。
.venv/bin/python -B -m "$MOD" verify --prepared "$P" --prepare-sha "$PH"
CUBLAS_WORKSPACE_CONFIG=:4096:8 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m "$MOD" run \
 --prepared "$P" --prepare-sha "$PH" --output "$O" --main-authorized
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m "$MOD" consume \
 --rgb-output "$O" --report-sha "$RH" \
 --output results/agent_p2_six_target_rgb_20261029_v1_consume_MAIN01.json
```

## 本轮实际验证与未运行

联合86passed/4.11s（专用RGB37项、roster49项）。包括24图/24pair、earlystop、完整Store→数组→consumer fixture与extra/inventory/order/resource/wall/receipt篡改；真实native/paid小metadata的schema/cp/SHA检查，不读取实际预测结果算新分数、不运行实际fullstrict。纯fixture encode是NumPy常量IO，不是模型替身证据。未运行真实build/verify/run/consume、权重/模型/physics或子代；由MAIN实际prepare检验整个上游兼容性，MAIN host CUDA才检验真实数值与cp0 gate。当前无已知ABI blocker，无未来结果或raw/science许可。
