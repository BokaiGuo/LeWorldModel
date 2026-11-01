# P5 frozen native visual goal scorer（MAIN HOST CUDA）

本stem source/tests/doc独占新增。完整prepare/verify/cuda/consume，不是missing manifest。只消费public七NPY和已封存四word模型预测/strict；不接受本轮physical report/analysis、block pose、contact、coverage、source seed或private lineage，不为选词读取它们。没有训练、normalizer拟合、readout、teacher、repair、adapter、新环境、physics或网络入口。worker只RAM，不实际prepare/编码/评分/CUDA。

## 原生合法接口与配对证据

- 原`planning/gc_agent.py:set_goal`先调用`model.encode(goal_state,act=False)`，L2时构造`ReprTargetDistMPCObjective`。
- 原`planning/planning/objectives.py:ReprTargetDistMPCObjective.__call__` visual-only tensor路径计算(target−prediction).pow(2).mean(feature axes)，`sum_all_diffs=False`取终点。不是coverage预测器。
- 原`app/vjepa_wm/modelcustom/simu_env_planning/vit_enc_preds.py:EncPredWM.encode`对RGB `/255`、原preprocessor.transform、dino patch encoder输出；支持visual-only goal，不要求伪造goal proprio。
- 原训练config `pt_4f_fsk5_ask1_r224_vjtranoaug_predAdaLN_ftprop_depth6_repro_2roll_save.yaml`指定`enc_type:dino`、`batchify_video:true`、`normalize_reps:false`，因此不额外添加layernorm/投影/归一化。三项native source/config hash继承封存owner.NATIVE；gc_agent/objective另按实际源码SHA绑定。
- 本次goal使用与已封存public context完全相同的official dinov2_vits14、原缓存权重严格映射和v8 CUDA loader；既有来源限制仍是official SOURCE + cached HF WEIGHTS映射，不升级称独立官方checkpoint认证。输入public `goal_rgb`已经由MAIN认证为独立native goal-scene原字节；本模块不重跑goal scene，不读取private publication sibling签新源证书。
- 精确复用已封存v1 `cuda_case` 的make_transforms(img_size=224,random_horizontal_flip=False,random_resize_aspect_ratio=(1,1),random_resize_scale=(1,1),reprob=0,auto_augment=False,motion_shift=False)，uint8 BCHW→CUDA float32/255，SDPA MATH、loader禁TF32/eval/frozen。取得原`x_norm_patchtokens` `(1,256,384)`。不改384维patch空间，也不拼goal proprio。

本程序不调用原wrapper init_module（该路径可下载/加载非所需资产），只按已认证loader配对接口执行。goal input为完整原图，可能含agent nuisance；feature-goal代价不等于纯block目标代价，也不能解释成native任务收益。

## 事先固定评分、词序与封存门

固定四词`xy,yx,x_minus_x,zero_zero`，全部保留。原raw float64、relative100、原stats和双packet append不改变；从原report的各`<word>_1_visual_out` `(3,256,384)`只取最后时刻，不重新预测。

评分严格为`mean((terminal.astype(float64)-goal.astype(float64))**2)`，与既有word-contrast的float64 saved归约口径一致；这保持原visual L2代数，但**不宣称与原Torch float32 reduction逐位相同**，不调用CPU神经。全部四个finite score保留；min为选择规则，精确相等依固定词序取首项，不设epsilon，不依据物理outcome换L1/cos/阈值/tie规则。选择仅模型校准输出，不是物理dispatch许可。

prepare绝不编码goal或计算goal score。run执行唯一goal encoder API后先保存原float32 `goal_patch.npy`并fsync，再保存`goal_encoding.json`：绑定prepared manifest digest、public payload SHA、原goal RGB record、patch FILE SHA、模型before receipt和protocol digest；文件与目录fsync后写`goal_sealed`链事件，之后才打开terminal进行评分。评分函数必须读回精确seal及feature SHA才能运行。没有seal或编码失败，则四词仍保留not_scored/null，不能评分或补零。

最终严格consume重算原四词numeric strict并要求其与实际receipt完全相等；重验所有public、模型spec/source/权重字节及本三件套；逐步复算5个event链：attempted→returned→goal_artifact→goal_sealed→scored。完整验证native shape、artifact SHA/bytes、goal seal、模型before/after、imported source闭包、全部四score/argmin/tie、meter、wall和精确inventory。CUDA来源可信度来自MAIN审读runner及实际执行，不是saved validator能够独立证明从未伪造tensor；无CPU神经重算encoder。

## 资源、失败与科学边界

新增：official_encoder **1API/1 forward sample/1 frame**；proprio/predictor新增均0。加载原三模型但predictor/proprio任何forward都会被predispatch拒绝；不得warmup/replay/padding。pre-hook检查实际batch1及CUDA输入shape，attempt在dispatch前计费，post-hook记录return，失败不退款。finally清理全部owned hooks，重验原模型状态及最终新增import源码闭包。source/private physical目录不需要挂载；Python允许列表不是OS沙箱，部署隔离由MAIN负责。

原封存模型成本单列为encoder1API/3samples/3frames、proprio1API/1sample/3frames、predictor8API/8samples/24frames；shared context只计一次。复用预测不重新执行、不给四词重复退款。新旧meter分开，API与samples不相加；AD/backward/JVP/VJP/SVD/physics=0、FLOPs=null，wall独立实测，不称等总算力。

工程cap16MiB（保留1MiB终端）、disk floor200MiB、wall600s+SIGALRM、CUDA free scratch4GiB（非峰值保证）。wall从输出Store创建起覆盖模型加载、goal执行、评分与final verification；初始saved preflight在Store前，不伪称全流程墙钟。原goal feature393216bytes，完整工程回归含实际尺寸。写前guard/exclusive create，无自动重试/恢复/覆盖/删除。C/CUDA可能延迟信号。任何异常保留partial/四词分母/耗用及cleanup错误，strict不接受；磁盘故障可能无法写终端，此时不产生有效回执。

单已知构造OOD根，旧四词预测及物理outcomes已经存在，本新评分规则不能倒签成前瞻trial。即使选词与某旧物理收益吻合，也不能称C2成立；所有scientific_admission/C2_established/physical_release恒false、task_success=null。真正前瞻任务收益需MAIN另封新trial声明、比较策略及原生任务指标后执行；本程序不读取物理指标进行该比较。

## MAIN精确CLI

cwd `/home/gbk/code/LeWorldModel/le-wm-main`。所有输出路径必须不存在，实际新SHA由MAIN生成，不能预填。

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p5_frozen_goal_scoring_20261028_v1.py

OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p5_frozen_goal_scoring_20261028_v1 prepare \
 --prediction-manifest results/agent_p5_contact_action_response_20261028_v1_prepare_MAIN01.json \
 --prediction-manifest-sha256 572e2fded1c8927302c4ffe66921ce05e58f5954b43a9bc44bc022c19c6bbfb6 \
 --prediction-report results/agent_p5_contact_action_response_20261028_v1_cuda_MAIN01/report.json \
 --prediction-report-sha256 4fd032a4e5fb365f5a11eecbaba416c1ace7352e7afd092aa0adf6b972bded7f \
 --prediction-receipt results/agent_p5_contact_action_response_20261028_v1_consume_MAIN01.json \
 --prediction-receipt-sha256 247a8a6328053f34b8367f39dca63f81fb009ca2a41734db0f618d1ea6eab2c3 \
 --manifest results/agent_p5_frozen_goal_scoring_20261028_v1_prepare_MAIN01.json
sha256sum results/agent_p5_frozen_goal_scoring_20261028_v1_prepare_MAIN01.json
```

下面`ACTUAL_*`只允许填上述新产物实际FILE SHA：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p5_frozen_goal_scoring_20261028_v1 verify \
 --manifest results/agent_p5_frozen_goal_scoring_20261028_v1_prepare_MAIN01.json --manifest-sha256 ACTUAL_PREPARE_SHA

env -u PYTHONPATH HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p5_frozen_goal_scoring_20261028_v1 cuda \
 --manifest results/agent_p5_frozen_goal_scoring_20261028_v1_prepare_MAIN01.json --manifest-sha256 ACTUAL_PREPARE_SHA \
 --confirm-host-cuda --output results/agent_p5_frozen_goal_scoring_20261028_v1_cuda_MAIN01
sha256sum results/agent_p5_frozen_goal_scoring_20261028_v1_cuda_MAIN01/report.json

OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p5_frozen_goal_scoring_20261028_v1 consume \
 --manifest results/agent_p5_frozen_goal_scoring_20261028_v1_prepare_MAIN01.json --manifest-sha256 ACTUAL_PREPARE_SHA \
 --report results/agent_p5_frozen_goal_scoring_20261028_v1_cuda_MAIN01/report.json --report-sha256 ACTUAL_REPORT_SHA \
 --output results/agent_p5_frozen_goal_scoring_20261028_v1_consume_MAIN01.json
```

本次RAM测试含缩小维度token recorder、hook成功/失败清理、无seal禁止评分、完整saved producer布局及篡改、准确四分母、严格source/manifest/receipt接线、原形状尺寸回归；不是fakecounter科学证据。真实编码、评分和strict产物均待MAIN实际执行。使用experiment-code技能仅固定协议/日志/RAM回归，不生成图或额外实验，不按结果“保优”。
