# 固定两臂 JEPA public numeric adapter v2

仅新增本stem source/tests/本文，四word v1和native v1全部继续冻结。MAIN占用GPU；worker只做RAM测试/已有公开数值与源码读取，不prepare、不加载模型/权重、不执行神经/physics/网络。experiment-code技能用于固定程序、预算、原值日志和失败回归，不用训练/拟合/adapter/teacher/repair。

## 协议

固定两臂 `push_x` → `raw_zero`，每臂原始float64[2,5,2]、raw10：前者十次[.1,0]，后者十次[0,0]。无候选选择、额外方向、调幅、padding、replay、筛根或失败补跑。该动作在本次模型执行前声明；不读private reference suffix、contact、block pose、source witness或新物理report来设计程序。几何/接触OOD校准不是fresh，不改变P1/P5/scientific admission。两臂保持完整失败分母。

唯一数值入口：MAIN实际发布的 `results/agent_p5_contact_source_native_20261028_v1_public_MAIN01_payload.json`，FILE SHA
`fdc063a6475577071fb74cf4ccd0853c4b2e4e5c061693d73efbc437686ffec3`。
恰好7NPY公域：RGB0/5/10、原生proprio[3,4]、raw prefix history[10,2]、normalized history[2,10]、独立goal RGB。public目录严格allowlist、不准夹私域文件/路径。goal只绑定，不编码/评分。程序不接受private lineage或strict receipt为模型输入；MAIN external public SHA是来源授权边界，本模块不签新source证书。MAIN部署应只挂载public+payload、本次manifest、模型与必要源码/依赖，隔离private sibling和source结果；Python allowlist不是OS沙箱。

共享真实RGB/proprio编码一次。两臂分别从相同visual/proprio/history的copy开始，每个5步packet追加第三包；visual预测末帧与patchwise proprio预测末帧共同append，history同步shift，不teacher force。实际预测raw float32输入/输出全保存，包括两次原始预测序列，不做latent到物理映射。仅float64计算终点patch `push_vs_raw_zero_patch_mse`；原5步与10步输出都在trace，可saved核算，但不把feature响应当作block运动/接触/任务成功。

## 原checkpoint—预处理配对证据

机器可读 `preprocessing_pairing` 写入manifest，verify重建相等；全部证据为已有source与saved model authority，不加载pickle：

|配对边界|可核验来源与实现|
|---|---|
|原checkpoint|`data/jepa_wm_checkpoints/jepa_wm_pusht.pth.tar`，SHA `9beca3eafe0739c3b3adb5d734fa435ccbda0fea8a65d53d4cccec176aaaa0eb`；原v8 CUDA loader严格匹配predictor/proprio shapes、keys、state receipt；原v12已保存before/after相同。不是新checkpoint或新独立架构。|
|配置|`external/jepa-wms/configs/vjepa_wm/pt_sweep/pt_4f_fsk5_ask1_r224_vjtranoaug_predAdaLN_ftprop_depth6_repro_2roll_save.yaml`，SHA `7a16e8edbf40b260aab303949e90c1879dd5f778d51f3b6d6cb4bea1d99969e2`；PushT、frameskip5/action_skip1、num_hist3、AdaLNpatch384/proprio16、224RGB。|
|训练原统计|`app/plan_common/datasets/pusht_dset.py:28`，SHA `30070c5a9f4770b84e3d6b6e5323df01d9c3405042cb988af07fcdcf43b6bf92`；AST只读四组literal，与pinned v12 stats精确相等。禁止import此dataset，因为原类会加载pickle。|
|动作|原dataset relative=True/action_scale100：dataset像素float32先/100后mean/std；本模块从raw64构建 `(100*u64).astype(float32)/float32(100)` 再用原float32 action stats；两次5×2按time-major展平成10。raw-zero不是normalized-zero。|
|proprio|原dataset agent位置+agent速度；`preprocessor.py:49`及`vit_enc_preds.py:391`先用原统计归一化，再embedding，不直接喂raw。`PROPRIO_MEAN=[236.6155,264.5674,-2.93032027,2.54307914]`；STD `[101.1202,87.0112,74.84556075,74.14009094]`。不是block速度或block labels。|
|RGB|原`vit_enc_preds.py:401`转float32后/255一次、native transform；ImageNet mean [.485,.456,.406]/std [.229,.224,.225]；no flip/autoaug/motion shift/erase。v2使用配置原scale=(1.777,1.777)、ratio1，而非v1/v12 scale1。|

原config的crop面积scale1.777在固定224方图下产生尺寸299>224，native `_get_param_spatial_crop`十次均无合法crop，fallback whole-image `(0,0,224,224)`，再bilinear/align_corners=False到224。旧v12 scale1直接得到同一空间crop。**空间结果相同不表示RNG消耗或参数相同**；v2实际使用原config1.777，不伪称v1参数完全相等。RAM只从绑定源码AST抽出这个纯坐标helper，核对两个crop均为全图，不执行torch/图像神经算子。

`preprocessor.py` SHA `b0454c593327230dd29a02fe1de65a54cefecc60d98d59c8830563821cc9574f`；`datasets/transforms.py` SHA `807446c6f944c2c771fdf8269dfac0ecab4a7fc7764e39d80b739eb0bdb13dc5`；`src/datasets/utils/video/transforms.py` SHA `61fd92a11e788e2b34e2d121ee45376044802544a076ecdc7c571e834cccdf3c`。原train.py:312传cfg augmentation，:385将dataset stats和transform共同给Preprocessor；SHA `c1fc4c57b99cab18df14405236adf463363fa4eef23705635a2a48e5d6e98285`。这些源/配置均已在原模型闭包，v2另单值pin并验原manifest统计配对。

视觉encoder保持原官方SOURCE+cached-HF WEIGHTS严格映射，不新增“官方checkpoint来源认证”。v12 manifest/report文件SHA分别 `41ef9bf8f5c5ae5c4ff8890a25e6a83137b7e637058a8033281a5eabc9cea07f` / `d8667a4e8537a5e005d13507d4d05b8d7907996549889decc8a00e1db873eda6`，只用于旧模型authority，不用于本次动作筛选。模型投影单值SHA `a4757f79a4a3e699e239892bdad0273f162a6f38ef61c3df96c0e9f7a9188cb5`，246seals仅源码/YAML/helpers/两权重，无旧factual arrays/视频/results；verify/CUDA无需重读原报告。

## 预算与输出

|轴|API attempts/returns上限|forward samples|context frames|
|---|---:|---:|---:|
|共享RGB encoder，batch3|1|3|3|
|共享proprio，batch1序列|1|1|3|
|两臂各2次predictor，batch1|4|4|12|

scope包含全部context编码、predictor及失败attempt。prehook先扣预算，真实posthook记return；失败不退款，pending不允许再次dispatch；无额外warmup/replay。共享项记一次，不重复退费。AD/backward/JVP/VJP/SVD/physics=0，walltime独立实测，FLOPs=null，不相加samples/API，更不称等总算力。

64MiB output、1MiB terminal reserve、200MiB free floor、600s wall+SIGALRM、4GiB预加载CUDA free工程护栏（非峰值显存保证）。所有输出exclusive create、original arrays SHA/metadata、63以内实际事件；共享3数组+两臂各2packet×5数组=23NPY。保存全两分母/partial/errors/actual attempts-returns，finally清理owned hooks、核模型state/来源未变。严格saved consumer核事件链、操作顺序、所有原值递推、统计配对、两臂分母、资源轴和完整inventory，partial不得通过，缺失不填0；原report/新receipt FILE SHA由MAIN实际提供。SIGALRM可能被CUDA/C延迟，超wall严格拒绝；磁盘终态失败只保留partial。

## MAIN精确入口

cwd `/home/gbk/code/LeWorldModel/le-wm-main`；此刻不运行神经，先审读/测试。

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p5_contact_action_response_20261028_v2.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p5_contact_action_response_20261028_v2 prepare --payload results/agent_p5_contact_source_native_20261028_v1_public_MAIN01_payload.json --payload-sha256 fdc063a6475577071fb74cf4ccd0853c4b2e4e5c061693d73efbc437686ffec3 --manifest results/agent_p5_contact_action_response_20261028_v2_prepare_MAIN01.json
sha256sum results/agent_p5_contact_action_response_20261028_v2_prepare_MAIN01.json
```

GPU空闲且MAIN审查批准后才执行；不能把占位符当未来SHA：

```bash
env -u PYTHONPATH HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p5_contact_action_response_20261028_v2 cuda --manifest results/agent_p5_contact_action_response_20261028_v2_prepare_MAIN01.json --manifest-sha256 <ACTUAL_PREPARE_SHA> --confirm-host-cuda --output results/agent_p5_contact_action_response_20261028_v2_cuda_MAIN01
sha256sum results/agent_p5_contact_action_response_20261028_v2_cuda_MAIN01/report.json
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p5_contact_action_response_20261028_v2 consume --manifest results/agent_p5_contact_action_response_20261028_v2_prepare_MAIN01.json --manifest-sha256 <ACTUAL_PREPARE_SHA> --report results/agent_p5_contact_action_response_20261028_v2_cuda_MAIN01/report.json --report-sha256 <ACTUAL_REPORT_SHA> --output results/agent_p5_contact_action_response_20261028_v2_consume_MAIN01.json
```

worker未对真实资产prepare/verify/CUDA/consume，不写MAIN results。RAM58项（全两臂hook成功/失败cleanup、重新封hash的feedback mutation、原生preprocessing源码回归与公开数值ABI），非新科学证据。最终测试结果与三文件SHA见冻结回复。所有scientific/P5/action_witness/physical_release仍false，task_success=null；物理paired efficacy、新任务成功、freshness尚未建立，不从先前contact source的通过推导新模型能力。
