# P1 sampling实际物理RGB：12次冻结编码producer

独立新producer，不改变24RGB MAIN driver或封存sampling/physical源。实现沿用
p1_witness_rgb_cuda_main_20261027_v1 的 load_compatible_pretrained、CUDA0、TF32=false、
eval/frozen、模型前后tensor_hash、p0.image_transform 与 screen.native_image_encoder。
不更换Q，不训练，不搜索、不重选物理动作；实际CUDA仅由MAIN运行。

## 固定原始来源与小manifest

- report：results/p1_sampling_physical_20261027_v1_main01/report.json
- report SHA256：c3eef328bee57ce5d42b25fcf5a9935af2d2282c9de3f8caf46f2b35b53a2ef6
- 物理prepare：results/p1_sampling_physical_20261027_v1_prepared01/prepare.json
- prepare SHA256：450e620dbe3bf1ec1957c8d2d06fda9dfc05c7cc6bd50453939dc07a2133033c

严格原source slot顺序、每根cem/iid_random、每臂step0/5/10，共4臂12PNG；相同像素也不去重。
源report必须完整、8reset/44render/40step；原journal完整SHA/hash chain与终态、branch_closed、
branch_started原plan、checkpoint_saved、十步实际commands/termination全部核对。
每个plan.prediction严格等于原sampling arm report的独立winner replay，并经原task的check_prediction。
原sampling prepare/模型/goal/Q/context/normalizer/所有原来源绑定经physical.verify重建。
12PNG逐个验证file SHA、RGB字节SHA、尺寸/模式和物理state chronology。

prepare为单个hashmanifest JSON：保留12个预测native192向量、物理诊断和原task引用，
不复制PNG、源码closure、原task的192×192 Q或模型权重。goal/Q/context用原task_id和显式组合SHA绑定；
run/analyze读取原sampling task，而非重新生成Q。manifest绑定全部已继承journal/PNG/model/source，
也绑定本producer/tests/doc与24RGB driver和其transform/模型源码。verify从当前实际资产完全重建。
仅保存本地hash，不自称official或训练沿革认证。

## 输出与失败边界

全run输出共享cap2MiB，磁盘至少200MiB reserve；普通写预留128KiB供final report。
启动需202MiB，prepare仅需实际manifest字节加200MiB；独占新路径，逐次write/flush/fsync，
不压缩、不丢字段、不改变float精度，不自动重试。没有pytest tmp或复制大closure。

image.events.jsonl：

- 每帧frame_started带原task_id，所有native事件加frame_order/root/arm/step/pixel SHA；
- 原helper的native_image_completed逐帧保存实际raw latent192与实际输入RGB SHA；
- CUDA synchronize后frame_completed再保存raw_native192；每条事件durable；
- 12encoder+12projector；predictor/action_encoder pre-hook拒绝任何调用；无predict/AD/physics；
- helper沿用24RGB语义：encoder/projector completed一起在整个encode返回并写receipt后确认，
  不能把故障路径的0 completed解释为某module没有实际return。失败只报partial，不作准入。

analysis.json：12行native_prediction_mse、原sameQ predicted_Q_mse/actual_Q_mse、
原physical_diagnostic；step0预测为原root latent，step5/10为原plan.prediction对应context末latent。
物理state仅用于诊断，绝不传给encoder。没有额外goal PNG编码，使用原task已冻结goal latent。

report.json：实际计数、模型前后hash、输入prepare与物理来源SHA、journal/analysis SHA、
partial错误与禁用模块尝试；成功status为all12_sampling_checkpoints_encoded。
写失败/模型漂移/source漂移/不完整计数均拒绝成功；OS/disk故障后不能承诺final report存在，
所有已有partial保留。run失败退出非零。不可把partial raw latent做完整对照。

## 命令

目录：/home/gbk/code/LeWorldModel/le-wm-main

    PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_sampling_rgb_cuda_20261027_v1.py
    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_sampling_rgb_cuda_20261027_v1 prepare --output results/p1_sampling_rgb_cuda_20261027_v1_prepared01.json
    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_sampling_rgb_cuda_20261027_v1 verify --prepared results/p1_sampling_rgb_cuda_20261027_v1_prepared01.json --prepare-sha PREPARE_SHA
    # MAIN ONLY, after source review and disk gate:
    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_sampling_rgb_cuda_20261027_v1 run --prepared results/p1_sampling_rgb_cuda_20261027_v1_prepared01.json --prepare-sha PREPARE_SHA --output results/p1_sampling_rgb_cuda_20261027_v1_cuda_main01

## 解释与准入

数组fixtures仅验证合同与公式，不是实际encoder数据。prepare验证的是现存实际source，
不是执行CUDA；actual latent/误差数值须等MAIN。sampling根仍是复用calibration根，
causal padding不升级为observed history，无独立确认、等计算量或科学P1准入。
contact=0或任何单项指标不构成理论失败判定。

本任务按experiment-code skill保留自包含JSON日志/实际资产测试；用户禁止训练/神经测试，
因此不采用其通用训练/作图流程。不改共享文档或旧结果。
