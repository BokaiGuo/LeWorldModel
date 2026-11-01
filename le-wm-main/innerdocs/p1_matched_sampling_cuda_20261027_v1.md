# P1固定小预算sampling对照（预声明、非准入）

本轮只实现/mock测试/prepare。MAIN可显式授权HOST CUDA；不执行物理、不训练、不下载。
这只是三组实验所需强对照中的一个局部预算点，不是论文默认protocol或SOTA复现。
无独立阈值/确认，也没有在这里建立direct stall；P1_admission/model_task_eligible始终false。

## 输入与隔离

仅复用actual direct v2的两个任务、来源、native encoding receipts、模型/normalizer及计账元数据。
调用封存witness模块的collect（不调用其producer/verify、不读witness产物），重验原始source census、
真实task_id映射、RGB→native192、goal与padding incoming tokens，identity Q及原模型hash。
上游aggregate report仅供source/model/encoding完成性与继承资源，不读取direct方法轨迹/损失。
源是calibration_exploration，重复RGB/zero raw causal padding，不是observed transition history或fresh confirmation。
根按root_sha256升序固定；无根替换、无按收益筛选。来源/编码/旧direct成本单列inherited，并非新执行免费。

## 冻结算法

两臂cem/iid_random；每root各6批×10条完整H2。唯一zero占batch0/slot0；之后59条真正随机非零样本，
首批仅抽9条；不抽后丢弃、不在后批插入mean、不去重、不padding/隐藏重试。
原full context3、native192 goal、identity Q terminal MSE；无反馈、无物理state输入。
两个10维packet独立logit球cap=.75sqrt(5)，先径向投影并float32 inward round，再NumPy float32 tanh→raw[-1,1]。

CEM：每维初始mean0、sigma.5；采独立对角高斯后投影；每批按(cost,slot)稳定选elite3，
对实际执行的feasible logits取经验均值/总体标准差ddof0，sigma floor .05，无平滑、无softmax。
共6次更新（最后一次只留receipt，不再采样）。沿用本仓
`p2_full20_sampling_20261027_v2.py`的elite经验更新语义，不直接调用其旧40候选raw-box runner。
新球域、elite3、zero仅首槽是明确协议差异。官方stable_worldmodel CEMSolver使用torch std默认ddof1、
每代插mean；config/eval/solver/cem.yaml为300 samples/30steps/topk30；这里不声称默认配置等价。

iid_random：每packet独立N(0,I)方向归一化，半径cap*U^(1/10)，球内均匀体积；不自适应。
PCG64，seed=20261027+1000*root_index+method_offset，cem offset0、random100。
这四个seed、sigma、更新与域在读新损失之前固定；没有阈值调参。不是MPPI。
全60条非线性sameQ选优，严格小于才替换incumbent，exact tie保留首槽。

## 资源与验证

每臂每根sampling=60×H2=120 predictor及action_encoder真实module entries。
赢家另付全新nominal H2、whole H2、segmented H2=6；要求完整contexts/commands/cost逐值相同。
合126/126每臂每根，两根两方法504/504。每个candidate都从同root执行完整H2，不共享prefix。
新AD/J/VJP/SVD/solve/encoder/projector/physics全部0；generation6、selection60，CEM update6（random0）单列。
hook绑定model.predictor而非假设model.predict计数，_step逐次核对predictor/action_encoder恰各一次，
float32 incoming token在forward前与原NumPy normalizer exact核验，inference_mode禁止建AD图。
NativeCuda的image hooks拒绝image前向；meter对AD及image资源predispatch cap0。
与witness126仅forward调用匹配：witness的AD forward虽包含在其126内，另有反传/J/SVD/solve，
故不是总算力/FLOPs/耗时匹配。记录含I/O/校验/模型加载的wall time，不用其冒充公平速度结论。

## 持久化与磁盘

所有run文件共≤8MiB，普通写入预留512KiB给finalization；每次写前确保剩余≥200MiB+本次字节。
预检要求8MiB+200MiB，过不了则整预算拒绝，不降候选。prepare仅两个小JSON、source hash manifest，
不复制大closure；外部prepare SHA+protocol SHA绑定sources/tests/doc、输入、checkpoint及上游封存。
lossless gzip JSONL每事件Z_SYNC_FLUSH+fsync；attempt在module entry前durable，native每packet完整context即时保存。
candidate完整两contexts在selection之前写；每批预存全部logits/mean/sigma，所有elite更新/选择保留。
完成时写gzip footer；中断stream可用zlib.decompressobj(31)提取已落盘完整行，不能当complete。
报告候选摘要引用同arm journal完整arrays；winner保留完整独立replays。不缩精度、不删大数组冒充完成。
额度/磁盘/硬件错误留下partial，不重试；首错保留，final synchronize/model/source/journal错误分列。
真正介质I/O失败无法保证最终report落盘；已有fsync事件仍可核算attempt/complete，绝不能准入。

## 命令

从le-wm-main执行，所有output必须新路径：

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_p1_matched_sampling_cuda_20261027_v1.py
.venv/bin/python -m experiments.spectral_reachability.p1_matched_sampling_cuda_20261027_v1 prepare --direct-prepared results/p1_direct_cuda_calibration_main_20261027_v2_prepared01 --direct-prepare-sha 24eaf8b02873b3d93ebeeadbead9ca56c2caa079cea8c30b308357ddd8a3d2e6 --task-directory results/p1_direct_cuda_calibration_main_20261027_v2_cuda_main01 --output results/p1_matched_sampling_cuda_20261027_v1_prepared01
.venv/bin/python -m experiments.spectral_reachability.p1_matched_sampling_cuda_20261027_v1 verify --prepared results/p1_matched_sampling_cuda_20261027_v1_prepared01 --prepare-sha PREPARE_SHA
# MAIN only, after review and enough disk; never run by implementation agent:
.venv/bin/python -m experiments.spectral_reachability.p1_matched_sampling_cuda_20261027_v1 run --prepared results/p1_matched_sampling_cuda_20261027_v1_prepared01 --prepare-sha PREPARE_SHA --output results/p1_matched_sampling_cuda_20261027_v1_cuda_main01 --authorize-host-cuda
```

测试是NumPy/mock工程证据，不是actual CUDA表现；实际运行后还需独立saved-array/计账复核与物理后果，
本轮不产出上述科学证据。大预算/默认协议与总资源frontier仍属后续前置，不能用此单点替代。
