# Fixed local HDF5 frozen factual fidelity pilot

独占NEW stem；不改旧封存。固定episode indices [0,1,2,3]，每条只取pixels step[0,5,10,15,20]、actions[0:20]及episode_idx/step_idx[0:21]。任意len<21、越界、时间缺口、非finite/超出原raw动作域，整协议失败，无替补、裁剪、重排或选择窗口。prepare/verify不运行模型、不hash 46GB全HDF5；本agent不写prepare、不执行CUDA，MAIN独占执行。

## 设计与来源边界

实现开始时未找到Beauvoir forthcoming设计文档，当前会话没有跨线程通信工具；严格先采用用户明确合同，不伪称已通信。参考其已存在render/training与action stats provenance审计，说明publicnormalizer不是checkpoint认证training stats。若forthcoming设计揭示不同时间语义，应MAIN先审再冻结新协议，不修改已封存产物。

本地 `/home/gbk/.stable_worldmodel/datasets/pusht_expert_train.h5`，首次只读metadata核验：大小46,300,921,856 bytes；pixels uint8[N,224,224,3]、action float32[N,2]，N=2336736。固定前四episode len均109、offset0/109/218/327；本地slice actions在[-1,1]，不重新拟合任何统计。不能将文件名、stat或这四条视为官方revision/独立holdout/原checkpoint训练集认证。

每episode核前/后offset连续、offset+len==next_offset、全局行数一致、21条episode_idx恒等本index、step_idx逐个0..20。只读5个指定pixel索引与20个primitive action，记录精确全局/局部slice、4个packet边界及字节SHA/dtype/shape。file stat(dev/inode/size/mtime_ns/ctime_ns)读前后与verify必须一致。每张原RGB字节和确定PNG编码均SHA绑定；不resize或颜色转换作为落盘原图，encoder使用已封存p0 transform。

action time audit沿当前本地HDF5Dataset._load_slice：非action每5步取frame，action保留primitive再按Dataset.__getitem__ reshape。故packet[0:5]配frame0→5、[5:10]配5→10、[10:15]配10→15、[15:20]配15→20；indices与源码绑定可证存储/loader合同，**不能独立认证历史collector确实为pre-action记录**。本HDF5无collector revision，该因果lineage仍unknown，不将该假设静默升级成事实。

## 冻结模型与算术

同原LeWM cache/config/tensor SHA、native projected CLS192。public mean[-.0087,.0068]/std[.2019,.2002]，float32 `(raw-mean)/std`，不再/100、不fit。training_provenance_verified=false；未尝试用全H5 stats替换旧normalizer。数据像素可能含目标背景，与constructive target-free渲染协议不同，此项只诊断当前本地factual数据路径。

每root以observed z0,z5,z10与两个已执行normalized action packets构完整3state/2action context。固定顺序H1 factual a10:15、zero、sign-flipped -a10:15，均从同original observedroot10独立单步预测z15。actual z15仅事后mean-square fidelity评分，不送backend、不用于候选排序/选动作。然后全新factual H2 a10:20，独立计算两个packet到z20；不复用刚才factual H1结果，不用真实z15 teacher-force第二packet。actual z20也仅评分。另报持久不动z10→z15/z20的纯数组误差参考。

zero/sign-flipped对actual factual z15的误差仅为非因果control diagnostic，**没有对应counterfactual真值**；即使control更贴近记录目标，也不是该动作真实更好。无winner/selection。不同episode同数据来源不等于独立科学样本；前四条不可据结果替换。

资源：20encoder＋20projector（4×5，含所有历史与评分帧，不去重）；每root3H1+2H2packet=5predict/5actionencoder，共20/20；其中factualprefix确实付两次。每predict输入3 latent tokens，共60；每actionencoder输入3 packet tokens，共60，每token10坐标。0AD/J/SVD/solve/physics/training。目标帧encoding是资源开销但不泄露为预测输入。HOST CUDA only，SDP MATH、TF32 false、frozen eval、before/aftertensor及源重验。phase分别禁encoder/projector或predict/actionencoder；禁AD API。

## 输出与失败

8MiB全输出cap含partial，200MiB reserve，256KiB终态预留，独占新目录；gzip每事件syncflush/fsync，无失败重试。保存selected_payload_receipts、20PNG、20native NPY、4raw20x2 NPY、4诊断JSONgzip（所有候选commands与完整predictedcontexts）、journal/binding/report，全部artifact SHA。NPY的file SHA与raw array SHA分别保存，不能混用。report固定20 images及4episodes分母，未开始/失败部分不补齐；encoder/predict逐module attempt/完成后CUDA sync，失败不退款。完整status=`offline_fidelity_recorded_not_admitted`，只代表资源/封存完整，不是模型精度pass或P1/P2。

## MAIN命令

cwd `/home/gbk/code/LeWorldModel/le-wm-main`；agent只跑RAMtests/可只读material，不执行以下prepare/run。

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_constructive_offline_fidelity_cuda_20261027_v1.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_offline_fidelity_cuda_20261027_v1 prepare --output results/p1_constructive_offline_fidelity_cuda_20261027_v1_prepare01.json
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_offline_fidelity_cuda_20261027_v1 verify --prepared results/p1_constructive_offline_fidelity_cuda_20261027_v1_prepare01.json --prepare-sha PREPARE_SHA
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_offline_fidelity_cuda_20261027_v1 run --prepared results/p1_constructive_offline_fidelity_cuda_20261027_v1_prepare01.json --prepare-sha PREPARE_SHA --output results/p1_constructive_offline_fidelity_cuda_20261027_v1_cuda_main01
```

PREPARE_SHA须取MAIN实际prepare完整返回；不能沿用旧任务SHA。剩余前置：MAIN审阅/freeze与prepare，独占HOST CUDA；实际结果后做saved复算。任何误差大小无预设科学pass阈值，不事后调normalizer/窗口/动作使通过。

交付本地验证：25 RAM/数组tests覆盖边界/时间/动作域、五次forward含H2重复prefix、目标不进backend/选优、完整20次encoder/projector与NPY、失败partial分母、禁AD恢复、stat/SHA重建。实际HDF5仅只读material()已通过：4episodes、20frames、113bindings、prepare序列化33,886 bytes；未写prepare、未跑模型/物理。此为数据/工程合同验证，不是factual动态精度结果。
