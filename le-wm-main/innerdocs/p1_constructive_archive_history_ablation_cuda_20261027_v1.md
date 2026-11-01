# 固定archive-history F/S/A/SA条件消融实验

NEW同stem源码/tests/本文；非审计器，无独立审计agent，MAIN负责审读和实际CUDA。
experiment-code技能的固定协议、持久日志和失败保留用于实现；不训练/拟合/selection、不跑CPU神经/物理、
不改任何sealed文件，不重复outgoing F/Z/M或overlay render干预。

## 设计与固定预算（实现前已声明）

仅使用offline v2既有四分散episode[0,6228,12456,18684]的原20个native192编码、原raw动作和已保存F预测。
W10的states=[z0,z5,z10]、historical action=[0:5,5:10]、outgoing=10:15、label=z15；
W15的states=[z5,z10,z15]、historical action=[5:10,10:15]、outgoing=15:20、label=z20。
两窗口teacher-forced，W15仍使用实际z15，不用W10预测。四条件固定顺序，无看结果换窗口：

| 条件 | 两旧latent | 两包historical raw action | 当前latent/outgoing |
| --- | --- | --- | --- |
| F | 原真实history | 原真实history | 原真实不变 |
| S | 均改为当前latent | 原真实history | 不变 |
| A | 原真实history | 均为raw零，再public normalize | 不变 |
| SA | 均改为当前latent | 均为raw零，再public normalize | 不变 |

A不是normalized-zero，不是mask action embedding；所有条件仍通过原action_encoder及predict/pred_proj。
同窗口四臂的当前native192/outgoing原raw及其normalized token逐位不变；原raw专家数据不clip/tanh。
public mean/std与trainingunknown状态保持原值，不重算H5统计，不重编码，不读取HDF5。

8 windows×4=32 H1；0 encoder/projector，32 action_encoder/model.predict/predictor.forward/pred_proj.forward。
predict API和predictor分别计账，不相加为64预测；各3 tokens，共96输入latent/action packet positions与96 pred_proj samples。
0 AD/grad/engine/reset/render/physics/optimization；CUDA-only，原MATH SDP、TF32关闭、batch1/float32/eval/no-grad。
600s内部、外部650s；allocated3GiB=3221225472/reserved4GiB=4294967296，监测peak而非声称allocator硬上限。
输出≤8MiB（含partial/journal/final），其中1MiB final reserve，200MiB磁盘floor；启动free≥208MiB。
prepare<256KiB，只封小绑定清单，不复制旧大protocol或weights。

## 已有源与exact F

actual v2 report：`results/p1_constructive_offline_fidelity_cuda_20261027_v2_cuda_main01/report.json`，SHA
`e2ef4ce025d52f62e1688b268d2dab03021c248c4c8c5a7e2bf54ae98352b5e9`（本地只读核对）。
原prepare SHA=`580dd9e35431423885041da76bbdbca3a033808dae7db40aabbaff25e1c34adb`；
原v2源码SHA=`f11b7ed5ae9bb7fb64150c8a9e8285d206f0c57cd255e1f2f547071ce3d94e12`。

prepare/verify只消费report与hash绑定的小文件：原20 latent/PNG、raw20×2、原8个F的history/raw/tokens及完整输出。
文件SHA、arrayentry SHA/dtype/shape、固定slot/window身份和原public/model/source/software绑定均检查；
原source closure和本NEW三文件均封存并run末尾重验，不调用旧v2 material/verify，不重新编码。
原模型tensor SHA=`4b12c01f5859f4b46bea478f3a8d5e2d3f516df456a622a3feec31bde7e33546`。
加载同checkpoint，host GPU型号/torch CUDA/torch version与原receipt一致；原native F数值forward直接复用且不改封存源。
S/A/SA只变输入数组，再走相同F forward，不使用原M输出mask路径。

新F的完整3×192 prediction、action embedding前后输出逐字节比对原F，last192亦与其原完整输出核对。
检查dtype/shape/raw bytes而非allclose，包含signed-zero差异；不因“误差很小”改称exact。
出现不一致先durable保存该次输入/输出和delta/exact项，再工程fail-closed，保留32分母/未开始项、非零退出，
不重试、不换模型/normalizer/numerics或挑GPU结果。MAIN决定后续，不能据mock声称实际F已复现。
原v2 disk minimum已知mock污染，不采用为本次telemetry；从run初始化即注入真实output.parent disk探针。

## 保存、评分和解释边界

32请求按episode→W10/W15→F/S/A/SA固定；保存history/raw/tokens、embedding前后、完整prediction及last native192，
各有file/arrayentry SHA。module进入/完成、source绑定、错误均durable gzip Z_SYNC_FLUSH+fsync，新增文件目录fsync。
labels和原F reference都不进入backend白名单；所有32输出保存且8个F exact后，评分层才join同一实际successor。
逐8窗口列EF/ES/EA/ESA、S−F/A−F/SA−F、interaction=ESA−ES−EA+EF及persistence MSE；无拟合和新科学阈值。
首错优先保留，close/末尾source/model检查错误单列；cap/显存/时限错误也保留attempted/completed和未开始项。

只有原F具对应实际历史轨迹；S/A/SA是输入表征/历史条件消融，不是新的合法物理轨迹或counterfactual successor truth。
不能把对同factual label的误差解释成三种新物理结果的准确率，不能单凭此确定live失败由context引起。
archive四ordinal lineage/training membership仍unknown；live两根真实重复history的发现是提出诊断的动机，
不是本实验已经测到的训练来源或因果效应。不宣称P1通过、独立确认、success、non-greedy或真实能力改变。

## MAIN命令

从`/home/gbk/code/LeWorldModel/le-wm-main`执行，必须NEW输出：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_constructive_archive_history_ablation_cuda_20261027_v1.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_archive_history_ablation_cuda_20261027_v1 prepare --source-report results/p1_constructive_offline_fidelity_cuda_20261027_v2_cuda_main01/report.json --source-sha e2ef4ce025d52f62e1688b268d2dab03021c248c4c8c5a7e2bf54ae98352b5e9 --output results/p1_constructive_archive_history_ablation_cuda_20261027_v1_prepare01.json
```

MAIN检查prepare打印SHA，将`PREPARE_SHA`替换为其实际64位值：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_archive_history_ablation_cuda_20261027_v1 verify --prepared results/p1_constructive_archive_history_ablation_cuda_20261027_v1_prepare01.json --prepare-sha PREPARE_SHA
timeout 650s env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_archive_history_ablation_cuda_20261027_v1 run --prepared results/p1_constructive_archive_history_ablation_cuda_20261027_v1_prepare01.json --prepare-sha PREPARE_SHA --output results/p1_constructive_archive_history_ablation_cuda_20261027_v1_cuda_main01
```

本实现阶段仅RAM mocks＋saved RAM构建验证，不prepare、不跑GPU/CPU神经/物理。MAIN正式运行之前没有新科学结果。

交付验证：21个RAM tests通过；实际saved source的`material()`在RAM成功重建32 requests、8 labels、8组原F references，
222个绑定、canonical prepare大小44274 bytes，torch未导入。仅验证接口/源数组可消费，不代表新CUDA F已exact复现。
