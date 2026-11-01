# P3 joint-prefix：17张真实保存RGB的独立CUDA派生编码

## 固定任务与非执行边界

消费真实physical report `results/p3_joint_horizon_goal_metric_physical_20261027_v1_main01/report.json`，SHA256 `6c3659e199a8151fee2fa2c020a0cfe63db00b691b947cdbeb762c3c18d6004a`。不消费旧P1/P4终点作为prefix，不改planner/consumer/旧结果，不重跑规划或物理。实现参照P4 terminal20的local checkpoint/native encoder→existing projector模式，但不导入或修改其runner。

这里只交付source、非神经tests、prepare。只有main审阅后显式`run`才加载原模型、执行HOST CUDA。prepare/verify/consume不import torch、model loader、环境或旧executor/exporter。没有CPU神经fallback、训练、AD、physics或predict入口；没有下载入口。

完整分母：四个既有calibration roots ×四臂，16分支的**全部step5**；唯一H3分支（root_index1、goal_patch_Q__uniform）另加step10，合计17 saved RGB。重复图像也分别编码和计费，不去重、不挑损害根。顺序来自原physical plans（原planner balanced arm order），分支内部step递增；协议封存每帧order/frame_id/root/arm/step/selection/goal-RGB身份。

原source绑定：physical prepare SHA `1cfc5eb2ddcfc563176a3cce51a0ef90b88e92d298b3cf35d415c2e420317daf`；planner report SHA `23eae72d18fa02e9997859311bef282fc4f5e3478790d846524a3fa6f6b70950`、planner prepare SHA `7886406284604cf17ba97671030ef32fde9d32f6f4c18b8293db0c2129654a47`。

## Prefix chronology：不盲信下标

逐一核对原selected plan hash、winner、whole replay和segmented replay完整context相等、实际raw命令与journal observation次序，以及root/model/goal/active Q绑定。逐段重建float32 `(raw-mean)/std` time-major10维tokens，验证每个segment action history的内容、layout、metadata及仍保留的latent顺序。再由`initial_history + 各segment最新latent`取末三项，与最终winner context逐值相等。

由此推导而非硬假设：H2最终context为`[initial_last,z_step5,z_step10]`，所以step5在history[1]；H3为`[z_step5,z_step10,z_step15]`，所以step5/10在history[0]/[1]。还要求每个prefix向量与对应原segmented context的最新latent逐值相等，不能用nominal、terminal或另一candidate替代。prepare保存此推导的history index、预测向量及segment context SHA，不运行任何模型验证其真实性。

实际图像必须来自同分支journal中唯一对应step的observation，PNG文件SHA与uint8 RGB像素SHA分别封存，尺寸/模式固定224×224 RGB PNG。goal及两Q从原physical协议`root_metrics`读取并与planner record逐值核对、规范哈希绑定；不复制重复Q矩阵到新协议。原weights/config流式SHA绑定，不能换checkpoint或重建Q。

prepare还验证原source seal的live/archive bytes，并新封存本runner/tests/doc、pure IO/source-seal、原loader/transform与相关模型/预处理实现。checkpoint72MB只流式读哈希，不复制；旧大report/protocol/PNG只绑定原路径/哈希，不复制。原normalizer训练来源仍属历史限制，不因新编码而升级其科学证据。

## 真实CUDA合同（main执行）

使用原本地LeWM加载器、frozen eval模型、同CUDA device、原`p0._image_tensor + image_transform`、`model.encode({'pixels':...})`。单帧输入`[1,1,3,224,224]`；实际encoder entry `[1,3,224,224]`，existing projector entry `[1,192]`；输出原始native float32 `[192]`。

encoder全局cap17、existing projector全局cap17，每帧各cap1，严格按封存帧和encoder→projector次序。计attempt在dispatch前，sync成功后才记completed；失败不退款、不自动重试、阻断后续帧并保留全部17槽。predict/model.predictor/action_encoder/pred_proj以及常见autograd/torch.func AD入口均被hook拒绝并记录blocked。模型加载/preflight不是额外encoder样本；新神经预算只允许上述34个module entries。

每帧先写`.attempt.json`与fsync journal intent；模块attempt/return/error各自durable。成功后立即保存`.native192.npy`（禁止pickle，原float32，不Q投影）、fsync文件/目录并算SHA，写`.latent.json`，再记endpoint_saved和`.result.json`；**本执行器不在NPY保存前做Q代价计算**。如NPY已落盘而后续hash/journal失败，原NPY不删除，receipt保留路径/错误/可取得的hash，不把它算完整成功。

先为最终receipt预留512KiB空间，再加载模型；原冻结prepare保持不写入，用新的同级`<prepared>.launch.json`独占claim一次运行。再运行同prepare会拒绝，不因换output路径获得重复预算。claim后setup失败也保留失败receipt，不能自动删claim重试。

hook恢复逐项独立尝试，restore错误不能覆盖execution首错；final model hash、final source核验、journal hash各自保留错误。模型未加载时`model_unchanged=None`，不伪称True；加载后只在前后tensor hash匹配且frozen检查通过时接受完整结果。最终receipt给出34个实际entry、attempt/completed、blocked、journal_errors、first_error、全部errors、全部17帧census、原source/model/goalQ绑定、前后model hash。全局integrity结果覆盖全部帧，不是为每帧额外跑模型或tensor哈希。

完整状态`all_17_prefix_latents_saved`；否则`partial_or_failed`。硬杀、磁盘硬件/最终write完全失效时只能依赖已durable的journal/NPY/每帧JSON，预留空间不能保证任意I/O灾难下都能写出新最终receipt。不会以保存故障解释为任务负结果。

prepare与run分别有4,900,000 bytes小输出上限；source closure预先限3.5MB、每个JSON与journal限512KiB、每个NPY约896bytes，总17个约15KB；不保存图片/权重/旧大协议副本。prepare/run及每帧dispatch前均要求至少200MiB保留量加完整4.9MB输出预算，不能为本任务耗穿200MiB余量。实际prepare大小交付时另给，执行输出只有main运行后才有实测大小。

## 独立saved-only numerical consumer

同模块提供独立`consume`命令，不经CUDA运行路径。必须给main实际receipt的外部SHA；重新核对source/prepare及chronology、17个NPY SHA/shape/dtype、每帧attempt/latent/result文件及34条module attempt/return完整journal，不仅检查receipt汇总。

用新真实prefix latent与原selected-path预测prefix，逐帧给出：两套原Q下的actual/predicted cost、同Q actual−predicted cost、两Q projected prediction MSE，以及native prediction MSE。只在同Q内相减，禁止跨Q代价相减；不从MSE推断接触因果，不把prefix与terminal混作同一步反馈。尚未有真实17帧编码时不生成数值报告，也不拿mock数值补缺。

consumer还纯saved复用原physical report已存在的16个真实terminal latent，逐分支给出H2 step5/10、H3 step5/10/15的完整误差检查点序列。原terminal单列inherited、绝不再编码。可报告首次保存的非零native误差边界（精确非零、无拟合阈值），但这不是实际误差起点：例如step5已有误差只能限定“到step5已经出现”，不能断言step5或某个接触子步才开始。这一时间序列是后续main实际执行后的主要交付，不只是再次泛化gate。

这些是原16条轨迹的新派生观测，不是17个新rollout或独立根；原planner1066 predict/action-encoder、原physical165selected+240warmup以及16terminal encoder/projector是继承工作，不重算也不称零成本。新增prefix encoder/projector17/17单列，不称总pipeline/FLOPs匹配。所有P admission与efficacy均false。

## 非神经验证与main命令

工作目录`/home/gbk/code/LeWorldModel/le-wm-main`。worker独占生成下列prepare；main不要并发prepare同一路径。真实prepare SHA以交付回复及`sha256sum`为准，不能用旧P4/P3physical prepare SHA替代。

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p3_joint_prefix_cuda_encoding_20261027_v1.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.p3_joint_prefix_cuda_encoding_20261027_v1 prepare --output results/p3_joint_prefix_cuda_encoding_20261027_v1_prepared01
sha256sum results/p3_joint_prefix_cuda_encoding_20261027_v1_prepared01/prepare.json
```

仅main review后执行以下命令，用此次真实prepare SHA替换占位符：

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.p3_joint_prefix_cuda_encoding_20261027_v1 verify --prepared results/p3_joint_prefix_cuda_encoding_20261027_v1_prepared01 --prepare-sha ACTUAL_PREFIX_PREPARE_SHA
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.p3_joint_prefix_cuda_encoding_20261027_v1 run --prepared results/p3_joint_prefix_cuda_encoding_20261027_v1_prepared01 --prepare-sha ACTUAL_PREFIX_PREPARE_SHA --output results/p3_joint_prefix_cuda_encoding_20261027_v1_cuda_main01
sha256sum results/p3_joint_prefix_cuda_encoding_20261027_v1_cuda_main01/receipt.json
```

真实编码完成后，main可纯saved执行（无新神经/物理）：

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.p3_joint_prefix_cuda_encoding_20261027_v1 consume --run-dir results/p3_joint_prefix_cuda_encoding_20261027_v1_cuda_main01 --receipt-sha ACTUAL_PREFIX_RECEIPT_SHA --output results/p3_joint_prefix_cuda_encoding_20261027_v1_numerical_main01
```

tests使用真实saved输入作census/chronology核验，以及显式fake device/module接口（不创建torch tensor或模型）验证34次cap/次序、CPU/shape/重试拒绝、async及journal错误首错保存、每帧NPY持久化、坏dtype、hook恢复失败、model未加载不能True、独占launch、source漂移及saved consumer篡改拒绝。fake输出仅临时测试目录，不是真实prefix证据。
