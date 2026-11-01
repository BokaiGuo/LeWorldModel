# Hi/P2 跨计算 parity：固定 known2 RGB 与真实 history3

仅新source/tests/本文；不改任何封存源码/manifest/结果。worker不执行prepare、权重加载、fullstrict、NN/physics；MAIN审阅后主机CUDA执行。不是重复已完成 single-z/zero-history probe，不是prior三臂实验或raw发布。

## 固定输入和真正新检查

固定原P2 three_program prepare中的 root0/off、root1/off 两根，不挑新根；每根原blind包 actual observed0/5/10+goal 共4图，按原顺序各路径batch1编码，共16 encoder+16 projector。逐图同byte RGB；P2使用原 `preflight_pusht_d2p_p0._image_tensor/image_transform` 和实际 `model.encode`；Hi使用冻结 `rgb_high_low.preprocess/owned_pixel`。保存preprocessed张量、patch卷积、12层输出、finalnorm、完整257token、CLS/projector，P2仅观察hook不替换forward，finally清理hooks。Hi projector仍使用原inference-only节点，此像素输出仅诊断，不混入低层梯度输入。

低层另从原task读取真实已保存 `[3,192]` latent history + `[2,10]` between-state normalized动作。真实warm10/原blind/task/context/图像SHA精确join，三route都使用**同一P2记录输入**，不是分别编码后各自不同输入来掩盖配对。它们是P2工程坐标，不认证Hi原scaler；既不再次normalize也不借用P2常数作raw inverse。原stats未知，raw部署永远false。

固定future=`float32((arange30%11)-5)/32`，B1/S1/H3，原probe相同固定数值但这次为真实history3而非single-z。固定weights=`float32((arange192%13)-6)/128`，float64末帧加权sum，完整30维scalar VJP。三route：

- `p2`：原实际 `model.action_encoder`，原 `model.predict` 内原predictor→flatten pred_proj→reshape；保留原 `make_transition` rolling3 latent/rolling2 actions。只有去掉raw normalization，接受显式normalized width10，不换模型数学。观察predictor/proj hooks不替换任何方法。
- `hi_whole`：冻结可微facade v2 + normalized_transition全H3，完整history累计。
- `hi_segmented`：同facade/seam分段1/1/1，完整history保留。不因P2滚动而裁剪Hi stored history；比较时显式取causal对应槽。

两根共18 AE/18 low_predictor/18 projection；6 P2 predict API、0 Hi root predict API；6明确scalar VJP。每步predictor3槽，projection3行，总context tokens54/projection rows54；AE tokens P2=18、Hi=48，共66。AD primal已经包含在18，不再相加；内部engine backward/FLOPs未知。无新high、physics、训练、Qfit、raw或策略选择。不是相同总算力实验。

## 认证来源与元数据缺口修复

绑定实际catalog report SHA `5605d9d7966a38ac146a84f987a09099dae33da5a2864110a69da06fd1970650`、完整receipt SHA `9a5406bd076ccfd8a931d41943a64a1f905fa07a6a42711d419db923bcd7abd7`、join SHA `1523183330336bd3e66884a15fa96e7a15161abd8a343d1f50dba7a0427494b8`。prepare重建catalog saved消费、五组join全字段及整个Hi exporter实际receipt，不只status；五组目录全等且无layout differences为硬前置，但不当成计算已相等。

复用冻结 P2 loaded_model 原同buffer/config/weightsSHA/aggregate hash及完整目录守卫；Hi construct_cuda仍来自原data-only NPY/IR与真实verify receipt，全部425值前后精确hash。**本版prepare显式将loader source/tests/doc全部纳入sources**，run核construction.own_sha256均已事前绑定，不重演旧probe仅code入manifest的问题。frozen facade v2与seam源码固定SHA。runtime使用原venv；已用P2 runtime来源从实际catalog绑定，torch functional/autograd/attention、原transform/Hi配置/官方文本/IR以及本三件套事前绑定。原shader/二进制硬件真实性仍不是该manifest认证范围。

输入仅原已知task/blind/model资产；不读取新物理outcome，不对词/根/region做结果选择。prepare会读取已认证Hi NPY来验证SHA（无模型执行），worker本轮不运行该生产链。

## 数值判据、保存与saved消费

完整shape/dtype/C-index bytes比较，无ULP或相对容差。分别输出preprocess、patch、12layer、finalnorm、encoder、projector、低层三槽、三个cp、末预测、scalar和30梯度。AE只将末3输入相关输出作为跨滚动/fullhistory的规范化比较；原完整AE与histories仍保存。布局stride/offset/contiguity/device/requires_grad完整记录且另列，不把不同layout当同kernel保证。

数值不同不是停止捕获理由：只要接口可执行，所有root/path/slot继续，diagnostics.all_values_exact=false，保留差异。不为追求通过调整精度或容差。MATH-only/deterministic/TF32off/autocastoff/ordinary enabled grad，逐事件核验记录。saved scalar独立float64 fsum与原tensor sum只允许预先明确的192*eps*sumabs舍入界；这是算术复算边界，**跨route scalar仍逐位要求**，不是parity容差。

saved consumer不重新forward/VJP：核全文件/NPY SHA、全部已保存tensor layout、每步实际action append、AE到predictor last3、predictor→flatten→proj、cp完整rolling/fullhistory recurrence、prediction/scalar/VJP原输入输出、全模块分母与实际attempt/return。它是已保存接线复核，不冒独立梯度或全网络方程证据。原全12layer forward来自P2实际runtime vs Hi owned方程，不能把Hi/Hi同方程对照再说独立全网络验证。

完整成功捕获固定786个NPY，不去重：每根4图，每图1个raw RGB加两路径各34数组，共276；低层P2 45、Hi whole 36、Hi segmented 36，共117；每根393，两根786。consumer要求全数组引用与文件清单闭合。新编码observed槽与原paid latent history另列 `paid_history_checks`，不混作跨runtime同输入parity的通过条件，原历史与新编码的实际差异仍保留。

256MiB输出、8MiB final reserve、200MiB floor、900s cooperative whole wall（包括初末verify/load/hash/CPUcapture）。wall不能抢占kernel；nested callback/sync/capture wall重叠，不相加成GPU总时。任何异常保留partial arrays、所有root/route not_started和失败report，不padding/retry/选幸存根。进程强杀/磁盘损坏不保证finally产物。

## MAIN CLI（worker未执行）

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_agent_p2_hilewm_computation_parity_20261029_v1.py
.venv/bin/python -m experiments.spectral_reachability.agent_p2_hilewm_computation_parity_20261029_v1 prepare --output results/agent_p2_hilewm_computation_parity_20261029_v1_prepare_MAIN01.json
# 用真实prepare输出赋值 HILEWM_PARITY_MH
.venv/bin/python -m experiments.spectral_reachability.agent_p2_hilewm_computation_parity_20261029_v1 verify --manifest results/agent_p2_hilewm_computation_parity_20261029_v1_prepare_MAIN01.json --manifest-sha256 "$HILEWM_PARITY_MH"
CUBLAS_WORKSPACE_CONFIG=:4096:8 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.spectral_reachability.agent_p2_hilewm_computation_parity_20261029_v1 run --manifest results/agent_p2_hilewm_computation_parity_20261029_v1_prepare_MAIN01.json --manifest-sha256 "$HILEWM_PARITY_MH" --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_hilewm_computation_parity_20261029_v1_cuda_MAIN01 --main-authorized
# 用实际report输出赋值 HILEWM_PARITY_RH
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.spectral_reachability.agent_p2_hilewm_computation_parity_20261029_v1 consume --report results/agent_p2_hilewm_computation_parity_20261029_v1_cuda_MAIN01/report.json --report-sha256 "$HILEWM_PARITY_RH" --output results/agent_p2_hilewm_computation_parity_20261029_v1_consume_MAIN01.json
```

输出路径均须不存在。RAM fixture故意生成零输出/零梯度的合成trace来验证接线，不是神经结果。已有single-z已完成数值不在此重跑或冒充history3证据；本版真实数值与计算配对尚待MAIN。未知原scaler、raw部署/P1科学准入和独立fresh确认仍未闭合，不以这次parity自动放行。

## 本地交付验证

专用RAM测试35 passed（2.31s）：含两根786数组完整saved消费fixture、动作/时序/梯度/精度/构造来源篡改、原known2保存输入ABI及hook清理。fixture仅模拟算术和保存链，不执行torch神经、权重加载、生产prepare或fullstrict。MAIN后续真实CUDA才检验跨runtime数值，不以RAM通过宣称计算配对完成。
