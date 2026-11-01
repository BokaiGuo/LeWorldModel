# Hi-LeWM moving RGB → high CEM → FIRST waypoint → embedded low

仅本独占新source/tests/doc；旧 high_target_adapter_v1、loader、v3、real_high_CEM、moving source及全部结果不改。worker只做RAM/已保存metadata测试，不运行production prepare、CUDA、CPU神经、physics、下载、训练、拟合或pickle反序列化。MAIN审读后单进程HOST CUDA运行；本交付是可调用研究接缝，不是完整baseline或E1/P1准入。

## 实际接线与缺口

原loader v1的`root.get_cost`只支持high，pixel ViT和low根方法未实现。因此本版不是把旧toy/seam或旧RAM adapter改名：high仍直接调用未改`root.get_cost`；另增加**owned dense ViT方程**和明确自己的`OwnedLow.get_cost`，逐项转写原`get_cost_low/rollout_low/predict_low`支持本固定面板的子集。所有权重来自原425个export tensors，没有训练、新参数、随机初始化、原foreign类或checkpoint执行，没有monkeypatch任何root方法/globals。

原checkpoint config的`transformers_version=null`，原bytecode不具备认证；安装ViT的q_proj/k_proj布局与保存的query/key/value布局不同。本版按保存的旧layout映射至标准方程，**不宣称original-class/encoder parity**。MAIN需审本新增方程后才运行，既有v3四case不认证本新增pixel/low forward。源码记录并核对literal image224/patch14/192/12layers/3heads/intermediate768/eps1e-12/SDPA/零dropout/插值开关及实际参数形状；缺字段/不匹配拒绝，不猜config默认值。

ViT链：patch Conv2d14/stride14→flatten/transpose→CLS+原position257→12×[pre-LN→QKV→非因果SDPA→out linear→residual→pre-LN→GELU(approximate=none)MLP→residual]→final LN→CLS→原projector。224正方形、257与原位置表相同，原`interpolate_pos_encoding=True`走返回原表的分支，不进行新插值。eval dropout为0。新kernel仅调用可信torch functional和原owned权重节点；不调用冻结loader已拒绝的encoder.forward。

## 固定输入、原官方policy与当前共同controller分栏

固定原moving source report SHA `85ca1df3de910628bfd6c5bd1a0a482f61880a2ee76bfd404732a77dd8c7666a`的**全四条件**，顺序root0/off,on、root1/off,on。沿原`moving.packets/load_packet`受限blind NPZ验证全部history/goal pixel SHA及原raw10；新模型只见每包`rgb_history[-1]`（真实warmup10当前帧）和本条件goal_rgb。不打开fresh保留根；不读outcome挑根/条件。

| 接口 | 本版实际使用 | 不宣称 |
|---|---|---|
| RGB | 原`img_transform`顺序：uint8→float32乘1/255→ImageNet Normalize→Resize224；输入本来224，Resize为identity | 不沿用旧共同模型的latent/goal，非训练数据覆盖证明 |
| high | 单末帧z_init与重新编码goal；原high metadata，经HighBatchBridge保留后重附 | 非三帧history high，不假造官方macro bank |
| selected waypoint | solver mean actions截receding1，block2 reshape成2macros，原root.rollout_high→`pred[:,0,0,:]` | 不是末waypoint、不是final goal、不是最优候选重选 |
| low | SAME当前z_init.unsqueeze(1)、零`a_hist[1,1,10]`、精确FIRST `z_subgoal[1,192]`，原policy `_plan_low`合同 | 不把moving的三帧非零action history冒充官方单帧合同 |
| low候选 | 固定3×2×10：全0、全−0.25、全+0.25；**模型normalized grouped坐标** | 不做low CEM、不是可执行raw控制，不签共同controller/backbone等价 |

原官方eval在运行时对dataset action拟合StandardScaler；本版**不拟合**，也不借用其他backbone的normalizer。完整原raw inverse-normalizer与dataset provenance尚未认证，故不输出raw动作/不接physics。本版low面板直接处于模型坐标；原`a_hist=0`是policy字面值，不称原warmup动作等于0。

low严格按原时序：先向a_seq追加本步future grouped action→对完整a_seq做原Embedder的Conv1d1及MLP→ctx=min(3,z_seq长度,act_emb长度)→取末端z_ctx/a_ctx→原low_predictor→flatten/low_pred_proj→取最后预测并追加z_seq。两步分别action长度2/3、predict context1/2；不是把前一个动作错当当前incoming动作。所有low候选都付费保存，不选物理结果。

## 搜索、资源与saved认证

每slot独立installed CEM，B1/chunk1/samples8/iterations3/topk3/H2/block2/macro32→64，seed1729+slotindex，完整零init，无warmstart/actor。Box[−1,1]只是本工程面板bounds声明；**安装CEM不clamp**，本版不额外clamp、不称官方评测配置或bank。source固定这些数，不按结果改阈值/seed/根。

全四槽新增8encoder+8projector、12high cost（96候选rows）、4selected rollout；三high模块各56calls，high_predictor392 sample rows/972 context tokens，high projection972 flattened rows。4low cost/12候选rows；8action_encoder（24rows/60输入tokens）、8low_predictor（24rows/36tokens）、8low_pred_proj（36 flattened rows）。API、module、sample和token轴嵌套重叠，不相加为“总调用”或FLOPs。0macro encoding、AD、physics；FLOPs未知，非等资源/完整官方搜索。模型装载425 tensor copy不是predict。

prepare完整重建MAIN v3 strict `6151e93c40fc1c5b053b81f8ac73a8dca7644a8a037492686a50a2718e24ddc6`及real-high-CEM strict `7dfb4b49d552a6248da8b603e027a95c541c5db95af4fab777828ed412886e43`并与持久receipt整份相等，继承原export/IR权威；不重复运行合成CEM。绑定所有新旧source、安装CEM、torch functional/attention、ViT映射参考、原官方文本、图像统计/变换源码、输入NPZ和自身三件套，verify整份build重构，run前后再次验证及所有export权重/Buffer SHA相等。

只在MAIN显式`--main-authorized`后导入运行时；MATH-only context、deterministic/CUBLAS、matmul+cudnn TF32关闭、autocast/grad关闭，每个trace/hook及全局before/after持久核验math/flash/efficient/cudnn SDP状态。保留四行分母与失败attempt，不退款/不重试，partial数组及report保留。192MiB新输出总cap（8MiB final reserve）、200MiB floor、900s cooperative wall；不能抢占单个阻塞kernel。

每次module/API保存原float32输入/输出、shape/stride/storageoffset/device，含完整257 hidden、12layer边界、最终norm、CLS/projector、全部搜索candidates/cost、CEM九数组与输出、选中macro/完整rollout、低层完整两步原生history/Embedder/predictor/projection与输出。strict全量census/文件SHA/NPY原bytes/EOF/无对象，拒绝unknown/漏/乱序/父子错event；独立重建high/low autoregressive lineage及**FIRST到low goal逐位delivery**，从原blind RGB独立重算预处理逐值比对。192项平方和按gamma194 float32 arithmetic bound核查，**不作为神经parity容差**；CEM NumPy mean/sampleSTD/logging独立沿冻结8eps reducer工程容差。

当前来源回执不是本新增kernel正确性的证明；saved strict不神经重算、不能凭自己生成的arrays证明神经方程正确或硬件真实性。MAIN受控run提供真实身份；RAM测试只工程回归。

## MAIN CLI

cwd `/home/gbk/code/LeWorldModel/le-wm-main`。所有输出均新绝对路径、inner results下本TAG加下划线前缀，已存在则拒绝。manifest/report SHA必须MAIN实际产生，不填未来hash。

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_hilewm_rgb_high_low_20261028_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.agent_hilewm_rgb_high_low_20261028_v1 prepare --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_hilewm_rgb_high_low_20261028_v1_prepare_MAIN01
.venv/bin/python -B -m experiments.spectral_reachability.agent_hilewm_rgb_high_low_20261028_v1 verify --manifest "$HILEWM_RGB_MANIFEST" --manifest-sha256 "$HILEWM_RGB_MANIFEST_SHA"
CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python -B -m experiments.spectral_reachability.agent_hilewm_rgb_high_low_20261028_v1 run --manifest "$HILEWM_RGB_MANIFEST" --manifest-sha256 "$HILEWM_RGB_MANIFEST_SHA" --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_hilewm_rgb_high_low_20261028_v1_cuda_MAIN01 --main-authorized
.venv/bin/python -B -m experiments.spectral_reachability.agent_hilewm_rgb_high_low_20261028_v1 consume --manifest "$HILEWM_RGB_MANIFEST" --manifest-sha256 "$HILEWM_RGB_MANIFEST_SHA" --report "$HILEWM_RGB_REPORT" --report-sha256 "$HILEWM_RGB_REPORT_SHA" --receipt /home/gbk/code/LeWorldModel/le-wm-main/results/agent_hilewm_rgb_high_low_20261028_v1_consume_MAIN01.json
```

worker初轮40pass/1fail为冷导入旧证据层经pickletools间接import标准库pickle；新层延迟加载，冷测试保留拒绝。后续43pass/1fail为shape-only fixture不支持None维索引，定向修夹具。两者均未运行神经、非科学负结果。已运行专属44passed/1.81s及与冻结real-high-CEM联合83passed/3.63s：真实IR literal/257 ABI读取、shape-only pixel方程和low动作时序、全4临时NPY/report consume、完整资源/模型/精度/预处理/分母与native/CEM mutation拒绝。随后补固定输入分母拒绝回归，最终运行以交付消息记录为准。production prepare/run/consume尚待MAIN，不能把这些测试当E1证据。MAIN prepare/consume重建原证据层时可经其既有stdlib pickletools导入pickle模块，但从不执行load/Unpickler；新模块冷导入与本版worker RAM路径拒绝任何新增pickle/torch/foreign runtime导入。
