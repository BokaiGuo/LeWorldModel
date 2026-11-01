# terminal-witness 新物理终态 RGB 桥（独占 v1）

## 最小架构及真实 authority

只做 `真实physical consume → 四个选定动作的cp10 PNG → 原冻结encoder/projector → saved consume`，无新planner、selector、physics、预测、AD、训练、Q拟合或根筛选。新source/tests/doc，不修改旧封存；不伪造旧report/receipt schema，不做module globals patch。使用冻结低层实例hooks/encode guard，结束恢复。

Bohr实际模块为 `agent_terminal_witness_physical_20261028_v1`，source SHA `113961de2b48c0a9cb8d0c5829f676df9084fd11a0f2b16cee139f48adb08a7e`。使用它真实的 `consume(output,reportSHA)` 返回 `.consume`（不是假定的old mixed receipt）；要求 `terminal_witness4_physical_recorded_not_admitted`、全4行/known2/fresh0、无engineering errors。对外部给定实际 receipt SHA 核对后，重建并逐字比较整个consume结果。该consume已verify physical prepare，无需重复调用producer.verify；本桥从真实 `report.execution_binding` 读取原prepare。

已固定MAIN实际physical prepare：

`/home/gbk/code/LeWorldModel/le-wm-main/results/agent_terminal_witness_physical_20261028_v1_prepare_MAIN01.json`

SHA256：`b644dc1954f560448bca62a0131cee71319dc5f0af02c709672b91082af1f7a9`。

selector实际analysis SHA：`a3a317b557da8bb6eb7a38d582f7df4d230b1b849cced00eca1add85403e63d9`。模型prepare/report取自其 `inputs[family=mixedmode]`，分别固定SHA `8f6d027116b76334e80f9a761688ffd33825bc25d825484012d37230f083ed6b` / `d6166b98958af70cecb8529135396c77467b7c803bfe12a1a5de29c422c22ad2`。它们提供四个slot的原goal192/context/Q及已选candidate的真实nonlinear终态预测。MAIN实际physical report SHA `d02868f33f5b4b348c53d3684f30d2681124fc391b4f8d0f9b1a9e8992a04850`，完整264PNG strict receipt SHA `0f644f89d5864f818805ce95b663d19d4a5d094b65e8f320ef30e3ef3491e8d4`，均已绑定，无future receipt。worker已只读核实际四行horizon_completed、cp字段schema及四goal绑定，无评分修改。

接口对齐来自Bohr实际共享source和MAIN确认，本会话没有可调用跨agent消息工具；没有另造IPC/认证manifest。

## 逐行绑定与输入白名单

固定四行顺序 `(root0,off),(root0,on),(root1,off),(root1,on)`。严格绑定物理plan/selector panel/原slot/candidate的 `task_sha256,blind_sha256,selection_digest,selected_index,context_sha256,goal_sha256,q_sha256,raw_float32_sha256`；完整raw10是10×2 float32，不再tanh/clip/缩放。目标latent仍为原goal192、Q仍identity192，loss是float64 mean192 squared distance；不重新编码目标、不换metric。

cp10来自实际physical `rows[].checkpoints`，字段 `step/png/file_sha256/pixel_sha256/state/snapshot`，同时必须吻合该行 `trajectory` 的真正plan10、原condition及off/on overlayindex、final state，实际own_returned=10；不拿cp5或早停终态补cp10。全部4槽保留，缺cp10为null，预计编码次数随可用槽确定（0..4），绝不在off/on或相同行之间去重。

模型唯一输入为 `model.encode({'pixels':preprocessed})`；物理state、reward、native_success只保存在evaluation/provenance，既无planner也不进model.encode。原 normalizer 源AST和mean/std保持冻结；本桥本身不编码动作、也不作forward预测。模型config/weights byteSHA和tensor hash由原prepare及冻结MODELS核对，policy必须原目录 `/home/gbk/.stable_worldmodel/checkpoints/pusht/lewm`。

## CUDA与saved验证

显式MAIN `--authorize-host-cuda` 后才可进入run；cuda:0预检，无CPU neural fallback，offline本地模型、eval/无grad、batch1、TF32/autocast关闭、SDP MATH、前后tensor hash一致。冻结低层Trace保存完整原始NPY（losslessgzip包装）：PNG实际pixels、独立preprocessed `[1,1,3,224,224]`、encoder实际input `[1,3,224,224]`、**完整hidden `[1,257,192]`**、原CLS→projector input、projector output `[1,192]`、latent192。绝不把197/截断hidden伪作257。

不使用旧 `native.guard`（它会临时patch torch AD全局）。新 `encode_guard`只包装当前model实例的方法，使用原native.Ledger记encoder/projector顺序；公开AD入口由可恢复的当前线程 `sys.setprofile` 探针识别，**torch模块/类属性不替换**。探针取 `inspect.unwrap` 后的原函数code，避免共享decorator wrapper误杀普通推理；保存/恢复先前线程profile。首个forbidden立即终止该run、保留blocked，不得恢复推理或retry；CPython遇探针异常可能关闭profile，所以blocked不可清除。不是对未列出的内部C++backward引擎调用/FLOPs的独立计数；冻结eval/inference_mode且模型唯一输入为pixels。已有RAM测试验证globals身份和profile恢复及decorator误杀反例。

run即时验证原p0预处理与冻结 `preprocess_pixels` 的字节一致；consume重新打开原PNG、核文件/像素SHA并**独立重算pixels→preprocessing**，再逐位验证完整encoder/CLS/projector/latent数组链。不是只比较preprocessed与encoderinput。事件list逐个构造比对，开头/末尾/中间任何unknown或缺失、乱序事件均拒绝，不过滤eventkind。native seal先封存后算Q。

持久化：每个image/module/guard attempt和return都有独立fsync事件文件；即使forward失败也留下raw pixels、已写数组、部分row、error/blocked/modules和report，失败不能consume为完整成功；不重试、不补齐。prepare失败在CUDA前结束，不伪造执行receipt。输出独占新stem目录和metadata文件，不能覆盖。

## 资源和结论边界

完整四图最多encoder4/projector4；action_encoder/predict_API/predictor/pred_proj/AD/physics=0，禁止API有独立guard。保留实际attempted/completed、event_count、elapsed及全部artifact SHA；64MiB输出cap、8MiB final reserve、200MiB disk floor、600s整job限制。prepare/verify/consume只读saved arrays/PNG及代数，不加载neural。它们重建Bohr完整consume，而Bohr会重建selector；这会支付旧bank/ridge/SVD/array I/O的CPU验证成本，**不是零总计算**，不把这些嵌套核验重复算成新CUDA。receipt保留继承selector资源、实际新physical counts；API nested axes不能相加冒充FLOPs。FLOPs未知，不声称equal compute。

本桥只交付选定动作的真实观测latent、actualQ及prediction error；不形成同根多动作排序对照，不声称闭环改进、非贪心机制、freshness、P1通过。`model_prefix_uphill_from_anchor`仅沿用模型标识，不能改名为真实非贪心。MAIN已有“terminal选择仍无model uphill”等有限结论不被图像编码自动推翻。

## MAIN执行命令

工作目录 `/home/gbk/code/LeWorldModel/le-wm-main`。Bohr物理run和真实consume已完成，下列prepare可直接执行；run/consume中的RGB SHA替换为此前实际输出值，不是需另签的认证文件：

```bash
.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_terminal_witness_terminal_rgb_20261028_v1.py

.venv/bin/python -B -m experiments.spectral_reachability.agent_terminal_witness_terminal_rgb_20261028_v1 prepare --physical-output results/agent_terminal_witness_physical_20261028_v1_MAIN01 --physical-report-sha d02868f33f5b4b348c53d3684f30d2681124fc391b4f8d0f9b1a9e8992a04850 --physical-receipt results/agent_terminal_witness_physical_20261028_v1_consume_MAIN01.json --physical-receipt-sha 0f644f89d5864f818805ce95b663d19d4a5d094b65e8f320ef30e3ef3491e8d4 --output results/agent_terminal_witness_terminal_rgb_20261028_v1_prepare_MAIN01.json

.venv/bin/python -B -m experiments.spectral_reachability.agent_terminal_witness_terminal_rgb_20261028_v1 verify --prepared results/agent_terminal_witness_terminal_rgb_20261028_v1_prepare_MAIN01.json --prepare-sha ACTUAL_RGB_PREPARE_SHA

.venv/bin/python -B -m experiments.spectral_reachability.agent_terminal_witness_terminal_rgb_20261028_v1 run --prepared results/agent_terminal_witness_terminal_rgb_20261028_v1_prepare_MAIN01.json --prepare-sha ACTUAL_RGB_PREPARE_SHA --output results/agent_terminal_witness_terminal_rgb_20261028_v1_MAIN01 --authorize-host-cuda

.venv/bin/python -B -m experiments.spectral_reachability.agent_terminal_witness_terminal_rgb_20261028_v1 consume --output results/agent_terminal_witness_terminal_rgb_20261028_v1_MAIN01 --report-sha ACTUAL_RGB_REPORT_SHA --receipt results/agent_terminal_witness_terminal_rgb_20261028_v1_consume_MAIN01.json
```

run需MAIN正常host CUDA执行权限，不需额外IPC或许可manifest。worker不执行其中任何生产prepare/run/consume，不GPU、不physics。接口返回consume `.rows[]`：身份/选定raw摘要/goal摘要、`predicted_Q,actual_Q,prediction_minus_actual_Q,endpoint_MSE,actual192,goal192,latent_missing,evaluation`；全部4行，包括missing null。

## 测试

仅RAM实际schema/真实224-257-192 tensor hook doubles，无torch/neural/env执行：新physical schema拒绝old；task/raw/goal/context/selection/plan seal/预测终态/actual frame state/renderer条件/早停cp10/缺槽篡改；完整encoder+projector+禁止调用guard；部分及全部missing仍4行；unknown事件头尾拒绝；PNG像素和独立preprocessing/hidden197/projector latent/计数/guard篡改；未授权run在I/O前拒绝；torch globals不改/profile恢复、共享decorator不误杀。所有新增逻辑均先见失败测试再实现。最终冻结测试结果见交付消息。
