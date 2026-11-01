# 原 mixed surrogate winner 四槽真实 terminal RGB companion

## 范围与最小架构

独占新source/tests/doc，不修改冻结的terminal-witness v1或任何旧算法/结果/gate。只做原mixed固定winner的cp10图像编码，供MAIN随后与新terminal-witness winner作同任务 predicted→actual latent→physical配对。本模块不重选动作、不读outcome选根、不拟合Q、不运行新物理、不提供P1结论。

三步：`真实mixed.consume重建继承认证 → 按route取完整原行及原目录cp10 → 原冻结encoder/projector及严格saved consume`。不是旧schema改名，不把allmode原行拼成新mixed物理执行，不伪造新selector/新physical的report/receipt。

## 固定真实输入

全部路径在 `/home/gbk/code/LeWorldModel/le-wm-main/results/`：

|输入|SHA256|
|---|---|
|agent_mixedmode_physical_20261028_v1_MAIN01/report.json|58bd3b4fcb2042e79d796f3eb26e0004f88b098c70ffdb044d1b03a3eedcc965|
|consume_mixedmode_physical_MAIN_20261028_v1_MAIN01.json|eb3c2752dc6ae93d99319439895c99dd136b9847a81cc83dfb8ec71b6ace9e75|
|agent_mixedmode_physical_20261028_v1_prepare_MAIN01.json|667a212745650520c7c852c1192d34e98960c260fafbbd021e1fabed47877049|
|agent_allmode_physical_20261028_v1_MAIN01/report.json|3ca4891a5f0823bede431e85442513a095b1e9bc8d49c87feefbb56663dca3b4|
|agent_mixedmode_prefix_20261028_v1_prepare_MAIN01.json|8f6d027116b76334e80f9a761688ffd33825bc25d825484012d37230f083ed6b|
|agent_mixedmode_prefix_20261028_v1_cuda_MAIN01/report.json|d6166b98958af70cecb8529135396c77467b7c803bfe12a1a5de29c422c22ad2|

MAIN summary真正schema是 `MAIN.mixedmode_physical_receipt.v1`，**内部 `.receipt` 才是 `agent_mixedmode_physical_20261028_v1.consume`**。prepare调用真实 `producer.consume(固定目录,固定reportSHA)` 并逐字比较整个内部receipt，不以wrapper caveat/status冒充认证。producer.consume会重建原mixed模型consume、allmode物理consume及routes，包含原context/source/raw/action/state/renderer/PNG和完整原生ledger认证。它是saved验证，不会启动物理或神经；本桥不再重复单独调用allmode.consume。

source冻结：mixed物理模块SHA `e4fac2479254db07c34de5d9f88fbe0d19606b8ce2b1f8d7d5e42cbad8a042ae`；低层terminal-witness桥SHA `310eda7b2fcc91e069a917f5963a962b610e9ef239a5796d47cc2b466dbec396`。

## 整行继承，而非“空轨迹=缺图”

固定4槽顺序root0 off/on、root1 off/on；routes严格 `[execute,execute,inherit,inherit]`，不会根据收益选择route。

前两行在mixed本目录真实执行，直接使用该行cp10。后两行mixed wrapper的trajectory/checkpoints为空、own_returned=0，**不能当成missing cp10**：必须确认 `wrapper.inherited_evidence` 与prepare中 `old_physical_consume.rows[i]` 整行canonical完全一致，评价字段也完全相同；然后仅从这份完整原行取trajectory/checkpoints和final state，PNG目录为prepare的 `arguments.old_output`（固定allmode物理目录）。不拼接新旧prefix/suffix/观测，不借其他行、其他renderer或相近state图像。

继承额外逐字核对：原allmode/mixed编码slot的task/context/goal192/Q完全相同；source native root/goal metadata一致；原行plan task_id/task_sha256/blind_sha256/source_report_sha256一致；原allmode动作10×2 float32字节必须等于原mixed winner动作。不能把新terminal selector的动作、终态Q或物理schema移植成原winner证据。

cp10在唯一source整行中必须有唯一plan10 trajectory，condition/overlayindex、PNG/fileSHA/pixelSHA/native state/snapshot逐项吻合，并与真实final state一致；own_returned=10且initial_exact。真正早停缺cp10保留null，全4分母不删。原wrapper为空不计missing。读PNG时再验证文件SHA、原RGB224像素SHA，不做renderer重画。

## 预测值与目标

原mixed模型report所有222候选保持完整，重现原稳定 `argmin(objective,index)`，不按nonlinearQ或outcome重选。输出 `original_objective=stage+ridge` 单列；`predicted_Q` 取该原winner的**nonlinear_terminal_cost**，预测latent取 `predicted_terminal`，绝不把ridge objective当terminalQ。

原mixed slot给goal192/context/Q及task身份，使用原identity192、float64 mean192 squared distance；normalizer原源AST及float32 mean/std绑定，不新tanh/clip/scale，不重新编码goal、不拟合metric。native root/goal/state仅用于评价和溯源，model.encode唯一输入仍为pixels。模型已predicted-uphill标识仅模型诊断，不代表真实非贪心。

## CUDA及严格验证接口

复用冻结v1中policy/schema-free `process`、`analyze`纯算术ABI、`encode_guard`及Trace/Store底层，但**不调用其固定新selector的build/verify/run/validate/consume，不做任何schema/contract projection**。本模块自有这些入口和固定oldmixed authority；run编排因此独立，未修改旧source常量或globals。

与冻结v1同样HOST CUDA only、cuda:0、batch1、原本地offline冻结weights、eval/inference、TF32/autocast关闭、SDP MATH、前后tensor hash一致。guard仅当前model实例包装和可恢复线程profile，已包含 `inspect.unwrap` 修复，无torch/module globals patch；AD/public prediction/action_encoder/predictor/pred_proj禁止。唯一神经工作为四张实际PNG的encoder/projector，无CPU neural fallback。

保存lossless gzip原NPY：pixels、原preprocessed、真实encoderinput、完整hidden `[1,257,192]`、CLS projectorinput、projectoroutput、latent192。consume从原PNG独立重算 `preprocess_pixels` 并逐位核验encoder→完整hidden→CLS→projector→latent；不只比较input/preprocessed。所有image/module/guard attempt/return逐事件精确比对，未知事件不筛掉。先seal native输出再计算评价。

每个事件/数组fsync，失败保留partial ledger/report/errors/blocked，不重试、不补齐、不consume成成功；目录/metadata文件必须新建独占stem，不覆盖。64MiB cap、8MiB final reserve、200MiB disk floor、600s整job。

## 资源与配对边界

encoder/projector最多各4次（继承物理不等于继承latent；四图各自新编码，不去重）；新physics/predict/AD/训练/Qfit=0。prepare/verify/consume会重建历史完整消费链，支付原bank/ridge/SVD/数组I/O代数验证费用，不是零总计算；原mixed model/数学counts、mixed新物理counts、allmode继承物理counts分栏保留，不能相加嵌套axes伪作FLOPs。FLOPs未知，不称equal compute。全4槽仍known2/fresh0，新增4图不产生独立根。

consume `.rows[]` 含 `task_sha256,selection_digest,selected_index,raw_float32_sha256,goal_sha256,predicted_Q,actual_Q,prediction_minus_actual_Q,endpoint_MSE,actual192,goal192,evaluation,source_route,source_row_sha256,source_physical_report_sha256,original_objective`。MAIN应按root_slot/condition/task/goal/context/raw各自身份连接新旧方案，明确动作可能不同，不把task相同说成action相同。本模块不自动签收益、选择改善、freshness或P1。

## MAIN CLI

工作目录 `/home/gbk/code/LeWorldModel/le-wm-main`；所有上游真实输入均固定，prepare不需要未来SHA或新物理：

```bash
.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_terminal_witness_control_rgb_20261028_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.agent_terminal_witness_control_rgb_20261028_v1 prepare --output results/agent_terminal_witness_control_rgb_20261028_v1_prepare_MAIN01.json
.venv/bin/python -B -m experiments.spectral_reachability.agent_terminal_witness_control_rgb_20261028_v1 verify --prepared results/agent_terminal_witness_control_rgb_20261028_v1_prepare_MAIN01.json --prepare-sha ACTUAL_CONTROL_PREPARE_SHA
.venv/bin/python -B -m experiments.spectral_reachability.agent_terminal_witness_control_rgb_20261028_v1 run --prepared results/agent_terminal_witness_control_rgb_20261028_v1_prepare_MAIN01.json --prepare-sha ACTUAL_CONTROL_PREPARE_SHA --output results/agent_terminal_witness_control_rgb_20261028_v1_MAIN01 --authorize-host-cuda
.venv/bin/python -B -m experiments.spectral_reachability.agent_terminal_witness_control_rgb_20261028_v1 consume --output results/agent_terminal_witness_control_rgb_20261028_v1_MAIN01 --report-sha ACTUAL_CONTROL_REPORT_SHA --receipt results/agent_terminal_witness_control_rgb_20261028_v1_consume_MAIN01.json
```

worker只实现、RAM测试和实际saved接口只读核对；不执行生产prepare/run/consume，不GPU、不physics。MAIN普通host CUDA授权足够，无额外IPC或认证框架。

## 验证

37个RAM测试通过（2.90s）：真实mixed/allmode schema与完整222分母model_plan helper；空wrapper继承正确取原目录；route/整行/raw/context/root/goal/condition/state/早停cp/动作预测/summary内部receipt篡改拒绝；四图完整native张量链；new和inherited真正missing均保留4槽；独立pixel preprocessing/hidden197/latent/guard/资源/未知事件拒绝。首次fixture缺prefix/suffix字段导致测试失败，已按原helper真实schema补齐，未放松validator。未运行neural或物理，不把RAM断言计作科学结果。
