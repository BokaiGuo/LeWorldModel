# P1 固定库 terminal-witness selector（新增算法，不签 P1）

## 已核资产与边界

源为 `le-wm-main/results/` 下以下四文件；读取实际 schema 和全部候选字段确认：allmode 四行各42、mixedmode四行各222，全部已有 `predicted_terminal/nonlinear_terminal_cost/context`，并有同 raw 的独立 H2 replay。两库四个 task signature 完全一致。无需新神经计算来排序；不生成新物理证据。

|文件|SHA256|
|---|---|
|agent_allmode_prefix_20261028_v2_prepare_MAIN01.json|c11543eaa251903ff6d0429d2447f03c7427e1fe65be3d0aabfb0d9c4ef79d0d|
|agent_allmode_prefix_20261028_v2_cuda_MAIN01/report.json|c3e38899a12cd163d75918da79ef029adb14b2479f74f15bdfac0b6ffc57242d|
|agent_mixedmode_prefix_20261028_v1_prepare_MAIN01.json|8f6d027116b76334e80f9a761688ffd33825bc25d825484012d37230f083ed6b|
|agent_mixedmode_prefix_20261028_v1_cuda_MAIN01/report.json|d6166b98958af70cecb8529135396c77467b7c803bfe12a1a5de29c422c22ad2|

只读这些模型 prepare/report 及 report 所绑定的 float32 raw/module/J 数组；不递归重建 parent、不打开 checkpoint、不读图像或物理 outcome。`native` module tensor 是冻结模型内部数组，不是物理 state/label。引用原逐行 `validate_execution`，独立重算原 bank/每项 suffix ridge、原 winner，并验证全 forward/module/J 数组桥接；不是另行认证完整 upstream global journal/freshness，旧 consume 的职责不被替代。原已有 source 的直接数值 helper 哈希在 prepare 中核对；新 source/tests/doc 哈希进入 derivative。

## 两栏规则（无参数可调）

原栏严格 `stable argmin(stage_cost + own_reach_value, index)`，beta=1，保留原 suffix solve、Q、raw 和所有候选；新栏 `stable argmin(mean((Q@(predicted_terminal-goal))**2), index)`。原 Q 为固定 identity192，因此是192维 mean squared error，不是 sum，也不是 ridge 值。两栏逐候选完整输出，分别包含自己的 winner index/score/10×2 literal commands。

新栏是在**原候选同一 prefix + 原 own-ridge suffix 动作见证**上重排，不是求非线性最优 suffix，不是新 selector 获得了真实任务优势。不另拆 MAIN 所做 O−T 代数分解，不从其数值结果调算法。

分母：allmode168槽、mixedmode888槽，共1056候选槽/8库行；实际仍是同4个 root-condition、2个 known物理根，不是8个独立根。两个库重叠的首42逐库保留，不去重，不按 outcome 过滤。主 model 非贪心标识明确为 `stage_cost > initial_Q` 的严格 uphill；另标 `stage > zero-prefix stage`、`stage > stage-greedy stage`，三者不混同，也不声称真实非贪心。无额外 margin/threshold。

绑定：prepare/report 外部固定 SHA，四 slot 的 packet/Q/原 task/blind 哈希跨库一致；完整 root context/goal/Q 单独摘要。原真实数组桥接之外加 canonical context 比较与 float32 raw signed-bit 比较，tanh只执行原 NumPy float32 参数映射，不二次 squashing。原 packet 没有额外绝对时间字段，不凭空推定 teacher15/actual20；时间来源由原 task seal 绑定，不移植其他面板。

若纯函数输入缺失整套 terminal字段，保留该槽并输出 literal suffix H1（预测 prefix 后三帧上下文、同 raw）及独立 full H2（原上下文、完整10动作）请求，任何缺失即**不产生** terminal argmin；只缺部分字段则拒绝。当前固定生产 SHA 全部已完整，生产 CLI 仍先跑原完整 execution validator，不接受修改过的 report 冒充待补数据。没有 CUDA 请求执行器，也无新增预测必要。

## 资源与有限解释

新神经/physics/训练/Qfit=0。已有 allmode paid predict/AE/pred_proj 各1032，J172、VJP33024、成功 helper SVD176；mixed 各5352，J892、VJP171264、成功 helper SVD896。衍生 analyze 重算4+4个原 bank和1056个原 suffix solve，成功 helper SVD共1072，属于新CPU代数费，绝不是新神经费或零计算。另记 deterministic array读取次数/字节，selector cost计算共2120；原 validator 的 cost/比较和 mixed Pareto计算为额外代数开销，不声称精确 FLOPs。verify 再支付一整次同样 saved 费用。历史16编码、既有规划及失败 allmode v1预算不抹去，不声称总算力相等。

MAIN另报 mixed terminal indices [51,47,61,69]、原[51,51,23,23]，全部4个新 winner仍无相对anchor uphill，最佳predicted uphill候选的terminal亦更差。这是有限的模型库内结论；worker未据此改选或自产生产排序结果。新selector完成也不证明非贪心机制或真实收益，不倒签预注册，不签P1。既往teacher15反转与actual20→25真实任务结果不由此替代。

## MAIN 可执行命令

工作目录 `/home/gbk/code/LeWorldModel/le-wm-main`，没有 prepare/GPU步骤，也无需额外认证框架：

```bash
.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p1_terminal_witness_selector_20261028_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.agent_p1_terminal_witness_selector_20261028_v1 analyze --output results/agent_p1_terminal_witness_selector_20261028_v1_MAIN01.json
.venv/bin/python -B -m experiments.spectral_reachability.agent_p1_terminal_witness_selector_20261028_v1 verify --result results/agent_p1_terminal_witness_selector_20261028_v1_MAIN01.json --sha ACTUAL_ANALYZE_SHA256
```

输出必须全新独占stem文件，不能覆盖；analyze打印实际SHA。verify逐项重新构建整份紧凑derivative，不打印候选原大段context/J/重复字符串。源固定输入路径在 `SOURCES`，无需用户传入根/子集/选择模式。worker仅运行RAM测试、读取实际schema/字段计数和输入SHA；生产analyze/verify留MAIN。

测试含3候选可算toy证明两规则可分歧且新winner可model-uphill、tie稳定性、缺终态整分母、score/Q/goal/raw/signedzero/context/order/count/winner篡改拒绝、物理标签不影响选择；另复用原纯RAM producer，以真实42/222候选及258/1338 forward/module/J schema验证接缝，不是neural evidence。

本版本地结果：20 passed in 9.69s；module `--help` exit0。未运行生产analyze/verify，未生成科学结果或新的CUDA/物理产物。三件套交付后冻结。
