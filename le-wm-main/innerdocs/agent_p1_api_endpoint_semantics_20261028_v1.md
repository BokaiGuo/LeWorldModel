# P1 API端点语义修正：独占saved consumer

仅新增本stem source/tests/innerdoc及本stem结果；不修改冻结API v1、MAIN实际report/receipt、旧负结果或共享文件。不运行神经/物理/网络，不创建后代，不训练/拟合/改Q/门槛。旧个人goal不覆盖，本有界工程完成不等于研究goal或P1完成。

绑定实际API report SHA `38ad7134ba7f13b7244a9bc9ad311d476ccd67a0df53daf9fa7f314e7876ae3e`，receipt SHA `fc6f7022efc1df77b3eb85ff8262a87d0cbb5cf6bd1f65d6d77f512ae5651d68`。先完整重建冻结`api.consume`并要求与MAIN receipt逐字段相等，继续验证360实际entry/全数组/源模型闭包，不将不正确的字段删除后冒充旧receipt。另加载原teacher同word evaluate的真实`pred_proj.output`三行数组、旧shifted context、真实API两路predict output与public输出，以及原goal/actual arrays。

## 明确的三种数组语义

|数组|row0|row1|row2|
|---|---|---|---|
|native predict output full3|模型对输入位置0的输出，名义next10|模型对输入位置1的输出，名义next15|模型预测endpoint20|
|旧after/shifted context|保留输入观测z10|保留输入观测z15|模型预测endpoint20|
|本轮重建shifted context|输入z10原bits|输入z15原bits|本轮native最后一行|

native前两行是模型输出，不是保留的观测，也不因为名义next时间相同就等同真实观测。public返回另为四行：`[z5,z10,z15,预测z20]`，不得拿public前三行当native三行。原teacher后context来自transition的shift/append，不是原predict输出全三行。

三个合法比较分别输出，不混合：endpoint20↔旧endpoint20；native full3↔原真实native full3；重建shifted context↔旧shifted context。还列逐native行比较和两行保留观测的字节核验。每个comparison同时保留原numeric exact/maxabs/MSE与bitwise_equal，不放宽数值门槛。

原v1 `vs_original_prediction`是native3↔shifted context3的**跨语义比较错误**，前两行不具有相同角色；不能由它的false/maxabs/MSE推断旧预测漂移，更不能作causal drift证据。新报告逐槽逐route原样保留`deprecated_v1_field.original_value`并独立重现旧算术，标记`invalid_cross_semantics_for_drift`、`causal_drift_evidence=false`；旧报告不改。这个错误不影响旧consumer的public-vs-cached模块一致性或其基于最后一行的Q。

## Q与分母

从原goal array和两个端点重算float64 `mean((z-goal)^2)`、两端点Q差，以及与原actual39的误差/交叉项/平方项；每个current标量必须与旧receipt的**端点标量**一致。原Q必须与原prepare `original_predicted_Q`精确一致。不调用public criterion（sum）、不改goal/Q。真实新端点差异即使非零也保留为数据，不筛槽、不将其升级为因果解释。

固定40槽/80route，index12 actual仍null但预测比较照做；原rank-error完整80/20对及缺失rank语义原样保留。known roots仍2，P1/scientific admission/full40 release均false；不把latent指标当物理收益。顶层绑定实际report/receipt/prepare、原artifact census、源/模型闭包和新三文件；每个加载数组补齐file receipt、shape/dtype及计算出的array-byte SHA，旧压缩数组原三SHA不丢。

## MAIN saved-only命令

在`/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p1_api_endpoint_semantics_20261028_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.agent_p1_api_endpoint_semantics_20261028_v1 analyze --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p1_api_endpoint_semantics_20261028_v1_analysis_MAIN01.json
.venv/bin/python -B -m experiments.spectral_reachability.agent_p1_api_endpoint_semantics_20261028_v1 verify --analysis /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p1_api_endpoint_semantics_20261028_v1_analysis_MAIN01.json --analysis-sha <analyze实际SHA>
```

输出严格新建不覆盖；verify从原saved数组重建整个新报告。无CUDA入口，无physics release。RAM测试不是科学证据；本报告只消除一次具体语义错配，不证明模型正确或反转原因。后续修改只新版本。

本次工程验证：新33项RAM测试通过，与冻结API/history/rank联合129项通过。真实saved `analyze`两路各40槽端点numeric与bitwise均完全相同，maxabs/Q delta为0；native full3对原native full3、正确重建shifted context对原shifted context也各40一致。旧跨语义字段依旧各40项false，明确不能把该false当漂移。index12 actual未补。最终saved输出采用独占`analysis_NASH01.json`，MAIN可另建上述`analysis_MAIN01.json`重算；不改任何旧实际产物。
