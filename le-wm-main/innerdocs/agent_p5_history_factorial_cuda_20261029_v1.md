# Fixed164 history factorial: exploratory model-internal intervention

本独占source/tests/doc不改任何H1/H2/bridge封存文件，不运行worker神经/物理。
goal API仍返回旧blocked目标，未覆盖或完成该目标；本次个人完成条件是实现、RAM/临时文件测试、只读saved接线验证、三SHA冻结，非研究完成。

## 操作与完整分母

事后探索性设计，不称预注册确认。两known roots × off/on × 全原164分支/80 pairs全部保留，不依cosine、h、坐标或结果筛选；24原零响应也保留。
固定已执行H1 report `e7ac4ef9eb7491f619de25b04c55f5e56560e2833a840dd5305f3db6207b3a5d` 和 H2 report `e5a561215bfe7b6f1c1c224e7661211e3f237fb16458a6a409f5a1da0fc08579`，并绑定其原独立strict回执、prepare、source、完整artifact与journal hashchain。
prepare/verify逐byte SHA核完整旧目录与上游prepare闭包；承接原strict对原推理/rollover的认证，不声称重新执行旧NN或重建训练lineage。

H1原终段为actual5/10/15→20；H2原第二次调用为actual5/10+predicted15→20。两者终段完整context时间槽都是5/10/15，action为原raw6..10、11..15、16..20的3个packet，每packet5raw。最终outgoing16..20不改变；prepare逐位核完整3packet动作和raw6..20相同，拒绝错root/task/selection/condition/label/model/stats/time。
不是单独15槽干预，也不是纯horizon干预：交换完整保存的视觉块 `(1,3,1,16,16,384)` 与proprio嵌入块 `(1,3,256,16)`；保留原批布局/数值接缝，不强行使前两槽全等。

| cell | visual history | proprio history | 来源 |
|---|---|---|---|
| AA | H1 actual | H1 actual | 原H1 full3输出，0新增 |
| AP | H1 actual | H2 rolled | 新predict一次/branch |
| PA | H2 rolled | H1 actual | 新predict一次/branch |
| PP | H2 rolled | H2 rolled | 原H2第二次full3输出，0新增 |

固定branch-major AP然后PA，无重试、额外diagonal replay或padding。每次原生predictor调用前durable intent与完整输入NPY，返回后durable return与完整三项返回树（visual full3、None、proprio full3）。new IO保存顺序的5个tensor均唯一，不省略prop输出。
所有328新预测完成且模型不变检查后，落盘 `predictions_seal.json`。读取并核完全部328预测后才发`labels_unlocked`、解码旧标签特征及旧AA/PP输出作评分；旧上游report内既有标量分数不是新评分结果，也不传入predictor。

## 共同标签与数值

共同真实label使用原H1 actual20 patch `(1,256,384)`，每个cell完全相同。原H2 label batch为15/20 `(2,256,384)`，与H1可能有batch浮点差异；每branch记录bitwise_equal/maxabs，不以各自不同label计算factorial。
预测只取full3的最终token作MSE，但full3原输出全部保留。float64 NumPy C-order MSE，不拟合Q/阈值/新readout。
每branch四MSE、feature效应L2，以及MSE效应按固定代数：

```
visual = ((PA-AA)+(PP-AP))/2
proprio = ((AP-AA)+(PP-PA))/2
interaction = PP-PA-AP+AA
```

同公式也用于完整feature向量；每pair四cell response范数/cosine/plus-minus MSE及response效应L2全量报告。exact zero时cosine=null，不补epsilon、不作显著性或独立164样本宣称。
真实agent位置/速度是原JEPA额外权限；不是RGB-only或跨模型同输入公平排名。混合history可能不可达，只辨识冻结模型对输入块的计算响应，不是物理counterfactual、控制收益或充分根因认证。P1/P5 admission均false。

## 工程合同

仅MAIN HOSTCUDA offline，原native loader/原normalizer，frozen eval、inference、TF32off、MATH attention。模型加载按原strict schemas与actual loaded state SHA对齐H1/H2，预测前后与final全模型receipt相等。加载encoder模型不forward：所有三个top-module均安装hooks，两个encoder任何调用立即拒绝。
新predictor328 API =328 batch samples=984 token-frame samples；official_encoder/proprio_encoder API/sample/frame均0。0AD/训练/physics。原H1/H2付费计数及wall单列inherited_costs，不称新旧等总算力。
完整tensor IO原始有效字节上界精确806132160B，约0.751GiB（328×2457720B）；IO cap1GiB；NPY单文件4MiB；元数据单文件32MiB；输出总cap2GiB（含journal、NPY headers、report），terminal reserve64MiB；磁盘floor200MiB，开始要求cap+floor；全job wall900s包含verify/load/final重hash，失败fsync收尾可能继续。初始CUDA free要求4GiB，不冒充activation峰值；实际CUDA allocated/reserved/free/total和host RSS保存。
writer exactly1 LF canonical JSON，复用封存纯codec方法，不monkeypatch globals；strict journal拒绝空行、非canonical和断链。actual attempts/returns分开，partial保持164/80，失败不是科学负结果。只写NEW同stem目录/metadata，禁止覆盖；最终核完整源/权重/artifact SHA。

## CLI（MAIN执行；worker不prepare生产或CUDA）

在 `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
.venv/bin/python -m pytest -q tests/test_agent_p5_history_factorial_cuda_20261029_v1.py
.venv/bin/python -m experiments.spectral_reachability.agent_p5_history_factorial_cuda_20261029_v1 prepare --output results/agent_p5_history_factorial_cuda_20261029_v1_prepare_MAIN01.json
.venv/bin/python -m experiments.spectral_reachability.agent_p5_history_factorial_cuda_20261029_v1 verify --prepared results/agent_p5_history_factorial_cuda_20261029_v1_prepare_MAIN01.json --prepare-sha PREPARE_SHA
.venv/bin/python -m experiments.spectral_reachability.agent_p5_history_factorial_cuda_20261029_v1 run --prepared results/agent_p5_history_factorial_cuda_20261029_v1_prepare_MAIN01.json --prepare-sha PREPARE_SHA --output results/agent_p5_history_factorial_cuda_20261029_v1_cuda_MAIN01 --authorization MAIN_P5_HISTORY_FACTORIAL_HOSTCUDA_ONLY
.venv/bin/python -m experiments.spectral_reachability.agent_p5_history_factorial_cuda_20261029_v1 consume --output results/agent_p5_history_factorial_cuda_20261029_v1_cuda_MAIN01 --report-sha REPORT_SHA --receipt results/agent_p5_history_factorial_cuda_20261029_v1_consume_MAIN01.json
```

prepare/run/consume返回真实路径SHA，不预填未来SHA。strict重建源闭包、核全部inventory/hashchain/actual输入块/完整原生输出/计数、seal先于评分、全四cell/80pair算术、完整artifact角色与tensor字节，成功status=`all164_history_factorial_and80_pairs_verified`。partial仅返回`blocked_partial_capture`，CLI exit2。完整report ABI：rows164含predictions[两个arm]、score[四cell MSE/MSE_effects/feature_effect_L2/label数值接缝]；pairs80含四cell与response_effect_L2；calls328、counts attempted/returned、prediction_seal、model_before/after_predictions/after、strict_loads、imported_sources、memory、artifacts、journal_chain_tail。
前置：全部原H1/H2 actual目录/strict/prepare闭包原样存在，原模型缓存与loader/stats可用，MAIN源码审读及生产授权。没有新的实际运行/科学结果，测试仅RAM和临时目录fixtures，不据测试宣称研究通过。

## 已做工程验证（非生产结果）

包含真实Store两event→完整canonical journal/hashchain回归、NPY EOF、错time/root/task/selection/condition/action/kwargs拒绝、固定full-block混合、原生hooks attempts/returns与encoder禁调用、scalar/vector factorial、缺预测seal拒绝、preflight partial164/80保留。另有全尺寸328次IO合成端到端：完全不导入torch，不加载任何模型，用fixture native接口替身，真实写约0.751GiB tensor数据到pytest临时目录，执行run→seal→score→strict并拒绝输入关联篡改；其输出仅临时fixture，非实际NN或科学证据。测试建议预留约1GiB临时磁盘。
只读actual metadata smoke已核全164/328输入join、80pair、27251bindings，prepare序列化约5.83MB<32MiB；未导入torch、未解码label features、未写生产prepare/report。MAIN仍须独立审读、prepare/verify及真实HOSTCUDA/strict。
最终本stem46项与冻结H1/H2测试联合执行122 passed in20.15s；无模型/物理生产运行。
