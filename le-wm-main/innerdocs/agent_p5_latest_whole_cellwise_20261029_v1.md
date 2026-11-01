# latest × whole-history：完整保存IO逐cell核验

独占stem `agent_p5_latest_whole_cellwise_20261029_v1`。这是看到MAIN报告均值相同之后提出的探索性工程/机制定位，不是事前确认、不是新的预测实验。不修改latest、whole-history、Nash summary或其资产。worker只执行RAM/临时磁盘fixture；实际saved analyze/verify由MAIN执行。旧goal API仍是无关blocked目标，不覆盖、不伪报complete；本交付完成条件仅限这三个新文件及测试。

## 问题与固定范围

固定全164 branch、四cell AA/AP/PA/PP、原80 pair（含四baseline）；不选择非零/负向/某个root、off/on或有效例。保留原root/task/selection/raw outgoing16..20/actual与rolled全部引用/共同H1 label/label PNG身份。输出不是Nash的MSE均值、cosine、分层解释，不复制其summary。当前没有跨工作位消息工具，已请求MAIN转发这一边界，未假称已与Nash联络成功。

latest实际report必须为 `a9479f81d87b1b70ebd1f64db0289d9d2c6a5ae57932cd021f85cc5ecfd47316`，whole实际report必须为 `47b12d991d0b4c1c99ecf30ee90afe7bb5550f0babc4d786d6fca2e1cc5d8b8f`；两个strict路径/SHA都显式外部输入，并逐份调用封存owner.consume完整重建，要求整个dict相等。原两执行的prepare/source/model/inventory/journal/seal/call计账验证全部继承，不仅信任status字符串。

## 保存数组读取与身份

读取原实际module IO，而非只比较row['score']或请求重建数组。每branch每cell：

- AA来自原H1 predictor call `3*i+2`。
- whole PP来自原H2第二predictor call `4*i+3`。
- whole AP/PA来自whole实际两calls；latest AP/PA/PP来自latest实际三calls。
- 三个实际输入（视觉/原动作/本体）必须逐字节等于其正确cell的源拼接。latest锁前两槽、whole换完整块；禁止把whole PP当latest PP。
- 两个实际输出都核：视觉 `(1,3,256,384)` 与本体 `(1,3,256,16)`，中间返回值必须None。视觉指针还需等于原row记录。
- 原输入global frames `[5,10,15]`，三个输出successor frames `[10,15,20]`。输入末槽字段叫`latest15`，输出末槽叫`final20`，不混时标。

每branch额外逐槽比较actual与rolled原输入，用来确定前两槽是否原本就相同。每cell比较latest减whole的全部输入/full3输出/三个单槽/final20。每项保存shape、bitwise_equal、numeric_equal、不同float32 word数、数值非零数、maxabs、float64 L2、max_ULP及两侧原始C-order字节SHA。ULP采用单调float32数值格，正负零共享数值rank；bitwise仍严格区分正负零。NaN/Inf/shape/dtype错配拒绝，无容差。64MiB输出上限，不保存裁剪或降精度数组；完整原数组已有SHA引用。

所有80pair保留原完整schedule，关联minus/plus的全部cell比较，并列双端点full3/final20视觉exact相等标记。不从其中挑pair，不再次汇总响应方向、不把同基线pairs当独立样本。

## 可判定和不可判定

分类严格分为：相同输入相同输出（没有独立干预差异）；相同输入不同输出（执行数值接缝）；不同输入相同输出（机制未定）；不同输入不同输出。即使final20相同，full3前两输出不同也不能报告full3相同。相同MSE不等于相同输出。

源码证据读取实际prepare已绑定且report确实imported的AdaLN/attention文件，只作UTF-8/AST读取，绝不import这些模型文件。保留准确path/SHA/symbol/line/end_line/source text：

- 原v8 loader第124–128行：`local_window=(3,-1,-1)`、feature proprio、`proprio_tokens=0`、`proprio_encoder_inpred=False`；块内部`is_causal`默认False，但显式mask照样传给SDPA。
- `AdaLN_vit.py`第249–265行构造mask，第340–373行切片并传入每个block；FWAdaLNBlock.forward将mask传给RoPEAttention。
- `src/models/utils/modules.py`第38–49行builder中`local_window_time=T`、`t2<=t1`。原四帧mask截取实际三槽，其块允许矩阵为 `[[1,0,0],[1,1,0],[1,1,1]]`。最后槽允许读取5/10/15，不是屏蔽前两槽的架构。
- RoPEAttention.forward第269–273行把显式attn_mask传给SDPA。
- 原本体embedding Conv1d的kernel/stride由tubelet_size=1，原rollover只移位并追加预测15。这给出“前两槽可能本来相同”的源码理由，但是否相同必须由保存数组判定，不能先填结果。

允许边不证明有效影响；保存输出相同也不证明模型完全忽略history。不访问attention权重、不做新因果干预、不推断全部代码原因已排除。common-label身份严格对齐，分析器自身不计分；原strict重建仍会复算旧分数。这不是训练/物理反事实/控制收益/RGB-only/公平跨架构排名或P1/P5准入。

## 资源与CLI（MAIN执行）

0新predict/encoder/AD/physics/training。单进程逐branch读取，两次完整owner strict重建及最终全部bindings SHA复核；无常驻164份大数组。1800s wall、64MiB最终JSON、200MiB磁盘floor。原tensor字节成本仍在各自执行report，不能把此次saved读取当新模型调用。验证重复重建全部分析，不只验输出hash；写出仅限自身results前缀且exclusive，禁止覆盖。

```bash
.venv/bin/python -m pytest -q tests/test_agent_p5_latest_whole_cellwise_20261029_v1.py

.venv/bin/python -m experiments.spectral_reachability.agent_p5_latest_whole_cellwise_20261029_v1 analyze \
 --latest-output results/agent_p5_latest_slot_factorial_cuda_20261029_v1_cuda_MAIN01 \
 --latest-report-sha a9479f81d87b1b70ebd1f64db0289d9d2c6a5ae57932cd021f85cc5ecfd47316 \
 --latest-strict-receipt results/agent_p5_latest_slot_factorial_cuda_20261029_v1_consume_MAIN01.json \
 --latest-strict-sha c636f8d9c6624317d4da4f3ddb127b4f20710f9a5c02c212872d0d70c9cab642 \
 --whole-output results/agent_p5_history_factorial_cuda_20261029_v1_cuda_MAIN01 \
 --whole-report-sha 47b12d991d0b4c1c99ecf30ee90afe7bb5550f0babc4d786d6fca2e1cc5d8b8f \
 --whole-strict-receipt results/agent_p5_history_factorial_cuda_20261029_v1_consume_MAIN01.json \
 --whole-strict-sha 5d540a3bc6694501b74d97a52ded7aefd12ee2a70d9a46d75b1ddbb89325c456 \
 --output results/agent_p5_latest_whole_cellwise_20261029_v1_MAIN01.json

.venv/bin/python -m experiments.spectral_reachability.agent_p5_latest_whole_cellwise_20261029_v1 verify \
 --analysis results/agent_p5_latest_whole_cellwise_20261029_v1_MAIN01.json --analysis-sha ACTUAL_ANALYSIS_SHA
```

前置：完整原H1/H2/whole/latest目录、源/model/config与prepare全bindings、两个实际strict及上述外部SHA仍可读且未变。worker未执行实际analyze/verify、无生产输出/结论预填，最终结论等待MAIN。冻后修复只用新版本。
