# P3 固定五对 U/Z2：预测 → 真实 RGB → native 全轴

仅 NEW 本 stem 的 source/tests/doc。只实现 RAM/临时盘 saved 分析；MAIN 审读后实际 build/verify。
不改旧 result_protocol、scalar summary、RGB bridge、selector、科学门或结果。无 NN/AD/physics/fullstrict/子代理。
experiment-code 技能只用于固定分析、日志与失败回归，不采用训练/拟合/择优/新阈值或绘图。

## 固定来源与接线

所有 CLI 路径 canonical absolute，显式全文件 SHA，不接受 future/placeholder authority。

|输入|MAIN 实际 SHA256|
|---|---|
|RGB prepare|25037ca2cbcef58df997cf9f3fd52c2c8119fc329745e45e9d4f25f20cb608e2|
|RGB report|017a5fe3b47a418435bb7e86f4812558c43548da8a1a06fcc69e0ab2ddbc717d|
|RGB completion|90960e63603353cc205c0e838a08cb2866e6d97beb788ed6a2a3a743271816ec|
|RGB receipt|946ca6cc017fc66591920eca7b7d1d42bc07e8d9ebf169c0b67ecfb0b8c30cc9|
|RGB source|aa502fd6650cb90318bcca748349cca301c42ad9c19ffad3073255bd35859067|
|原 scalar summary|e59eeadc6e6de711e60e50709162c4310ce0e69e9b3fdb8a5db838fd74797b19|
|原 capability summary|88587a2d9f610df775c321eda6497bf3602c6ccd993b8779d7f36c6ad4744a4d|
|Stage B receipt|682a78366b10a60895c4530f2fc6cc401e6439b625f4ba0a8aa996dc2fed79a7|

Pauli `.consume` 的 ABI 来自冻结 source：rows/tasks 十槽、report_sha256/completion_sha256、source_authorities 三个物理来源，
没有 prepared 字段。prepare 路径由 report/completion 与 CLI 的 prepare 指针连接，不假造 receipt 字段。
原 capability summary 无 equal_total_compute 字段；按其实际 producer flags 明确检查，不使用缺省 false。
所有新输出 equal_total_compute=false，不能把共同模型/已付费 call 数解释成公平预算结论。
无可用 agent 消息工具；通过冻结源码及字段 metadata 对齐 ABI，并向 MAIN 报告，而非伪称直接联络成功。

仅复用已冻结 scalar summary 的文件/hash/rawbits 通用 IO 工具，不调用其 build/verify，也不调用任何 producer 算术。
新 latent/native 数学本地独立实现，NumPy 只用于保存数组、原精度 token 算术及单独标注的 NumPy Q 对照。
全文件 SHA、完整十 rows/report、tasks 原文件和所有实际 latent NPY 复核；不重读 PNG/encoder full IO/journal 或模型权重。
这些事实通过 MAIN full receipt 继承，非新的物理/神经 fullstrict。完整资源向量原样保留，不只数 predict。

## 分母、context 与数学

十槽顺序固定 root0 C/Z/P、root1 Z/P，各 U/Z2；固定 control15/absolute25。
每 endpoint 的 U 和 Z2 必须具有相同 cp10 task/context/goal/model/normalizer。
U 取 Stage B whole[0]（与独立 segmented 完全相同）；Z2 取该 endpoint 的原 nominal_context（原零 raw5 的 forward）。
检查全部三帧 shifted latent context、两 action tokens，实际 raw15 为原 causal10+U5 或 float32 正零5。
真实 cp15 task 由原 cp10 context 后两帧 + 实际 NPY latent 及 control6..15 原 public normalizer 计算两 tokens，完整字段认证。
不把某角色的预测与另一角色/context 预测相减，不制造跨 policy 模型收益：输出 cross_context_model_comparisons=[]。

每分支保存原 goal192/predicted192/actual192，逐元素 float64 差，独立 math.fsum：

`Qreal = Qpred + mean(error²) + 2 mean((pred−goal)·error) + roundoff`。

原 identity192 NumPy Q 与 Stage B 已存 Q 精确核对，fsum/NumPy 差另栏；不以新归约替换原 Q。
每对固定 model_gain=Qpred(Z2)−Qpred(U)、real_latent_gain=Qreal(Z2)−Qreal(U)，正值均偏 U。
native reward-distance 同方向，以及 native7/XY/agentXY/objectXY/velocity/raw-angle/wrapped-angle 全轴分别同方向差。
native7²=jointXY²+raw_angle²+velocity² 的舍入残差保留，不挑指标、不合并不同单位成新分数。
原 native success 单列，不能用 Q/distance/velocity 推翻成功标签；历史 native angle bug 与 diagnostic wrapped angle 并列。
符号直接 >/< /== 判定，不用乘积，ties/null 分开。每轴 sign alignment 都保留，不因已知 metric mismatch 筛选。

所有五对与十槽保留；缺 cp15/早停 actual=null，不能用 terminal 代替15，模型预测仍保留。
原 scalar terminal 与全部八 C-control /四 logical C 原样继承，不新增跨 context 模型差；原即时代价 false 不被延迟表现覆盖。
unknown FLOPs 保持 null，known roots=2/fresh roots=0。三层符号一致不保证因果中介、局部 SVD 可兑现或 P1/P3/C3 通过。

## 运行边界与测试

输出及单个 JSON 上限256MiB（复用已冻结 IO 限额）；NPY 只许192 float32、完整 literal bit SHA、原 NPY1 128-byte header，
gzip 限制读897字节以拒超出896-byte 小向量。全 source/data 前后核 SHA；exclusive write、文件/目录 fsync；不覆盖已有结果。
whole verify 仅重建本 saved 分析，不递归运行旧 fullstrict。RAM fixture 标记 fixture_only，不可通过生产 CLI 实际 pins。

最终联合 RAM 回归：114 passed / 22.73s（新43项、旧 scalar/protocol 71项）。包含临时盘全10 build/write/verify、
全空/单边缺失保持全分母、source/完整 receipt 篡改、重封外层 SHA 后的 context/raw/snapshot/NPY 位级拒绝、
原四 logical C/八对照/三 no_witness 与 immediate false 不可篡改，以及固定 XY 改善但速度/native 总距受损样例。
本轮无失败测试；仅检查实际文件的 schema 字段名，不运行实际 build、神经/物理或任何上游 fullstrict。
剩余仅 MAIN 独立审测与实际 saved build/verify。三件套正式冻结后不再改写，以免 sourceclosure 漂移。

## MAIN CLI

从 `/home/gbk/code/LeWorldModel/le-wm-main` 执行（仅 MAIN）：

```bash
.venv/bin/python -m experiments.spectral_reachability.agent_p3_capability_three_layer_summary_20261029_v1 build \
 --rgb-prepare /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_same_time_cp15_rgb_bridge_20261029_v1_prepare_MAIN01.json \
 --rgb-prepare-sha 25037ca2cbcef58df997cf9f3fd52c2c8119fc329745e45e9d4f25f20cb608e2 \
 --rgb-report /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_same_time_cp15_rgb_bridge_20261029_v1_cuda_MAIN01/report.json \
 --rgb-report-sha 017a5fe3b47a418435bb7e86f4812558c43548da8a1a06fcc69e0ab2ddbc717d \
 --rgb-completion /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_same_time_cp15_rgb_bridge_20261029_v1_cuda_MAIN01/completion.json \
 --rgb-completion-sha 90960e63603353cc205c0e838a08cb2866e6d97beb788ed6a2a3a743271816ec \
 --rgb-receipt /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_same_time_cp15_rgb_bridge_20261029_v1_consume_MAIN01.json \
 --rgb-receipt-sha 946ca6cc017fc66591920eca7b7d1d42bc07e8d9ebf169c0b67ecfb0b8c30cc9 \
 --rgb-source /home/gbk/code/LeWorldModel/le-wm-main/experiments/spectral_reachability/agent_p3_same_time_cp15_rgb_bridge_20261029_v1.py \
 --rgb-source-sha aa502fd6650cb90318bcca748349cca301c42ad9c19ffad3073255bd35859067 \
 --physical-summary /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_capability_probe_result_summary_20261029_v1_MAIN01.json \
 --physical-summary-sha e59eeadc6e6de711e60e50709162c4310ce0e69e9b3fdb8a5db838fd74797b19 \
 --capability-summary /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_capability_signal_summary_20261029_v1_MAIN01.json \
 --capability-summary-sha 88587a2d9f610df775c321eda6497bf3602c6ccd993b8779d7f36c6ad4744a4d \
 --stageB-receipt /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_same_time_capability_probe_cuda_20261029_v1_consume_MAIN01.json \
 --stageB-receipt-sha 682a78366b10a60895c4530f2fc6cc401e6439b625f4ba0a8aa996dc2fed79a7 \
 --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_capability_three_layer_summary_20261029_v1_MAIN01.json
.venv/bin/python -m experiments.spectral_reachability.agent_p3_capability_three_layer_summary_20261029_v1 verify \
 --summary /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_capability_three_layer_summary_20261029_v1_MAIN01.json \
 --summary-sha <MAIN_build_实际SHA>
.venv/bin/python -m pytest -q tests/test_agent_p3_capability_three_layer_summary_20261029_v1.py tests/test_agent_p3_capability_probe_result_summary_20261029_v1.py tests/test_agent_p3_capability_probe_result_protocol_20261029_v1.py
```
