# Early-history factorial：656全四cell新运行 v1

本独占source/tests/innerdoc，继承已封存协议`innerdocs/agent_p5_next_identifiability_protocol_20261029_v1.md` SHA `5e80815ceceb1599ee88226dc58dd80965f59d6952ff7d1cef7c62f7fe903f0c`。MAIN在本轮任何新运行之前明确选择656方案；不是492结果后追加RR。本worker只实现和RAM/临时synthetic文件测试，无生产prepare/run/consume、NN/physics/网络。

## 固定问题、分母、时间及共同标签

固定原164枝、80 signed pairs、四root×off/on层（各41枝/20对）；原zero/null全保留。新每枝AA/AR/RA/RR各一真实T3调用，合计656 API=656 batch samples=1968 frame samples。A/R分别指early5/10的actual/causal-rolled；visual为第一个字母，proprio为第二个字母。**所有cell的最新15槽均固定short pred15**，原packet indices[1,2,3]（raw6..10/11..15/16..20）全buffer相同，不回滚/重新归一化。

actual5/10直接取old H2第二predictor真实input前两槽；rolled5/10/15取short末次真实inputs，并逐位核其对应short前三次真正causal最后输出槽。禁止用旧full3早输出伪装pred5/10、零padding、重编码或把raw1step当packet。四cell最新槽均固定short生成路径，不以old H2 pred15替代。

RR是本job新预测；所有三个native inputs及完整full3 visual、None、full3 proprio与old short末次native IO逐位比较，含shape/dtype，无allclose/tolerance。任一不等保partial、未解label，不重试。其余cells仍属model-internal拼接，不认证物理可达性。

目标固定共同H2 batch2实际15/20之slot1，保留完整batch2副本且仅在全部656真实full3预测完成、RR全核通过、model前后核通过并durable seal后才读取数值。旧label文件可以被SHA/size/inventory检查，但prepare/verify不np.load这些未来label。输入actual5/10属于过去context，不是未来label。run中所有paid native IO先保存在输出目录。

## 来源闭包及验证边界

ROOT=`/home/gbk/code/LeWorldModel/le-wm-main`。固定上游：

- short actual report `results/agent_p5_short_start_full_roll_20261029_v1_cuda_MAIN01/report.json` SHA `1b014fc5c2e9868ee5e08e0dc05b602b5e9708a342c9c842698183f3430297c4`；完整strict `results/agent_p5_short_start_full_roll_20261029_v1_consume_MAIN01.json` SHA `3a20eb780028ddd40a88d11c59b8ada9dcbc0c7147bf1df9aa9acde332aa26fd`。
- H2 report `results/agent_p5_raw164_jepa_h2_cuda_20261029_v2_cuda_MAIN01/report.json` SHA `e5a561215bfe7b6f1c1c224e7661211e3f237fb16458a6a409f5a1da0fc08579`；完整strict `results/agent_p5_raw164_jepa_h2_cuda_20261029_v2_consume_MAIN01.json` SHA `783b65f67899791b5697bfe0819ccfb3231d57212b9e3e85aefc02419459c0ea`。

`prepare/build`正常复用冻结comparison的`bundle`（source SHA `45c1b957f60b0e9be1b731c88b38c30e60ba5c55502e62969ff205fafc770f1e`），不是其analyze/verify_analysis；只取short/H2，不加载H1 outcome或挑stratum。该函数核exact报告/receipt SHA、schema/status/164/80/counters/model-before/after、原prepare/source全bindings、完整artifact SHA/size目录清单、规范journal hashchain、native descriptor样本/IO算术。再由本build逐branch/task/selection/root/condition/rawhistory/normalizer/packets以及具体native指针交叉核验，读取过去/预测inputs确认真正early差异与causal来源。

**不调用旧verify/consume/fullstrict，不重建上游全部物理/encoder/模型语义**；完整上游语义证据继承其已授权持久strict，而不是把本层metadata核查说成重新fullstrict。新verify确定性重建本prepare，仍不嵌套fullstrict；新consume完整验证本job所有native IO、chronology、RR、label、算术与计账。不回证未保存的历史加载字节。

source refs保directory/原descriptor（file/SHA/shape/dtype/nbytes/file_bytes）及report JSON pointer，不按filename猜身份。short `/calls/{4*i+3}/inputs/$tuple/0/$tuple/{0,1,2}/$array`为RR输入；H2同位置为A来源；short `/calls/{4*i+j}/outputs/$tuple/{0,2}/$array` j=0,1,2为causal-last-token证据；H2 `/rows/{i}/label_patch`须与其真实label encoder call和short authority exact链接。全部未来label只核metadata/SHA，seal后才decode。

prepare自有input census逐枝记录early visual/proprio equal标志；相同枝保留，整个因素全164相同则fail closed不可称新对照。不要求完整overlay分割或按coverage选mask。继承source/model/config闭包与两个完整strict成本单列；当前loader通过同buffer state/load receipts、strict schema及model前后哈希再次核模型，imported source必须属于准备bindings。不训练/拟合/新根/改Q/改门槛。

## 资源、失败及后端

每call full native input/output有效tensor=2,457,720B，656共1,612,264,320B；共同batch2标签副本128,974,848B；总1,741,239,168B<2GiB。prepare只存pointer/metadata，无重复输入bank。输出cap3GiB包含prepare实际bytes、NPY headers/journal/report；tensor cap2GiB；单array4MiB/metadata32MiB；900s含verify/load/dispatch/终态哈希，floor200MiB、reserve64MiB。caps不足即partial/拒绝，不丢full3/降精度/缩分母。RSS/GPU峰值与actual native IO分轴，旧short/H2成本继承分栏，不与新656混为新调用，不称等FLOPs。

继承已审probe/full-roll backend，不修改globals。自有Meter为4×164个T3，旧T1/T2/T3 guard完全不动。所有loaded model均装hooks；只有predictor可dispatch，encoder/proprio_encoder调用、CPU、grad、TF32、错误shape/device/dtype及额外调用均拒绝。实际observed fused attention按原TorchDispatch guard拒绝，requested MATH不是实际MATH kernel证书；没有CPU fallback。

run持久记录dispatch intent→inputs→observed return→validated full outputs；attempted/returned/validated分列。branch/cell由完整固定branch-major顺序归属，history_ready带input rawbytes摘要。任何异常含KeyboardInterrupt保partial与全部164/80分母；partial consumer仅证inventory/hashchain，非科学负结果或完整语义通过。所有source/model前后不变；预测seal前没有未来label数值。工程退出不开放P1/P5。

## 计分、输出及strict

report含protocol/execution_binding/rows[164]/calls[656]/pairs[80]/summaries[5]/counts/inherited/model/attention_ops/artifacts/journal_chain_tail。每row predictions[cell]保存完整`$tuple` native输出树（不仅finalvisual），RR_verified含3input+完整visual/None/proprio等值旗标，label为本次共同batch2副本，score为全部4cell final20 MSE/效应。seal保全部656 full-output树及RR证据。

float64 C-order全256patch×384 MSE，不改metric。visual=`((RA-AA)+(RR-AR))/2`；proprio=`((AR-AA)+(RR-RA))/2`；interaction=`RR-RA-AR+AA`；joint=`RR-AA`。每pair全4cell预测/真实response L2/dot/cosine/error、/2h central差分；任一范数exact0→null、无epsilon。全体及四层都保positive/negative/zero/null与六个cell-pair符号转移矩阵。向量效应只报L2，cosine不假作线性可加。实际56/24若仍一致由共同label决定，不硬编码胜出子集或删zero。

consume验证整个inventory/NPY/no trailing、完整事件逐字段/顺序、每call descriptor与实际数组、所有选择输入/原动作/最新固定、RR全部native字节、seal-before-label、共同label字节、每branch MSE/效应、80pair和四层汇总、tensor/cap/counters/终态source闭包。receipt status=`all164_early_history_four_NEW_cells_and80_pairs_verified`，仍P1/P5 false。结果未知，不预填效果；early block影响不等同单帧/物理因果、泛化或控制收益。

## MAIN执行CLI（仅MAIN生产）

在ROOT执行；PREP_SHA/REPORT_SHA取各命令实际返回，非未来常量：

```bash
M=experiments.spectral_reachability.agent_p5_early_history_factorial_20261029_v1
P=results/agent_p5_early_history_factorial_20261029_v1_prepare_MAIN01.json
O=results/agent_p5_early_history_factorial_20261029_v1_cuda_MAIN01
.venv/bin/python -m "$M" prepare --output "$P"
.venv/bin/python -m "$M" verify --prepared "$P" --prepare-sha PREP_SHA
.venv/bin/python -m "$M" run --prepared "$P" --prepare-sha PREP_SHA --output "$O" \
  --authorization MAIN_P5_EARLY_HISTORY_FACTORIAL_HOSTCUDA_ONLY
.venv/bin/python -m "$M" consume --output "$O" --report-sha REPORT_SHA \
  --receipt results/agent_p5_early_history_factorial_20261029_v1_consume_MAIN01.json
.venv/bin/python -m pytest -q tests/test_agent_p5_early_history_factorial_20261029_v1.py
```

新测试仅RAM/synthetic native-shaped temp NPY+fake tensors，实际Store/hook/journal/ownconsume；临时N=1只作用本NEW模块，CLI强制164，旧源globals不动。fixture receipt不是科学资产；不把tests通过冒称HOSTCUDA/实际RR通过。尚需MAIN审读、真实prepare、CUDA及新strict；上游全部原始目录/source/model/config/持久strict缺失或变化即拒绝。
