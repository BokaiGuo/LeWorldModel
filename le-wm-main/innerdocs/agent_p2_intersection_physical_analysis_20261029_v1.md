# P2 intersection 六槽 fixed-denominator 实际末端分析

在观察新物理结果前固定：两 known off roots、六槽按原顺序全部保留；逐槽输出实际最后一次 labelled return 的 step、native success、原 native7 状态/goal/residual、前四维 joint XY L2、agent/block XY、native angle term、wrapped angle（仅诊断）、agent velocity、原包含速度与未 wrap 角度的 native7 距离和 reward。复用 frozen common-resource saved-scalar 函数算术，不冒称第二独立数学实现。没有 labelled endpoint 为 null，不能当失败、零收益或补填计划末步。

提前 terminal/truncate 的 endpoint 按真实返回步数报告，不冒充 15 步同时间对照；保留 native stop reason、缺 checkpoint、工程错误和实际 attempted/returned。唯一汇总是 success 的 fixed6/known/unknown 计数，不排序选优、不跨根当独立 fresh 样本，不推断 P1/P2、物理收益或 equal FLOPs。

physical receipt 的全局计数及与其绑定的 report 的完整 operation receipts 逐槽重计 attempted/completed/failed，与总账及原 cap 严格相等；各槽保留上游原 planner 完整账户，绝不把全局物理计数平均分配。物理操作嵌套不相加为总 FLOPs；未观察成本保持 unknown。若 constructor 内部发生未被 hooks 捕获的工作，不宣称成本完整观测。

要求 MAIN 已完整认证的 physical strict receipt、其实际 SHA，以及由 receipt bindings 认证的原 report 和 SHA；校验两者完整六槽内容、身份、账户和资源一致，继承 upstream strict 认证，不重复 full consume、不加载模型或执行环境。凭自行捏造的 receipt SHA 不能建立 authority；由 MAIN 提供权威 receipt。可以分析 authenticated partial_or_failed，不能据此认证执行成功。

```
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p2_intersection_physical_analysis_20261029_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_intersection_physical_analysis_20261029_v1 analyze --report REPORT --report-sha REPORT_SHA --receipt STRICT_RECEIPT --receipt-sha STRICT_SHA --analysis /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_intersection_physical_analysis_20261029_v1_MAIN01.json
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_intersection_physical_analysis_20261029_v1 verify --report REPORT --report-sha REPORT_SHA --receipt STRICT_RECEIPT --receipt-sha STRICT_SHA --analysis /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_intersection_physical_analysis_20261029_v1_MAIN01.json --analysis-sha ANALYSIS_SHA
```

`verify` 整份重建本分析但不重复 upstream strict。新分析≤16MiB、floor200MiB、独占输出；无生产输入则仅 RAM/fake 环境测试。原 native success 固定 joint XY<20 且原 angle term<pi/9；native angle term 可为负，不能改写公式或把 wrapped diagnostic 代替成功条件。
