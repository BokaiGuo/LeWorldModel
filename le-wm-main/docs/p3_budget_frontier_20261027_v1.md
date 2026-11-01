# P3 matched-root quality/cost frontier 独立重算

本模块只读真实保存 receipt，不运行 GPU、神经 CPU、physics、训练或下载。
既有 quality/cost v2 已覆盖基础审计；本版独立实现事件计数、合同匹配、
严格准入和反例回归，不把重算当新实验，也不修改任何旧 sealed 文件。

## 入口与输出

在 `/home/gbk/code/LeWorldModel/le-wm-main` 运行：

```bash
.venv/bin/python -m pytest -q tests/test_p3_budget_frontier_20261027_v1.py tests/test_p3_quality_cost_audit_20261027_v2.py
.venv/bin/python -m experiments.spectral_reachability.p3_budget_frontier_20261027_v1
```

唯一新目录 `results/p3_budget_frontier_20261027_v1/`，已存在即拒绝运行。
写出 report.json、report.md、源码/测试/本文快照及 SHA256 manifest；
读取前核验旧 v2 manifest 的完整 input closure 和 artifact 字节，
重新验证上游 live/archive source seals，写出前后再校验输入。
不加载 checkpoint 或 dataset；外部模型 hash 仍是上游绑定断言。
复用旧模块的严格 JSON、hash、只读 Evidence helper，不调用其 audit 核心。

## 重算和计费合同

- 完整 SHA root、method、repetition join，拒绝丢失、重复身份和外来根。
  两次相同 replay 全保留，但不是两个独立科学样本。
- 从每层每个一阶 kernel 的实际尝试重算 planning；空 depth=3 层不增加
  实际 horizon。单独计 selected-path independent verification，不额外把它
  再加到已包含 verification 的总 predict 中。
- 每 root-arm 共用 planning/verification 只计一次；实际物理尝试、返回、
  terminal encode 按全部16次执行分别计费，包括四次 guard abort。
- predictor、action encoder、processed context tokens、backward calls、VJP rows
  均是不同坐标，不相加为所谓等价 compute。VJP rows 不是 dense Jacobian 数。
  runtime no-grad 的四个 backward/VJP 坐标为已保存的0；未单独 instrument 的
  dense Jacobian、上游 compilation、reset/warmup、加载、source hashing、
  SVD FLOPs、总 FLOPs 记 null/unknown 或明确排除，绝不填0。
- 物理 journal 的 attempt/return 数及顺序、trace 和图像 hash 独立复核。
  原始 branch 对象、failure、failures、guard、全部重复和实际终态不删减。

## 比较及 frontier 的范围

同 root、任务/模型 binding、goal/projection、action bank、predictor context、
实际初始 state/history/goal/RGB 和反馈合同才允许比较。
反馈合同由已封存执行器源代码解释：预选 open-loop commands，每步 RGB appearance
guard；无 online replanning，evaluator-only state 不进入 controller。
guard 的实际观察/中止时刻是结果，不用于伪造另一份反馈合同。
该合同不是新 instrument 的独立 feedback receipt，source-derived 边界保留在报告中。

full-path 要求全部保存资源/时域/执行计数相等且两臂完整完成；
guard-partial 需同计数、同合同且两臂 guard 中止；其他一律 unmatched。
即使完整执行，相同总 predict、但 search H 不同，也不能叫 matched。
对 partial/unmatched 只保留带范围标记的终点差值，不进入 full-path frontier。

frontier 仅在同 root/repetition 的两个已保存 arm 上，以分坐标成本和两个真实任务
终距（pose cost、official state distance，均越小越好）作 Pareto 比较。
不跨 root 排序、不混合指标、不把两次 replay 池化为独立样本。
这是一个冻结 cap 的两个点，不是 budget sweep 曲线；等 saved counters 不等总 FLOPs。
总 pipeline 成本未知，所以总成本 frontier 永远不从此结果推导。

## 已有证据及确切缺口

应复现四根八 arm、16次执行、四次 guard abort、1152 planning +19 verification
=1171 predict samples，1171 action encoder samples、3513 processed context tokens、
154次物理 attempt/return、16次 terminal encoding。配对为2 full-path、
2 guard-partial、4 unmatched。full-path 仅一个根的两次依赖重放，路径相同且
quality/saved-counter Pareto tie；没有已证明 adaptive 优势。

严格 P3 gate 明确要求而当前缺少：P1通过、全 cohort full-path matched、
完整 pipeline 成本（或未来预注册且合法的排除合同）、独立 held-out 根、
prospective budget sweep、配对不确定性/多重性控制、真实 quality/cost 增益。
任务/反馈同合同检查本批可通过，但不能替代上述七项。
P3_admission=false；无新科学样本、无置信区间、无总算力相等结论。

反例仅在测试的内存拷贝中改值：计费重/漏算、空层、缺/重复receipt、错任务/模型、
改goal/projection/context/反馈、把abort改为full等；这些不是新实验观察。
