# P3 K17 六槽：规划账本与公平计算边界

## 结论

现有原始规划事件可以按**相同的 27 个已定义 hook 资源轴**独立重算六槽的整树尝试及完成计数；它们共享轴上限，但实际开销向量不相等。物理侧六槽接口调用数相同，与规划计算不在同一轴，不能用来宣称总预算相同。这是一项已保存事件的工程描述，不是等 FLOPs、等计算、实时公平调度或科学效果证明。

重建源为 `agent_p3_c3_compact_host_cuda_20261029_v1_cuda_MAIN01`，继承固定外部 MAIN receipt SHA256 `cdd61fc20ff4b819e64092f226cbe84ce3777edfc2a2c8472322a01ec3a0601f`；保存 handoff SHA `3b2caf555f390426234b34a96d8bd84670933da5364eba4a1f22df9a424a6df9`、物理配对分析 SHA `0a8fa2c7a822aa702f5a5ed1bbbe6d168dfd24deddc33feaa251da39ba09debb`。独占结果 `results/p3_k17_compute_ledger_codex_20261029_MAIN01.json` SHA256 `48cefdfd9ec44a6ab2d215a539452ae6bbc2f3829e3a794dcf7e033809abc90f`。

每槽读取受 manifest/completion 和 report 绑定的原始树，按 event resource、shape 和 VJP directions 独立映射为 27 维增量；区分 returned、failed、blocked-before-dispatch；校验顺序、cap、计数、scope、inclusive 时间，并核对原树 receipt 和上游严格 receipt 的事件语义 hash 与计数。不是复制之前的预算汇总。计划的 task/plan hash 与既有 handoff 相符，前五步原始 f32 位串与物理配对结果相符。物理侧来源与先前配对输入的验证被继承，不在本次重做 native 来源认证；上游不排除未被 hook 观测的隐藏调用。

## 六槽真实已记录规划

| root / slot | 方法 | 整计划动作步数 | predict API | VJP directions | Jacobian API | replay calls | 整树墙钟秒 |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 / 3 | fixedH1 | 15 | 114 | 3456 | 18 | 2 | 84.965 |
| 0 / 4 | staticH3_progress | 45 | 198 | 3456 | 18 | 2 | 88.770 |
| 0 / 5 | adaptive | 45 | 446 | 11904 | 62 | 74 | 283.358 |
| 1 / 9 | fixedH1 | 15 | 114 | 3456 | 18 | 2 | 84.271 |
| 1 / 10 | staticH3_progress | 45 | 198 | 3456 | 18 | 2 | 89.926 |
| 1 / 11 | adaptive | 45 | 278 | 4992 | 26 | 2 | 130.660 |

每槽尝试数与完成数相等；整树账本分别包含 10979、11315、38011、10979、11315、16311 个事件。root 0 的 adaptive 相对 staticH3 多 248 次 predict API、8448 个 VJP directions、72 次 replay；其 replay 中 72 次属于 program phase、2 次属于 verification phase。root 1 的 adaptive 相对 staticH3 多 80 次 predict API、1536 个 VJP directions，而 replay 均为 2；虽这两槽真实物理前五步动作逐位相同，不能据此认为其规划计算相同。每对逐轴 B−A 和完整 attempted/completed/cap 向量、四种 phase 分解均保存在 JSON。

表中为整个已生成计划树以及相关验证的历史账本，计划长度 15 或 45，而物理只执行了各自前 5 步。该账本不提供独立的、以这五步作为计费范围的在线重规划计算，因此不能除以步数推算五步成本，也不能将完整树的更长计划视为五步之后实际执行。各事件计数是同种 hook 指标上的比较：predict、action encoder、predictor、projection 及其 sample/context 计数存在嵌套或重叠，不能相加成总运算量。inclusive operation 时间亦有嵌套，不可相加；整树墙钟是保存的主机计时观察，并非同步后的 CUDA 时间或严格可重复的性能基准。

## 缺失与可执行下一步

FLOPs、CUDA 时间、backward engine 实际调用数未测量，保持 null；hook 自身无法证明无未计账调用。没有受认证的模型计算与物理计算统一标尺、五步限定的增量计算账、各方法物理阶段单独墙钟，也没有保证等实际计算的比较设计。此前物理采集的 provenance 标记仍为 false；只有两个已知 root，不能解释为独立泛化。不要用全局 outer 墙钟、相同物理步数、共用上限或相同首五动作替代缺失预算证据。

若要推进公平计算比较，需要事先冻结每个方法的可比工作单位和成本边界，区分整树预计算与真实同时间决策；对每个候选决策记录实际尝试/完成、batch/context/方向、完整计时与缺失，安排相同资源轴的预设匹配或明确的预算前沿，并独立认证钩子覆盖及实际来源。是否需要新的计算或物理采集须另行授权，本分析没有启动任何此类运行。

## 复核与测试

在 `le-wm-main` 下：

```sh
env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m experiments.spectral_reachability.p3_k17_compute_ledger_codex_20261029 results/p3_k17_compute_ledger_codex_20261029_MAIN01.json --verify --sha256 48cefdfd9ec44a6ab2d215a539452ae6bbc2f3829e3a794dcf7e033809abc90f
env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_p3_k17_compute_ledger_codex_20261029.py
```

verify 重读原始 sealed transport 并逐事件重算，不只检查输出自哈希。测试中的合成事件验证计数公式、错误/拒绝与篡改拒绝；它们不是真实运行证据。真实账本重建和输出 verify 均不运行 GPU、训练或物理。
