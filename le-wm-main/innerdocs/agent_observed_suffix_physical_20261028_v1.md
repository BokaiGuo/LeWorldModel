# 真实观察后 suffix 重优化：独立原生物理 adapter v1

独占三件套：`experiments/spectral_reachability/agent_observed_suffix_physical_20261028_v1.py`、同 stem `tests/test_*.py`、本文。只新增，不修改任何已封存模型、原 runner、结果、阈值或 gate。终态 RGB companion 由 Bohr 独占，本 adapter 不实现它。使用 TDD 技能先写 RED，再执行纯 RAM/native-schema 与保存资产读取测试；worker 未运行生产 prepare、环境、物理或任何神经 CPU/GPU。

## 实际输入、真实 schema 与复用接缝

上游是 Nash 冻结 `agent_nonlinear_suffix_observed_recompile_20261028_v2`，不是旧 all168 report。prepare 接收外部实际 `compile_output`、`compile_report_sha`、`compile_receipt`、`compile_receipt_sha`、`compile_source_sha`。无猜测未来 SHA/default report。先校验外部 source/report/receipt SHA，再调用真实 `compiler.consume`（含其完整 verify/build→midpoint→terminal→原 physical 严格消费），完整比较持久化 receipt，不以 status 代替校验。通过 actual report 的 prepared/prepare_sha 找实际 prepare，绑定其全部 arrays/ledger/artifacts/source/model/installed classes；所有依赖来源及本三件套 SHA 最终复核。

只读 ABI 回归使用 MAIN 已生成的实际产物：

- compile source SHA `e66bac861fce455ab6bb7a0cf12ba1644258c6c2e9cb6f2905973ac146e33567`。
- prepare `results/agent_nonlinear_suffix_observed_recompile_20261028_v2_prepare_MAIN01.json` SHA `9aa3d63db85857b80c8a614a00a5a7b22c77c26e688e149970a83825c0d74411`。
- report `results/agent_nonlinear_suffix_observed_recompile_20261028_v2_cuda_MAIN01/report.json` SHA `3de58e380acbef66269e69150cab7bca312604b7df88a0f1ae5663764d691b45`。
- receipt `results/agent_nonlinear_suffix_observed_recompile_20261028_v2_consume_MAIN01.json` SHA `09d617dbece12ce3ff2ca7246390e433454b1d8ac29042ab07eadd2be75238ec`。

`assemble` 严格联结四个 `c.rows[].request`、四个 report sealed rows、真实 consume `plans`。保持原完整 task（binding/root context/goal192/identity Q/mean/std）、模型身份和原 prefix5；只采用新模型选定 suffix5。新原生 plan 使用本 adapter `.execution_plan` schema，同时完整保留真实 `.plan` 于 `compiler_plan`。旧原预测 Q、observed incumbent Q、新 suffix Q 分栏，不覆盖。**own schema 保持 `digest=mixed.core.digest`**；仅 producer chosen 的 `selection_digest`、`goal_sha256`、`observed_context_sha256` 通过显式 `producer_digest=compiler.digest` 核验。其余 producer parent/model/midpoint 哈希与已认证的原字段直接相等比较，不用 own 编码重建。原 native journal 的 event hash 仍按原 `source.digest` 验证。RAM fixture 生成 compiler plan/goal/context 哈希也使用 compiler.digest，而非 m.digest；回归证明普通 `[1,2]` 的两种编码哈希确实不同，并拒绝逐项混用。

原 physical authority 从完整绑定链的 midpoint report→midpoint prepare→terminal report→terminal prepare→原 physical report/prepare/receipt 逐项取出，所有路径/SHA 必须已在 compiler bindings 中。不靠 glob、猜目录或 outcome 选 root。原 strict 的全部行已由真实上游 consume 校验，本工具只提取 prefix51/cp5 作身份参照；绝不把旧 row/outcome 复制成新执行。缺原完整 cp5 authority 拒绝 prepare。

新 driver 自有 execute/report/validate_saved/consume，不调用旧 execute 来伪装 schema，不改旧 globals、不使用 FunctionType/global patch。复用的是原 `mixed.episode`、public dispatch、native hooks、placement/goal binding、renderer/guard、Meter 底层计费和 flat durable writer。四条均 constructor 新 env，从原 seed/placement/warmup10 运行完整选定 raw10，无继承模式。执行过的 raw10 不重算 tanh/clip/scale；原 native 内部 action scale100/relative 等语义保持不变。模型完全不加载。

这是用既有真实 cp5 观察编译新 suffix 后，再在四个 fresh env 精确重放 prefix 的评估；不是声称在本次 env 内调用 CUDA 优化器。前置身份 guard 证明新 suffix 接收的物理/RGB 历史与生成它所依据的原 cp5 相同。

## cp5 因果 guard（冻结前 MAIN 要求的局部修复）

旧版草稿只在 episode 结束 finally 核 prefix，不能保证先认证再执行 suffix。现在局部 `class Meter(base.Meter)` 在 **plan public_step6 的 dispatch intent 之前**：

1. 确认本 branch 五个 plan public steps 已按1..5全部返回。
2. `prefix_match` 核所有51帧（warmup0..10，plan0..5，off/on/off）的 phase/step/condition/overlay、PNG bytes SHA、pixel SHA、state、完整 snapshot、pre/post、renderer、goal；另核 cp5 canonical checkpoint。
3. 写入 fsync/hashchain 的 `cp5_guard`，附 prefix-reference、实际 prefix51、实际 cp5 digest；仅 accepted=true 才交给原 Meter 派发 step6。失配记 accepted=false/error，**没有 step6 intent、没有 suffix native callback**。finally 再核身份，不替代前置 guard。

strict 将 `cp5_guard` 加入精确允许的事件集，核唯一性、分支/证明 digest、51帧和五个返回均早于 guard、guard 早于全部 suffix intents；拒绝移位、缺失、重复或伪造 accepted 标志。正常 native stop 发生于 step5 时可无 guard，不补 cp10、不强行执行 suffix。缺帧/guard 工程失败保留原始 partial，不发完整成功认证。

相较 MAIN 首次预读的源码，source 增量限于：原 Meter alias 换局部 guard subclass；execute 设置 `guard_context`；validate_saved 允许并调用 `audit_guard`；以及实际 ABI 测试发现后的逐 schema 哈希修复（own digest 不变，增加 producer_digest，仅三类 chosen 字段改用它）。曾临时整体切换 digest，最终已按 MAIN 要求恢复 own 编码，未生产 prepare/执行任何物理。无上游改动。完整新增段可直接按 `class Meter` / `def audit_guard` / `guard_context` / `audit_guard(r, events, p)` 定位审读。

## strict、原任务轴和资源

完整校验四条固定顺序/known2 root 绑定、原始 placement 和 warmup source identity；所有 allowed events/hashchain、嵌套 operation intent/return/failure、public raw bit、连续步骤、off/on/off、完整轨迹和 checkpoints。每个 warmup/plan `returned_transition` 都独立以相应 canonical 目标核 native7 state/snapshot/reward/terminated/truncated：warmup 使用源 placement 的 native_goal，最终任务使用真实 goal_attached 的原目标，并核 binding chronology。不是按 `(goal,state)` 查表；反复静止 state 不会跨 phase 混用。

保留原 native success：joint `||state[:4]-goal[:4]||<20` **且**原 native angle term `<pi/9`。native7 reward 是含速度的负7维欧氏距离；jointXY、angle、velocity、native distance 均按原 `mixed.metrics` 输出，wrapped angle 只作诊断。0/4 success、原生 early stop、无改善或损害全部保留，绝不重选、调 Q、改阈值或签 P1。科学/freshness/independent_confirmation 均 false；fresh env 不等于 fresh independent roots。

完整 horizon 的新实际预算沿原 physical：constructor/close各4、reset8、set_state16、goal_render8、goal_setter12、goal_bind4、public_step80、physics_step816、explicit_render264。计费是原真实 native 回调嵌套 API，各轴不相加为独立样本/FLOPs。guard 仅写事件，不新增神经/物理调用。early stop 不补量，attempted/completed/error/cleanup 原样入账。上游72 forward/32 scalar reverse及既有全部来源成本保留于 planner_authority；本 driver 新神经调用0。

输出仍沿原物理合同32MiB、final reserve4MiB、disk floor200MiB、wall900s。**producer report/receipt/preparation 与本 prepare/receipt 元数据读取/写入使用128MiB cap**，依据原 compiler 的128MiB输出合同，不用8MiB小 cap 截断/简化完整元数据。native output cap 与 metadata cap 分开，不擅改原物理预算。所有264张 actual PNG 独立验 file SHA、RGB224/PNG/mode、pixel SHA、原 `moving.guard`；额外 artifact、篡改或缺 PNG 拒绝。partial/fsync日志保留，无重试/覆写/补帧。

## MAIN CLI

在 `/home/gbk/code/LeWorldModel/le-wm-main`；必须使用尚不存在的自身输出名。prepare/verify/consume 全为 saved-only；**只有 MAIN 的 run 创建环境并执行物理**。实际 prepare/report SHA 用各步 CLI 返回值替换占位符。

```bash
.venv/bin/python -m pytest -q tests/test_agent_observed_suffix_physical_20261028_v1.py
.venv/bin/python -m experiments.spectral_reachability.agent_observed_suffix_physical_20261028_v1 prepare --compile-output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_nonlinear_suffix_observed_recompile_20261028_v2_cuda_MAIN01 --compile-report-sha 3de58e380acbef66269e69150cab7bca312604b7df88a0f1ae5663764d691b45 --compile-receipt /home/gbk/code/LeWorldModel/le-wm-main/results/agent_nonlinear_suffix_observed_recompile_20261028_v2_consume_MAIN01.json --compile-receipt-sha 09d617dbece12ce3ff2ca7246390e433454b1d8ac29042ab07eadd2be75238ec --compile-source-sha e66bac861fce455ab6bb7a0cf12ba1644258c6c2e9cb6f2905973ac146e33567 --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_observed_suffix_physical_20261028_v1_prepare_MAIN01.json
.venv/bin/python -m experiments.spectral_reachability.agent_observed_suffix_physical_20261028_v1 verify --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_observed_suffix_physical_20261028_v1_prepare_MAIN01.json --prepare-sha ACTUAL_PREPARE_SHA
.venv/bin/python -m experiments.spectral_reachability.agent_observed_suffix_physical_20261028_v1 run --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_observed_suffix_physical_20261028_v1_prepare_MAIN01.json --prepare-sha ACTUAL_PREPARE_SHA --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_observed_suffix_physical_20261028_v1_MAIN01 --authorization MAIN_OBSERVED_SUFFIX4_RAW10_PHYSICS_ONLY
.venv/bin/python -m experiments.spectral_reachability.agent_observed_suffix_physical_20261028_v1 consume --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_observed_suffix_physical_20261028_v1_MAIN01 --report-sha ACTUAL_PHYSICAL_REPORT_SHA --receipt /home/gbk/code/LeWorldModel/le-wm-main/results/agent_observed_suffix_physical_20261028_v1_consume_MAIN01.json
```

交 Bohr 的真实接口：`consume(output, actualReportSHA)` 返回本 stem `.consume`，`rows` 是新运行四行（无继承 wrapper）；`row.plan` 为本 `.execution_plan`，内有完整 `model_task`、`compiler_plan`、literal raw10。`row.checkpoints` 含 cp0/5/10 的 png/file_sha256/pixel_sha256/state/snapshot，PNG 位于该 output flat 目录；缺失由 missing_checkpoints 表示。`planner_authority`、完整 bindings 及 `physical_report_sha256` 在 receipt 中；真实 report 使用 `execution_binding.prepared/prepare_sha256`。无假旧 all168 或 terminal-witness schema。

## 开发验证边界

只读实际 schema 接线测试不重复生产 consume、不签任何 acceptance；生产 prepare 必须仍完整执行真实 compiler.consume。RAM 环境是原测试数组 callback，不是 CPU 神经/真实 physics。开发中真实 RED 保留：缺模块 collection error；首轮30 failed/2 passed定位误用旧 conditional task validator，改成 compiler.core；随后1 failed/13 passed定位 fixture 引用缓存 original row，独立 deepcopy 修复；实际 saved join 1 failed/34 passed定位 digest canonicalization；前置 guard 的 RED 为缺 cp5_guard StopIteration。均是本新增 adapter/fixture 的实现接缝问题，不解释任何历史 P1 负结果。

最终测试数量、SHA 以交付消息为准，不在本文自引用；冻结后不回写生产结果。没有执行本版真实物理或模型。不能由 tests/compile success/driver guard 自动签发科学、freshness 或 P1 通过。
