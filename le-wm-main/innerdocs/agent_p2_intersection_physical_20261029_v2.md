# Intersection v2 六槽 raw15 native 物理执行及 saved 验证

本 physical v2 修正 v1 从 low 四槽合同遗留的 `fixed_denominator=4`，现明确固定 6 并与 order、plans、roots、H15 和 checkpoint 做一致性检查；四槽 `arm=allmode_full10_conditional_D2P` 与 prefix51 的 allmode 文案也改为本六槽身份。其余 low 继承字段已逐项核对：warm10、prefix51 定义、原 7 维状态、off/on/off renderer、动作缩放、两个 known roots、fresh=0、科学标志均仍适用；caps、预算和原生证据算法不变。v1 与其 prepare 不被修改或接受为 v2 prepare。错 4 分母的 prepare/report 必须拒绝。

独立 own-schema，固定 root0 joint/composition/grasp、root1 grasp/composition/joint，均为已知 off 根；不按 outcome 挑计划，不训练，不改变原科学门，不签 equal-FLOPs、freshness、P1/P2 或 physical release。只执行实际 v2 导出的 logits/context 派生 raw15；null 留原槽、不构造替代、不调用环境。执行不继承旧 physical 行。

prepare 调用实际 intersection v2 `consume(output, report_sha)` 完整 saved 重建并逐字比较持久 receipt，再调用其 `verify` 完整重建原 prepare；绑定实际 report/receipt/source SHA、模型任务来源与全部输入文件。原 30 logits 的完整 joint ball、float64 tanh→float32 raw15 和完整三段 action/latent shift 由 frozen preparation 算术重新验证。全部 source native root/goal/RGB/warm10 直接绑定；planner 账户原样保留，不冒认旧 schema，不将旧经济预算变成此次物理调用。

新执行和 saved 验证复用已经审读的 common-resource physical 的 native episode、资源语义与校验算法：真实 placement、warm10、source history3/原 7 状态和 goal 绑定；每槽按 literal raw15 执行、逐动作 bits 与 dispatch 对齐，cp0/5/10/15，terminated/truncated 即停、不补；原 reward/7维状态/速度/角度，off/on/off RGB 原始 PNG 全保留。saved consumer 全部事件种类、hashchain、nested intent/return、六槽与原子状态、真实 PNG 字节/pixels/guard、状态奖励和全过程 count/caps 独立重算；未知事件、额外文件、断链、缺字段或不符拒绝。不会重跑神经；其数学验证复用原方法而非声称独立神经证书。

六槽满程上限 constructor/close=6/6、reset=12、public_step=150、physics_step=1524、set_state=24、goal_setter=18、goal_render=12、goal_bind=6、explicit_render=486；attempted/completed 独立计账，失败不退款。128MiB 总输出、16MiB final reserve、磁盘 floor200MiB、全段900s；错误及未完成槽保留，不能将失败认作成功。最末 report fsync 的完成时刻不由 wall_seconds 单独认证。worker 仅 RAM/fake env/tempdisk；MAIN 审读后才物理执行。

从 le-wm-main，实际 SHA 由 MAIN 填入：

```
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p2_intersection_physical_20261029_v2.py
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_intersection_physical_20261029_v2 prepare --planner-output PLANNER_OUTPUT --planner-report-sha PLANNER_REPORT_SHA --planner-receipt PLANNER_RECEIPT --planner-receipt-sha PLANNER_RECEIPT_SHA --planner-source-sha PLANNER_SOURCE_SHA --output PREPARED
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_intersection_physical_20261029_v2 verify --prepared PREPARED --prepare-sha PREPARE_SHA
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_intersection_physical_20261029_v2 run --prepared PREPARED --prepare-sha PREPARE_SHA --output OUTPUT --authorization MAIN_P2_INTERSECTION6_FULL15_V2_PHYSICS_ONLY
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_intersection_physical_20261029_v2 consume --output OUTPUT --report-sha REPORT_SHA --receipt RECEIPT
```

路径为绝对路径；新产物 basename 必须以本 stem 加下划线开头且不存在。调用 `prepare`/`verify`/`run`/`consume` 会再次运行上游完整 saved consume，可能耗时；worker 不执行这些生产入口。实际结果才决定各槽是否有 native stop、缺 checkpoint 或失败，不将 missing 填 0。

