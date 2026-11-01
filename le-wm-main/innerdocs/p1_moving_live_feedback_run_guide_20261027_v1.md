# Moving live feedback：输入预诊断、partial 解释与历史排重入口

本指南仅为运行/分析交接，不审计实现，不增加实验，不修改冻结 driver/source/tests/doc。
MAIN 已启动真实连续 driver；本指南不读取该运行中的 journal/report，不提前声称成功或失败。
实际结果由 MAIN 外部 SHA 与最终封存产物决定。测试记录采用 MAIN 提供的同 SHA 实测
**22 passed / 68.33s**；本线程早先所报24不作为 MAIN 验收分母，也不为对齐计数重跑或改 sealed 文件。

## 1. 先给 fresh-root 排重 worker 的输入范围

完整历史范围不是“最近 sourcepack 的成功 slots”。须覆盖已使用、失败、中断、未选中、仅 prepare
预留的源；既要保 seed/span，也要保实际 observed roots。以下清单是本次只读文件名 census 中
确认存在的明确入口，**不是全历史排除证书**。路径存在不代表其 schema/内容已获准或任务独立。
不得把 solver/outcome/witness 内容传给后续 planner；排重只投影来源与已观察身份。

已有规则入口：

```text
/home/gbk/code/LeWorldModel/docs/D2P_MOVING_FRESH_ROOT_PROTOCOL_20261027.md
/home/gbk/code/LeWorldModel/docs/D2P_MOVING_OBSERVED_FEEDBACK_PROTOCOL_20261027.md
/home/gbk/code/LeWorldModel/le-wm-main/innerdocs/p1_conditional_source_metadata_20261027_v1.md
```

先封存 cutoff，再用以下**只读、无名称过滤**命令取得 results 下完整显式文件路径分母；
它们只输出文件名/时间/数量，不执行 Python、模型或物理，不生成文件。
运行中目录必须留在清单内并标 `in_flight/unsealed`，不能因为没有 report 就当未使用。

```bash
date -u +%Y-%m-%dT%H:%M:%SZ
rg --files --hidden --no-ignore /home/gbk/code/LeWorldModel/le-wm-main/results | LC_ALL=C sort
rg --files --hidden --no-ignore /home/gbk/code/LeWorldModel/le-wm-main/results | wc -l
find /home/gbk/code/LeWorldModel/le-wm-main/results -type l -print
```

这些命令是可重复的路径发现规则，但 live tree 不具原子快照语义；记录 cutoff、命令和完整 stdout，
MAIN/worker 在独占新产物中封存，不覆盖旧 inventory。enumeration 前后有变化则列为未知增量，
不能拿第一次路径列表宣布后续时刻全覆盖。不要使用 `head`、只搜 `sourcepack.json`、只搜成功报告、
只用 `rg` 默认忽略规则或目录名没有 moving 就跳过。符号链接外部目标与 results 以外的引用依赖
必须另记录；上述路径 census 不自动覆盖它们。不得下载失踪文件来补“完整”。

### 新 source / 原零历史 / 非零历史：显式入口

```text
/home/gbk/code/LeWorldModel/le-wm-main/results/p1_conditional_procedural_source_20261027_v1_calibration_request.json
/home/gbk/code/LeWorldModel/le-wm-main/results/p1_conditional_calibration2_main_20261027_v1_request.json
/home/gbk/code/LeWorldModel/le-wm-main/results/p1_conditional_calibration2_main_20261027_v1/sourcepack.json
/home/gbk/code/LeWorldModel/le-wm-main/results/p1_observed_history_source_20261027_v1_main01/sourcepack.json
/home/gbk/code/LeWorldModel/le-wm-main/results/p1_constructive_reachable_source_20261027_v1_prepare01.json
/home/gbk/code/LeWorldModel/le-wm-main/results/p1_constructive_reachable_source_20261027_v1_main01/sourcepack.json
/home/gbk/code/LeWorldModel/le-wm-main/results/p1_constructive_moving_history_source_20261027_v1_prepare01.json
/home/gbk/code/LeWorldModel/le-wm-main/results/p1_constructive_moving_history_source_20261027_v1_main01/sourcepack.json
/home/gbk/code/LeWorldModel/le-wm-main/results/p1_constructive_moving_history_source_20261027_v1_main01/execution_binding.json
/home/gbk/code/LeWorldModel/le-wm-main/innerdocs/p1_conditional_procedural_source_20261027_v1_calibration_config.json
/home/gbk/code/LeWorldModel/le-wm-main/innerdocs/p1_conditional_procedural_source_20261027_v1_calibration2_config.json
/home/gbk/code/LeWorldModel/le-wm-main/innerdocs/p1_conditional_calibration2_main_20261027_v1.json
```

每个 sourcepack/request/prepare 还需显式展开其外部路径依赖，记录外部 SHA、seed/reset-span、
root state/snapshot、history/raw hash 与 lineage。不要按旧 resolver 支持的 schema 静默删掉新 envelope。
压缩/gzip_base64 封装、journal-only 中断、无 per-root seed attribution 都应记录具体 unresolved 原因；
不能把“未识别”当“未用”。旧 metadata resolver 自称 coverage unknown，不能将它自签为完整 authority。

### 原四根与 P1/P3 派生 observed roots：显式入口

```text
/home/gbk/code/LeWorldModel/le-wm-main/results/p3_quality_cost_audit_20261027_v2/manifest.json
/home/gbk/code/LeWorldModel/le-wm-main/results/p3_quality_cost_audit_20261027_v2/report.json
/home/gbk/code/LeWorldModel/le-wm-main/results/p3_fixed_horizon_budget_schedule_20261027_v2_prepared/prepare.json
/home/gbk/code/LeWorldModel/le-wm-main/results/p3_action_jacobian_budget_schedule_20261027_v3_prepared/prepare.json
/home/gbk/code/LeWorldModel/le-wm-main/results/p3_horizon_residual_budget_20261027_v4_prepared01/prepare.json
/home/gbk/code/LeWorldModel/le-wm-main/results/p3_horizon_residual_physical_20261027_v4_prepared01/prepare.json
/home/gbk/code/LeWorldModel/le-wm-main/results/p3_joint_horizon_goal_metric_scheduler_20261027_v1_prepared01/prepare.json
/home/gbk/code/LeWorldModel/le-wm-main/results/p3_joint_horizon_goal_metric_physical_20261027_v1_main01/report.json
/home/gbk/code/LeWorldModel/le-wm-main/results/p1_goal_consistent_conditional_physical_20261027_v1_main_v1/report.json
/home/gbk/code/LeWorldModel/le-wm-main/results/p1_goal_consistent_conditional_physical_20261027_v1_source_prepared/prepare.json
```

这些入口不能代替全部旧 PushT。完整路径 census 内的 `pusht*` 旧 report/protocol/root/args/source-seal
也要逐个投影身份，**包含 root_N.prefix_M**；旧 prefix 是已访问物理状态，不能只保 root_N。
`source_seal_0…N` 数量不固定，按实际路径分母全取，不能写死0…3或省略额外 seal。
其他环境文件只有取得明确环境身份后才归类为不适用，不能靠名称近似作全局排除。

### Moving overlay / 计划回放 / 反馈：显式入口

```text
/home/gbk/code/LeWorldModel/le-wm-main/results/p1_constructive_moving_history_physical_20261027_v1_selection_main01.json
/home/gbk/code/LeWorldModel/le-wm-main/results/p1_constructive_moving_history_physical_20261027_v1_prepare01.json
/home/gbk/code/LeWorldModel/le-wm-main/results/p1_constructive_moving_history_physical_20261027_v1_main01/report.json
/home/gbk/code/LeWorldModel/le-wm-main/results/p1_moving_target_overlay_source_20261027_v1_prepare_main01.json
/home/gbk/code/LeWorldModel/le-wm-main/results/p1_moving_target_overlay_source_20261027_v1_main01/report.json
/home/gbk/code/LeWorldModel/le-wm-main/results/p1_moving_target_overlay_plans_physical_20261027_v1_prepare_main01.json
/home/gbk/code/LeWorldModel/le-wm-main/results/p1_moving_target_overlay_plans_physical_20261027_v1_main01/report.json
/home/gbk/code/LeWorldModel/le-wm-main/results/p1_moving_baseline_physical_20261027_v1_main01/report.json
/home/gbk/code/LeWorldModel/le-wm-main/results/p1_moving_baseline_physical_20261027_v1_kino_main01/report.json
/home/gbk/code/LeWorldModel/le-wm-main/results/p1_moving_extended_baseline_physical_20261027_v1_main01/report.json
/home/gbk/code/LeWorldModel/le-wm-main/results/p1_moving_live_feedback_physical_20261027_v1_selection_main01.json
/home/gbk/code/LeWorldModel/le-wm-main/results/p1_moving_live_feedback_physical_20261027_v1_prepare_main01.json
/home/gbk/code/LeWorldModel/le-wm-main/results/p1_moving_live_feedback_physical_20261027_v1_main01/execution_binding.json
```

最后一个运行中目录本指南只确认入口存在，不消费其运行结果。MAIN 封存后才用新 report/journal
补实际 observed roots；直到那时，既定 seed/prefix 范围仍按 reserved 排除。
同根 overlay/不同目标/速度/signed zero/角度加2π 都不产生新物理 pose；沿用 fresh-root 协议
既定 pose/alias 规则，不能看 planner outcomes 再选容差。复制 lineage 可归并，但不同 actual prefix
states 不能被归并丢失。路径重复数、replay 数、root-goal hash 数均不是独立根数。

## 2. 输入预诊断：仅 MAIN 的只读检查，不重启正在跑的 job

分析器应拿 MAIN 明确给出的 prepare/report 外部 SHA，不根据“最新目录”猜版本。
固定 frozen driver source SHA：`6742a1b35102e433d7772e4a208b4f97b37b76f815b991cfc0e472763754ace9`；
suffix source SHA：`b421af2a74bf8af5639f043ae3a06790895f24157880e67b556695b82bb348f6`。
完整 selection 字段与运行命令见本目录的 `p1_moving_live_feedback_physical_20261027_v1.md`，
不要把本指南当成新 selection/prepare authority。

|输入/现场现象|可解释边界与 MAIN 所需材料|
|---|---|
|路径不存在、SHA mismatch、prepare reconstruction 拒绝|输入或 seal 前提未满足；需确切路径、期望/实测SHA、first error；不是模型性能结果|
|modelbinding/source_closure 不匹配|原 suffix.modelbinding、模型文件/原normalizer/source封存不一致；不得删closure字段或patch globals绕过|
|HOST CUDA preflight/load/hash/显存/deadline 拒绝|工程限制；不得切CPU神经、提高cap或重跑同输出目录|
|native constructor/wrapper/reset lineage 拒绝|物理来源前提未建立；不得仅计public步后补假reset/physics账|
|warmup RGB/state/snapshot/raw mismatch|原根重建失败；不能reset到saved prefix再冒充连续feedback|
|同层 O/I/R prefix mismatch|paired前提失败；保存三分母和已发生工作，不跨层借图/latent|
|encoder/suffix异常或seal验证失败|该episode后续命令未获授权；不得fallback旧suffix、拿partial数组当完整输出|
|输出/磁盘/fsync拒绝|durability前提失败；缺文件不是数值零/策略成功；保原现场，MAIN决定新版本修复|

异常复现给 MAIN/Jason 的最小包：外部 prepareSHA、目标输出路径、episode/slot/arm/mode、
first error + secondary、最后 completed 与未返回 attempt、最后已durable observation/encoding/seal、
缺失文件名与所在 phase。不必复制checkpoint或权重，不删除现场、不重写原report。
真实接口故障是工程问题；反馈没有改善才是另一个需要完整 actual endpoint 的结果问题。

## 3. Jason 分析器：缺测与 partial 不填值

固定四层×O/I/R=12，独立任务数2。主配对 `R−I`，其次 `R−O` 与 `I−O`。
主report schema 为 `p1_moving_live_feedback_physical_20261027_v1.report`；
startup另有 `.startup_failure`，也不从不完整目录自行造一个 report。
row 的 `native_success` 必须结合真实 `terminated`、`truncated` 和异常状态，不以 `full_horizon` 替代；
`summary.native_success` 是12分支计数，不是独立根计数。

|保存终点/状态|允许分析|不允许补造|
|---|---|---|
|not_started / startup_failure|固定分母、模型是否已load及错误|native失败率、Q、物理不可达结论|
|native_stopped_warmup|实际已返回warmup及native标志；尚未绑定目标任务|任务goal success、prefix配对、Q5/Q10|
|native_stopped_prefix|实际终止点/任务标签；缺step5编码时显式缺失|复制旧prefix latent、假suffix、假完整10步|
|native_stopped_suffix且step<10|实际命令/终止标签、可用Q5及截短路径|Q10=0、补terminal编码、假full10|
|completed10且无工程错误|同层own-goal identityQ5/Q10、native标签、实际suffix差异|完成10即成功、Q改善即任务改善|
|failed / partial_or_failed|保first/secondary、实际资源与已返回观测；完整部分单独标范围|删除坏臂、用最后帧当step10、给缺值作均值零填充|

字段可能不存在：`encoding5`、`encoding10`、`actual_Q5`、`actual_Q10`、`suffix_request_sha256`、
`suffix_namespace`。O没有suffix compiler request是设计事实，不是漏跑；I/R未编译完则是缺测。
checkpoints按phase/step语义取，不拿数组位置猜时间。编码receipt中的latent是新NPY entry，
PNG SHA、pixel SHA、NPY file SHA、array SHA、JSON digest 各有语义，不可互换。
I新编码用于评价但不喂compiler；R自身编码才是feedback。三者都实际付编码成本，不去重计账。

工程完成与配对完整分开列：一臂合法native early stop可工程正常，但同层full10终态差值不可计算。
任一端点缺失则该pair输出缺测原因；仍保留12rows、各mode4、各slot6原分母。
两臂动作不同的终态差值是策略干预差异，不是相同动作的factual预测误差。
主实验没有额外teacher；若无对应真实预测receipt，不自行推导teacher误差。

## 4. 预算解释速查

正常无constructor隐式reset的12完整分支：constructor12、reset24、set_state48、goal_render24、
goal_setter36、goal_bind12、warmup120+control120 public、physics2448、PNG264。
这些是预期完整账，不是本次实际运行结论。constructor额外reset也必须计入actual。
`attach_goal`实际调用`_set_goal_state`，所以setter关系是 reset+goal_bind；不能少记12。
partial使用attempt/completed实际值；未返回attempt不退款，不用fullcap减partial去推遗漏调用。

完整新神经账：24 encoder/projector各；40 predict API/predictor/action_encoder/pred_proj各；
含8 AD-forward、8 J、8 SVD、8 jacobian API及8内部grad API、1536数学方向。
API、module与数学方向不能相加当独立forward或API总数。O不编译，I/R各固定5 forward，
不称O与I/R全pipeline等资源。旧source/model/task/plan成本单列继承，不因cache存在记为免费。

`wall_seconds`为run entry至final checks，含模型加载；`execution_wall_seconds`只是execute scope。
封存与final fsync也可能失败，不能只看最后控制step成功。工程cap600s/3GiB allocated/
4GiB reserved/64MiB全输出/200MiB磁盘floor，不因负结果临时提高cap。

本指南使用 documentation-and-adrs 技能组织输入、失败语义与不可变决策；未改任何旧closure，
未运行tests/prepare/CUDA/物理，未审计其他worker代码，未生成独立审计器或科学准入结论。
