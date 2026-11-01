# Moving overlay capability probe：saved analysis 接口与边界

仅本 stem source/tests/innerdoc；不修改任何 sealed 文件、结果或共享总账。experiment-code 技能用于预先固定分析、日志与失败合同；不采用其训练、效果择优或强制非零结果模板。全部实现验证限 RAM，不启动神经或物理。

## 内部 normalized 合同（先定义；原生适配见后）

初次检查时 producer 源未出现，先定义此最小接口并在 commentary 通知 MAIN。随后 Nash 源落盘并已全文读取，现已提供 **直接原生 report/prepare/journal 适配**，不要求 Nash 增加下述 normalized 字段。此节是内部算术合同，也保留可选 manifest 入口，不是已有真实资产存在的声明；不能将 fixture 当真实报告。

manifest 顶层：

- `schema = moving_overlay_capability_probe_physical_20261027_v1.saved_analysis`
- `prepare_sha256 / source_report_sha256 / planner_report_sha256`：实际外部封存 SHA。
- `units = {position: px, angle: rad, action: raw_command, time: public_step}`。
- `source`：`{path:absolute, sha256}`，指向本次 probe 的真实 report.json。
- `prepare`：`{path:absolute, sha256}`，指向本次 probe 的封存 prepare；SHA 同 prepare_sha256。
- `protocol`：事前封存字典，含 `roots[2] / cases[8] / probes[5] / prefix_steps / probe_steps / warmup_steps / meter_per_full_branch`；实际 prepare 必须有逐字段相同的 `analysis_protocol`。分析不自行选择动作幅度、时域或资源阈值。
- `rows[40]`：固定 slot-major / off,on / GN,D2P / zero,+x,-x,+y,-y，所有未开始及失败项也必须存在。

`roots` 每项：`root_slot(0/1), task_id, source_root_state[7], goal_state[7]`。这里 task_id 保留 off/GN 原 packet 身份作为引用。任务方向从原根到原 goal 定义，同 root 各 condition/arm 共用，不从新终点/own residual/最优 probe 定义。

`cases` 每项：`root_slot, condition(off/on), arm(line_search_GN/full10_conditional_D2P), task_id, selection_digest, prefix_commands:float32[prefix_steps,2], expected_start:{state[7],snapshot,history_sha256}`。八项固定，不合并相同动作或相近状态；snapshot 为真实导出的 body 身份，必须含 `state_float64_bits`，与 state7 little-endian float64 bytes.hex 精确一致。完整 snapshot / history 只在同 case 内比较，不要求不同 case 相同。相同 root 的实际 root_state/goal_state 必须一致；同 condition 的 GN/D2P task_id 相同，但 off/on packet task_id 可不同，不可误拒合法条件分离。

`probes` 每项：`probe_id, commands:float32[probe_steps,2]`。顺序固定 zero,+x,-x,+y,-y；zero 全零，±x 仅 x 坐标非零且逐步同幅等反，±y 同理。幅度在 (0,1]，以真实封存 raw float32 值为准，可 x/y 不同。不是 logit 坐标，不作 tanh 或重标定。

每 row：

- `root_slot / condition / arm / probe_id / selection_digest / task_id`。
- `source_report_sha256 / planner_report_sha256 / prepare_sha256 / probe_sha256`（probe_sha 是该 probe 字典 canonical JSON SHA256）。
- `status`：`completed / native_stopped / engineering_failed / not_started`。
- `start`：与该 case expected_start 精确相同，或尚未重建完成时 null。
- `prefix_commands`：实际返回的 prefix 动作，可合法缺尾；必须是对应封存 prefix 的精确 float32 前缀。
- `trajectory`：从 probe step0 开始的连续真实返回，条目 `{step,state[7],snapshot,native_success:bool|null,reward:number|null,terminated:bool,truncated:bool}`；无法到 probe 起点时为空。step0 state/snapshot 必须等于 start，reward 为 null；未返回步不能补造。原生 producer 未独立保存 native_success 时保持 null，不能从距离或 termination 伪造该缺失字段；原始 terminated/truncated/reward 另行保留。
- `returned_commands`：实际返回的 probe 动作，必须逐位等于封存 probe 的已返回前缀。
- `counts`：每资源 `{attempted:int,completed:int}`，键集与 protocol 的 `meter_per_full_branch` 相同；completed 不得大于 attempted；任何项不得超过封存完整分支上限。
- `journal`：实际逐资源 attempt/return/failed 事件列表，事件含 `resource,kind`（attempt/return/failed）；严格配对并逐资源独立保序。其计数必须等于 counts，不能将预估成本当实账。允许失败时尚未返回的 attempt。

资源键至少 constructor/reset/public_step/physics_step/set_state/goal_setter/goal_render/goal_bind/explicit_render/close。public_step 完成数须为 warmup+实际 prefix+实际 probe；全分支 completed 时 counts 全部 attempted=completed=封存预算。部分执行额外保存 `warmup_returned`（0..warmup_steps），按实际账核对；prefix 不完整不得开始 probe。
manifest 中的 rows 必须逐字段等于 source report 的 `analysis_rows`，report 的 `analysis_protocol` 同 protocol，report 三个上游 SHA 同 manifest；prepare 的 `analysis_protocol` 同 protocol、source/planner SHA 同 manifest。外部 manifest SHA 是分析准入边界，不由代码虚构未来 SHA。

完整性字段缺失、行/动作/状态/来源/计量矛盾属于拒绝，不静默变 missing。真实合法缺失则保留整行、null 指标和明确原因。不得把 native stop 的短轨迹终点冒充固定 probe_steps 终点。
若在 warmup/prefix 阶段原生早停，允许 `start=null, trajectory=[], returned_commands=[]`，但必须保留 `preprobe_stop={phase:warmup|prefix,step,state[7],snapshot,native_success,terminated,truncated}`，step 等于相应实际返回步数且有真实停止标志。此记录与 probe endpoint 分开输出，不把它填入缺失 probe 的 native_success。工程失败可有未完成 attempt；无失败/悬置 attempt 时额外核对 physics=2reset+10public、set_state=2reset、goal_render=reset、goal_setter=reset+goal_bind。

## 预先固定的分析

对每 case，真实固定时域 endpoint 作单尺度中心 secant：`B[:,x]=(s(+x)-s(-x))/(2*a_x)`，y 同理。分开输出 objectXY(2×2, px/raw_command)、agentXY(2×2, px/raw_command)、unwrapped object angle(1×2, rad/raw_command)。角度另报 wrapped 中心差分及 branch-cut 歧义，只是诊断，绝不改 native 标签。位置与角度不混成无单位拼接矩阵。
各矩阵保留列级 missing；缺一对 ± 则该列 null，整体 SVD/条件数为 null，不能丢列后冒称完整矩阵。SVD 是 saved numpy 算术，不拟合模型/控制器。条件数用全部已保存矩阵的双精度奇异值；精确最小奇异值为零则 JSON null 并标 singular，零矩阵保留全零奇异值，不设置科学成功阈值。
每 probe 同时保留 endpoint−zero 的 signed 有限响应、zero 漂移、中心对称残差。沿原根→真实 goal 的 objectXY/agentXY 单位方向分别投影；方向严格为零则 null/undefined，不换方向。另保留同单位 agentXY+objectXY 四维共同方向。有限投影保留全部零与负值；不选“最好” probe，不生成待执行动作或可达证书。
配对输出固定 root/condition 的 D2P−GN 和 root/arm 的 on−off，均逐 probe 及响应矩阵，不剔除失败对，不以 probe 数扩根数。n=2 仅校准描述；没有因果泛化、全局可控性、P1 通过或模型机制正确的结论。

角度主列取完整 snapshot 中 block 的真实 unwrapped_angle，state7[4] 是原生 modulo 2π 观测，二者分别保全并核对。wrapped secant 不是从 modulo 状态直接相减；另标注 signed pair 的真实 unwrapped 差达到 π 时存在圆周分支歧义。角度不与 px 混权，也不投影成 native 成功。矩形矩阵条件数明确定义为 min(m,n) 个奇异值的最大/最小比；1×2 非零角度行的该比值为 1，不代表两个动作方向都可辨识，另保留 input_nullspace_dimension_lower_bound。近奇异但非精确零的奇异值不按新阈值截断。

## API / MAIN CLI

`analyze(manifest)`：纯 RAM/已保存 numpy 算术，拒绝合同矛盾。
`load_saved(path, external_sha)`：核对 manifest、实际 probe report、实际 prepare 的 SHA、protocol 和 rows 一致性；最终再次检查输入未漂移。不调用 producer 或其 verify，不读取目录中“最新”报告。

`normalize_native(prepare, report, events, upstream_report, prepare_sha)`：对齐 Nash 当前真实源码的原生保存布局，不导入其模块。固定 raw float32 `.1`、prefix5/probe5/warmup10、8×5 顺序与原生计量预算。上述幅度来自 producer 的事前动作字典，不是分析器选择的新效果阈值。
`load_native(output, report_sha, prepared, prepare_sha)`：独立验证实际 probe report/prepare 外部 SHA、execution_binding、全部 prepare bindings、原八臂 physical report、journal 文件 SHA/原生 spaced-JSON hashchain/terminal tail；最后再次检查输入未漂移。逐 branch 从真实 operation_attempted/completed/failed 重建计量，逐 returned_transition 验证 warmup/prefix/probe 原始动作 bits/actual dtype、step 连续性与实际状态 snapshot；与 responses 字段独立交叉比对。prefix 起点还逐项复核原完整 51 项 off/on/off warmup/prefix trace 的 state/snapshot/file/pixel 身份和 suffix_start_verified 事件。不依赖“分支号”等于 journal index，按 root/condition/arm/probe 精确配对。

MAIN 首选实际 producer 原生接口（变量须填实际绝对路径和实际外部 SHA，输出不得已存在）：

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.moving_overlay_capability_probe_analysis_20261027_v1 --physical-output "$PROBE_PHYSICAL_OUTPUT" --physical-report-sha "$PROBE_PHYSICAL_REPORT_SHA" --prepared "$PROBE_PREPARED" --prepare-sha "$PROBE_PREPARE_SHA" --output "$NEW_ANALYSIS_JSON"
```

本任务没有运行此命令，没有制造真实 SHA。`PROBE_PREPARED` 是 Nash **probe** prepare，不是其前置 recompile physical prepare；report_sha 是本次 probe report 的实际 SHA，不是 planner report SHA。

可选的 normalized manifest 接口（仅适用于另有明确 `analysis_protocol/analysis_rows` saved exporter 的产物；不要求 Nash 改旧文件，不可与 native 参数混用）：

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.moving_overlay_capability_probe_analysis_20261027_v1 --manifest "$PROBE_SAVED_MANIFEST" --manifest-sha "$PROBE_SAVED_MANIFEST_SHA" --output "$NEW_ANALYSIS_JSON"
```

输出路径必须未存在；JSON 使用独占 `xb` 写入、fsync，不覆盖旧结果。输出 schema 为本 stem `.report`，固定 denominators roots=2/cases=8/probes_per_case=5/branches=40/comparisons=8；rows 为八个 case，每个列全部五个 native/missing 结果、object_xy/agent_xy/joint_xy/object_angle_unwrapped/object_angle_wrapped_diagnostic。`analyzed_with_missing` 是成功保存缺失信息，不是补齐数据；它不表示研究 gate 通过。合同矛盾以异常/nonzero CLI 拒绝。
source report 的引用名 `source.path` 是 **本次 probe physical report**，不是 upstream source_report_sha256 所指的最初 moving source，两者不能混淆。

## 回归与冻结限制

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_moving_overlay_capability_probe_analysis_20261027_v1.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.moving_overlay_capability_probe_analysis_20261027_v1 --help
```

本次最终 RAM 回归 53 passed in 3.46s；CLI --help exit0。人工线性 fixture 只验证算术和合同，不写物理报告，不是研究证据。变异覆盖行分母/顺序、task/selection/probe/source/planner/prepare 身份、history/state7/body bits、动作/步数、原生停止、缺零/缺±、journal attempt-return/计量、JSON duplicate/NaN、单位与实际 report/prepare 不匹配；另包含 Nash 原生布局和 direct loader 的 RAM metadata 回归、原生 spaced-JSON chain、跨 condition 不同 task_id、实际动作/trace/response/全局计量变异拒绝。保留零响应、负投影、角度跨界、工程失败、probe 前及 probe 内 native stop。

Nash 真实 producer 源在本接口初次定义时未存在，后续已读并完成原生 adapter 的 RAM metadata 回归；不声称实际 probe 产物已经验证或物理已经完成。此任务不运行实际 prepare、物理、模型、CUDA、拟合或实验；没有新的真实 probe 资产、科学报告或未来报告 SHA。
封存的 manifest/prepare/report 是证据边界；normalized journal 必须由 producer 如实导出，本分析不独立证明真实物理已经执行，不序列化隐藏 solver/contact cache，也不能由 snapshot 相等推断全部隐状态相等。真实执行/源码审计由 MAIN 负责，零负结果不得用于再筛根/动作/阈值。新接口缺失时应停止真实分析，不自动宽松降级。
