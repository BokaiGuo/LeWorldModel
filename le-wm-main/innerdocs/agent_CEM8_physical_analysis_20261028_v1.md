# CEM8 与 spectral24 teacher_full 的 saved-only 配对分析

本实现只消费已 terminal 的 saved 产物。MAIN 审读后运行；作者只执行 RAM/临时元数据测试和 CLI help，未执行真实 consume/分析、prepare、neural 或 physics。测试通过不是科学证据，更不是 P1 通过。不修改 producer、原报告、阈值、选根或 observed4 cohort。

## 独占交付

- `experiments/spectral_reachability/agent_CEM8_physical_analysis_20261028_v1.py`
- `tests/test_agent_CEM8_physical_analysis_20261028_v1.py`
- 本 innerdoc；无其他项目文件变更。

## 真实接口与先验源绑定

CLI 的唯一分析入口 `analyze()` 在任何 consumer import 前检查两个冻结源码 SHA（CEM 模块会传递 import spectral helper，不能等到第二次 consume 前才检查）：

- `agent_moving_CEM8_physical_20261028_v1.py`: `d9c72806141f1291aa1e7d088c0419035b34b9b529a452af2ae1ea0921b33fc2`
- `capability_suffix_factorial_physical_20261027_v1.py`: `15125f4d86e5a6b3344f750205101a0e8a6f4a8dfba82ce46ce3d925f4ea8bd6`

随后完整调用两个真实 `consume(output, actual_report_sha)`，各一次；没有跳过验证的 CLI 开关，也不以 prepare、测试回执、commands_released 或 MAIN 口头状态代替 consume。消费异常直接传播，不输出成功分析。真实消费者负责 journal/hashchain、完整 source/raw/task closure、原51帧、PNG文件/像素、状态与操作回执验证。分析器保存两个完整 consume 返回值，包括全部 spectral24 行及 bindings；另保留原物理报告除 rows 外的全部 diagnostics/errors/resources。

输入必须为绝对路径和实际非零 SHA；读取限制 CEM report 64MiB、spectral report 192MiB、prepare 16MiB，拒绝重复 JSON key、非有限数、符号链接文件。合并 prepare/consumer 的全部 hash bindings，冲突失败关闭，最终及写出前重算输入 SHA。绑定自己的 source、真实 report、prepare、全部已有产物与原输入 closure，不制造 future SHA。

## 配对与真实 ABI

固定顺序是 `root_slot=(0,1) × condition=(off,on) × prefix_arm=(line_search_GN,full10_conditional_D2P)`，共8锚点；spectral24 的变体顺序必须是 `open_full,teacher_full,teacher_small`，比较行 ordinal 严格为 `3*i+1`。全部24行先做 ordinal/key/variant 身份检查；不筛选成功行，不混 observed4。

两个 prepare 的原 plans、anchors（包括51条 trace）、原 physical report SHA 必须完全相同。逐项绑定 raw float32 prefix、完整原始 source/planner/blind/task/selection、ordinal、selected_suffix 和自身新 suffix。绝不重新 tanh、clip、scale 或换 raw buffer。spectral `original_plan` 的真实 ABI 比物理 prepare 多保留 schema/prediction/logits 等字段：保留这些字段且由真实 consume 验证；物理 prepare 的全部原身份字段仍必须逐项相等，不能错误要求两种完整字典 keyset 相同。

逐对检查 root native7/goal 与 SHA、goal pixel、goal_pose、checkpoint0/5 的 state/snapshot/fileSHA/pixelSHA。PNG 文件名因分支 ordinal 不同不作为相等依据。checkpoint5 还须等于原 anchor；缺已绑定身份则在 `identity_unknown` 逐项列出，两侧配对统计结果均 unknown，不补数据。精确重放匹配不是完整物理等价、新鲜根或 reset-bound 认证。

## 独立指标与 early-stop

逐个实际 suffix response 复算 native7 distance、reward（负 native7 L2）、native success、delta、wrapped delta，并与回执原字段严格相等；最终状态须来自最后实际返回，H10 必须来自第五次 suffix 返回及真实 checkpoint10。

分解包括 agent XY、object XY、joint first4 XY、wrapped object angle、native7 中最后两维 velocity、joint XY + wrapped angle L2。保留 native success 的原公式 `jointXY<20 && min(abs(dtheta),2*pi-abs(dtheta))<pi/9`，包括多圈时 native angle 可负的既有行为；另外报告 wrapped-joint success，不静默修复 native 语义或新增 velocity 阈值。object_success 继续 unknown。

- `observed_endpoint`: 有实际返回则保留复算结果，即使工程失败；不是可统计的完整任务结果。
- `endpoint`: 仅正常完成或 native_stopped 的真实 endpoint，包括真实成功 early-stop。
- `horizon10`: 仅实际达到 checkpoint10 的正常执行；early-stop 保留 null，绝不 padding 或将提前 endpoint 当 H10。
- 工程失败/not_started 的可统计 endpoint 为 null，不转成零或 false。

输出逐行8项，汇总 all8、GN/D2P、off/on、condition×prefix、每个已知 root_slot。各组分母固定；任何成员 unknown，完整分母 mean/success count/rate 为 null，同时给 available/unknown 数。不输出成功子集均值伪装完整均值。配对差定义 CEM minus spectral，只有该组全部配对值存在才计算。

只有2个已知根，8 anchor 不是8个独立 root；没有独立8样本检验或置信区间。fresh_roots/new_independent_roots/new_independent_samples=0，freshness/independent_confirmation/P1/scientific_admission/equal_total_compute/equal_FLOPs 始终 false。CEM64 与 spectral 候选/重放预算不同，只保留各 producer 的实际资源回执；nested physics/render/public 不能相加当总预算，分析自身 new_neural_calls/new_physics_calls=0 不是上游成本为零。latent Q 变化不能推出任务改善，即使任务指标改善也不是 equal-compute/fresh P1 证据。

## MAIN 精确命令（此处尚未执行）

以下两个 report SHA 已仅用只读元数据/schema smoke 核对；不表示作者已运行严格 consume。CEM 实际运行状态由 MAIN 报告 7full+1earlystop、0 engineering；这些数值不是代码硬编码的分析结论。

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_CEM8_physical_analysis_20261028_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.agent_CEM8_physical_analysis_20261028_v1 \
  --cem-output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_moving_CEM8_physical_20261028_v1_MAIN01 \
  --cem-report-sha f359812367d8e901b171921552ee464cb3be399168e825bdca4c3fdde4c0a2aa \
  --spectral-output /home/gbk/code/LeWorldModel/le-wm-main/results/capability_suffix_factorial_physical_20261027_v1_MAIN01 \
  --spectral-report-sha cc92465141b400659e024e052b05683153b4dc6c013b81bdf1e3e29355fd2663 \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_CEM8_physical_analysis_20261028_v1_MAIN01.json
sha256sum /home/gbk/code/LeWorldModel/le-wm-main/results/agent_CEM8_physical_analysis_20261028_v1_MAIN01.json
```

分析命令只读所有输入，唯一写入是新的独立 analysis JSON，输出父目录须已存在；拒绝重入/覆盖、向 producer output 内写入、覆盖输入。输出≤32MiB，写前 free≥200MiB+payload，以 `xb` 排他创建、fsync 文件及父目录。失败不删文件、不重试覆盖；若写中出错残留半文件，须 MAIN 保留为失败证据并另选 NEW 文件名，不能当成功回执。

## 实际测试及限制

定向 RAM/临时文件测试共60项，覆盖真实 ABI extra fields、完整双 consume 接线、源码预检顺序、输入/hash冲突/漂移、错误传播、固定分母与负向 key/raw/goal/checkpoint/metric 篡改、成功早停/截断/未启动/工程失败、不补H10、native多圈语义、输出上限/排他/空间floor及 CLI。测试不导入真实模型/环境，不执行物理模拟；真实消费者在接线测试中以 RAM 返回字典替代。`--help` 已运行。

真实 consume、真实 saved 分析输出、任何科学结论均待 MAIN 审核运行；本实现不修复旧阻断，不改变任何 producer/source seal、原报告或冻结模型。最终三文件 SHA 由交接消息给出，避免文档自包含自身 hash。
