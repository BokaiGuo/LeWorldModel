# Moving overlay capability probe：实现交接（MAIN 审读后执行）

本工具只为 C2 mechanism calibration 提供固定未来动作干预，不是任务胜出、P1 通过、模型修复或新独立样本证据。只新增同 stem 源码、测试、本说明；不修改任何 sealed 文件、结果或共享总账。

## 固定实验与接口

输入为真实 recompile physical 输出目录及外部给定 report SHA。prepare 通过原 physical.verify 闭合原 prepare、原 compiler.consume 的全部 8 个计划、paired source 的 blind NPZ/PNG、模型与源码依赖；额外闭合 physical report、execution_binding、journal hashchain、warmup0..10 和 plan0..5 的完整 off/on/off PNG、状态和 body snapshot。身份同时包含 root_slot、condition、arm、selection_digest、task_id。核验原 public transition 的 requested/actual float32 bits，不凭目录名或行号近似配对。任一缺失、SHA 漂移或 join 不闭合即失败；不换旧 loader、不读取 witness 来选择动作。

固定顺序：slot0/1 → off/on → line_search_GN/full10_conditional_D2P → zero,+x,-x,+y,-y。8 anchors × 5 probes = 40，独立根仍是 n=2。每 probe 重复 5 public steps，固定 raw float32 动作为 (0,0)、(+.1,0)、(-.1,0)、(0,+.1)、(0,-.1)。.1 沿用 frozen warmup 的 raw 幅度，不是 tanh/logits，不拟合、不根据输出改变幅度或动作。native relative=True、action_scale=100、dt=.01、control_hz=10；raw ∈[-1,1] 内，无额外 clip/归一化。冻结 native step 中 position clip 本就注释掉，不能声称存在位置 clipping。

## 精确 suffix 起点：不能调用 native setter 作 teleport

native PushT `_set_state` 只设置 agent position/velocity、block position/angle，缺少 block velocity/angular_velocity 等，且额外 `space.step(dt)`。因此严禁将 checkpoint 的 7 维向量送入 setter 冒充状态恢复，也不逐字段写 body 状态来猜 solver cache。

本实现的恢复方法是每分支全新环境，完全重放原 recipe：reset(seed=slot.reset_seed) 取得 native goal → 原 sample_state recipe → reset(seed=0, options={state,goal_state}) → 相同 10 步 warmup 和 0/5/10 历史 → 相同离线 task goalRGB 绑定 → 原计划前 5 个 raw commands。所有 reset 内 setter/physics 均由原封存 hooks 计量，suffix 不调用 setter。warmup0..10、plan0..5 全 51 次 off/on/off render，与原 durable trace 比较 PNG 文件 SHA、pixel SHA、native7、两 body float64 bits、forces/torques/sleep、goal_state/goal_pose/goalRGB、renderer；任一不同，在该分支第一个 probe action 之前 fail closed。每个 probe 各自从同一封存 anchor 重放，不从另一个 probe 终点续跑。

必须明确边界：原 `moving.snapshot` 注明 solver warm-start cache 不可序列化。本实现是原路径的确定性重新构建及全可导出 trace 的位级核对，不是对内部 cache 字节的独立认证；测试不能证明真实模拟器确定性。若 MAIN 的“精确恢复”要求包括对不可见 solver cache 的独立位级证明，当前 artifacts 不足，本实现不能满足该更强要求，禁止将导出字段相等当作该证明，必须先停下来决定是否有可审核的 native 完整恢复机制。不得用容差、近似 setter 或仅 PNG/7 维相等替代。实际重放通过与否只能由 MAIN 真实运行判定。

goal_pose 完全不改。native 默认 outline 不是构造 task goal；with_target 只切 renderer。执行动作时 with_target=False，与原 physical 控制协议相同；每个观察均保存 off/on/off，选取 condition 对应 PNG 作为标签。模型调用为零，目标仅用于原 native 评分与原 goal attachment，不用于 probe 选择。

## 输出、失败和预算

report 顶层 schema/status/rows/errors、prepare/planner/physical SHA、contract、实际 counts、journal chain tail/SHA、inputs_final_verified、new_neural_calls=0、P1_admission=False。rows 恰好 40，含 anchor、root/condition/arm、selection/task digest、probe、start_verified、start_state、responses、完整 trajectory/PNG/checkpoints。每次响应保存真实 native7 state、相对该 anchor 的 `delta_native7`，另外独立保存 wrapped-angle delta，不把不同单位压成加权分数。原始角度 delta 不替换。零值、负值全部保留；无响应是空列表、未知 start_state 为 null，绝不填零。

`paired_descriptive_responses` 固定 20 项 root/condition/probe 配对，逐项保存 D2P−GN 的 prefix5 native7 状态差及每步相对各自 anchor 响应差。任一侧 start/step 未验证或缺失时对应差为 null，零差保留；没有合成跨单位分数、胜负筛选或根排序。

原始 public transition journal 保存 requested/actual bits、真实 reward/terminal/truncated/state/snapshot；PNG 在 guard 拒绝前持久化。合法 suffix native stop 保留已返回响应，status=native_stopped，不补足 5 步；工程失败留 attempted/completed 账、错误与 40 分母，不换根、不重试。失败某分支后可执行其他固定分支，不是补采。环境 close 和 hook 恢复保留错误；输出 I/O 本身坏掉时无法保证 report 落盘，但函数返回失败，不冒充完整封存。

最大 900s（MAIN verify 时间也计入）、192MiB output、200MiB disk floor、8MiB final reserve。全 horizon 上限：40 constructor/close、80 reset、800 public_step、8160 physics_step、160 set_state、120 goal_setter、80 goal_render、40 goal_bind、2640 explicit_render/PNG。每次 dispatch 前检查 wall/disk/cap，已失败尝试不清零；只允许 close 绕过 wall/output 预检查。最终重新核全部输入 SHA，检查 native operation census；不因错误放宽预算。

## MAIN CLI

以下均从 `/home/gbk/code/LeWorldModel/le-wm-main` 执行；本 worker 未调用真实 prepare 或物理 run。将 PHYSICAL_SHA 换成 MAIN 独立核对的 physical report SHA，将 PREPARE_SHA 换成 prepare 回传值；不能用 planner report SHA 冒充 physical SHA。output/prepare 路径必须不存在。

```bash
.venv/bin/python -B -m experiments.spectral_reachability.moving_overlay_capability_probe_physical_20261027_v1 prepare --physical-output results/p1_moving_overlay_recompile_physical_20261027_v1_MAIN01 --physical-report-sha PHYSICAL_SHA --output results/moving_overlay_capability_probe_physical_20261027_v1_prepare_MAIN01.json
.venv/bin/python -B -m experiments.spectral_reachability.moving_overlay_capability_probe_physical_20261027_v1 verify --prepared results/moving_overlay_capability_probe_physical_20261027_v1_prepare_MAIN01.json --prepare-sha PREPARE_SHA
.venv/bin/python -B -m experiments.spectral_reachability.moving_overlay_capability_probe_physical_20261027_v1 run --prepared results/moving_overlay_capability_probe_physical_20261027_v1_prepare_MAIN01.json --prepare-sha PREPARE_SHA --output results/moving_overlay_capability_probe_physical_20261027_v1_MAIN01 --authorize-physics
```

物理 run 仅 MAIN 审计认可恢复语义后执行。prepare/verify 只读 saved 数组/metadata/源码哈希，不 instantiate 环境、不 neural、不 physics。

## RAM 回归

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_moving_overlay_capability_probe_physical_20261027_v1.py
```

测试使用显式 numeric mock/内存 PNG 和 hashchain，覆盖固定动作、40 全接线、完整计量、原 prefix/state/body/非选中 on-frame 漂移拒绝、真实事件前置持久化、suffix early-stop/抛错/动作替换、无授权/超预算、metadata 身份缺失。没有运行模型、真实 PushT 或实际 prepare；测试是工程证据，不是科学或真实恢复成功证据。
