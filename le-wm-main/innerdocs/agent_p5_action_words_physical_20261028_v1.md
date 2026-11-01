# P5 native action words：全12物理重放，非任务成功认证

独占 source/tests/本说明，旧 v12、action_words v1/v2、MAIN 结果一律不改。
worker 仅 RAM 测试，不 prepare、不导入 simulator、不执行神经/物理/网络。
所有 native 执行由 MAIN 读码审核后显式授权。测试 token 环境不代表真实 physics 或科学证据。

## 已有精确重建入口

复用 `p5_jepa_factual_panel_20261027_v3.bound` 返回的 sealed v2/v1 helpers，
以及 panel.capture 的原五文件 schema。不借旧 endpoint，不 reset(savedpoststate)，
不 setter、不 monkeypatch、不改变旧 globals。原 constructor 为 `base.CTOR`：
`reset_to_state=None, relative=True, action_scale=100, with_velocity=True,
with_target=True, render_action=False, render_size=224, shape=T, legacy=False`，
其余原值完整封存。特别不能套用其他实验的 with_target=False。

每分支独立 constructor（内部 seed(None)，此时尚无 physics）→显式原 seed→
原 reset 一次（含一个 .01s Space.step）→原十个 nonzero prefix 控制步。
没有另加 warmup。seed 固定 2026102702/03/04，每根依次 xy/yx/x_minus_x/zero_zero，
全部12，绝不读取预测分数选 word、筛接触、挑根或换阈值。

prefix raw64 来自 sealed `p5_jepa_factual_nonzero_20261027_v2.SCHEDULE[:10]`：
`(.12,.04),(.08,-.06),(-.05,.09),(.11,.03),(-.04,-.07),
(-.07,.10),(.03,.13),(.09,-.02),(-.11,.05),(.06,-.08)`。
frame0..10 每帧 RGB/state/proprio/full_public_state/transition 共55文件逐字节比较，
包括真实 native goal/render/coverage/RNG/history；十个 command 文件也字节相等，
显式 seed 后 RNG 另核。全部11帧 gate 通过之前不得 dispatch suffix。
constructor 未显式 seed 前随机种子另记，不错误要求它与旧 constructor 一致。

suffix 直接取实际 v2 prepare 的 raw float64[2,5,2]→[10,2]，不从 normalized 逆算：
X 为五步 [.1,0]，Y 为五步 [0,.1]；四 word 是 XY/YX/X−X/全零。
单位原生相对 u，内部目标位移100u；raw0不是暂停，仍真实调用 physics。
禁止 normalized padding、额外 replay、插入 goalRGB 或 goal setter。

## 实际 authority 与科学边界

源码 PINNED 固定已有 producer/source、原 prepare、actual CUDA report、MAIN wrapper、
exact strict receipt SHA；prepare 重算原 producer.consume（仅 saved NumPy）并要求精确等于
MAIN 保存 JSON，随后绑定完整来源 closure。不是拿 receipt physical_release=false 当执行许可。
run 还须 MAIN 显式 `--confirm-physics` 和实际本模块 prepare FILE SHA。

- CUDA report：`cdbccbdf894d8d6e7c078e02f7078bd8cd26e802c9a00d59421ad0e4b0883bbd`。
- strict receipt：`5237000615227a5cb6741e19c3feb4414c710077983399c46804b96c223b01db`。
- 已有 v12 owner compact digest 仍由原 v2 边界处理，不改旧 digest/hash。

原 native `_setup` 定义 goal `[256,256,pi/4]`、coverage 阈值 .95，原 RGB 含该 goal。
这些是已认证的**重放环境设置**，不是独立 task-goal、接触前提或本轮任务成功协议认证。
明确 blockers：独立 task-goal/protocol certification；预声明 contact/action-task witness acceptance。
task_success 永远 null，scientific_admission/P5_admission/action_witness_established 永远 false。
允许 MAIN 独立检验物理 replay/action-order 差异，但不由此宣称任务成功、控制优势、fresh 证据。

绑定源 `pusht_env.py:505` 的行为是 **done=False**，后面的 coverage 比较只是注释。
saved validator 严格要求 transition.done **is False**，不能从 coverage>.95 发明 done。
coverage、reward（native clip(coverage/.95,0,1)）、done 三者分开记录；覆盖率即使1也不
升级本研究 task_success。异常 native done 在 runner 保留早停、绝不补步；它违反当前绑定源
合同，strict saved validator 不会把该 partial 当原生合同通过。

## 保存与计量

每分支完整保存 frame0..20 的五文件，checkpoints0/5/10/15/20均为实际返回，
native7=`[agent_x,agent_y,block_x,block_y,block_angle_mod2pi,agent_vx,agent_vy]`，
proprio4=state[[0,1,5,6]]。full public body 保留原精度、block native 未 wrap angle；
角度距离仅用 atan2(sinΔ,cosΔ)描述原 goal，不拟合 readout 或任务门。
每次 step 使用原返回 obs/info，不另调 get_obs/render。原 goal/source/root/word/plan与
predispatch全12清单及实际 prepare FILE SHA 一起封存；不回填未完成 endpoint。

完整执行每分支：constructor1、seed2（含 constructor）、reset1、public control20、
physics Space.step201、render21、get_obs21、close1、setup1、set_state1、Space constructor1、
goal-body41（render21+reward20）、Body constructor44（setup3+goal41）。
全12：constructor/close各12、seed24、reset12、public240、physics2412、render/get_obs各252、
setup/set_state/Space constructor各12、goal-body492、Body constructor528。
所有轴独立，不相加，不叫 FLOPs 或 equal-total-compute；此处神经调用为零。
`base.NativeCalls` 是原 Python sys.setprofile call-entry 观测；Space.step调用不冒充C内部完成。
每个 native 操作 durable intent 在前、durable returned 在后；失败不退款。
`operation_meter` 分别计 durable intents/returns，返回后日志失败属于 indeterminate，不自动重试。
close 另记 attempted/returned/durable_return，保留 original_error；不完整 cleanup 阻断后续分支。
所有原12分母保留。identity/工程失败阻断后续，native早停保留实际步数、不补步。
owned profiler 必须 finally 释放，不覆盖预存 profiler。

128MiB accounted output cap（96MiB payload，8MiB final reserve；最多2048 nodes，每node
保守8192B开销）、200MiB free floor、900s wall，均为工程限制而非科学门槛。
写前预扣，失败写不退款。close和最终失败回执可用 reserve，不借正常预算；磁盘满仍可能
无法持久化，此时没有完整 strict receipt。run 使用 SIGALRM，边界再核 wall/floor；C native
阻塞可能延迟 Python signal，不保证OS实时抢占。最终超wall receipt不会strict通过。
RAM full-layout size回归覆盖252个真实尺寸RGB token文件及全部12journal/row/command；
真实 native public geometry 可能更大，硬cap仍在写前拒绝，不省 bindings/删负结果。

## MAIN 命令（先审读、跑RAM，再执行；worker未执行）

工作目录 `/home/gbk/code/LeWorldModel/le-wm-main`。输出父目录须 MAIN 新建；不覆盖已有目录。

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_agent_p5_action_words_physical_20261028_v1.py
mkdir -p results/agent_p5_action_words_physical_20261028_v1_prepare_MAIN01
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/spectral_reachability/agent_p5_action_words_physical_20261028_v1.py prepare --manifest results/agent_p5_action_words_physical_20261028_v1_prepare_MAIN01/manifest.json
sha256sum results/agent_p5_action_words_physical_20261028_v1_prepare_MAIN01/manifest.json
```

下列 `<ACTUAL_PREPARE_SHA>` 和 `<ACTUAL_REPORT_SHA>` 必须由 MAIN 替换为真实文件SHA，
不得填未来hash；没有这些实际值，命令 fail closed。

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/spectral_reachability/agent_p5_action_words_physical_20261028_v1.py verify --manifest results/agent_p5_action_words_physical_20261028_v1_prepare_MAIN01/manifest.json --manifest-sha256 <ACTUAL_PREPARE_SHA>
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH=/home/gbk/code/LeWorldModel/le-wm-main/.jepa-physics-site .venv/bin/python experiments/spectral_reachability/agent_p5_action_words_physical_20261028_v1.py run --manifest results/agent_p5_action_words_physical_20261028_v1_prepare_MAIN01/manifest.json --manifest-sha256 <ACTUAL_PREPARE_SHA> --confirm-physics --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p5_action_words_physical_20261028_v1_physics_MAIN01
sha256sum results/agent_p5_action_words_physical_20261028_v1_physics_MAIN01/report.json
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/spectral_reachability/agent_p5_action_words_physical_20261028_v1.py consume --manifest results/agent_p5_action_words_physical_20261028_v1_prepare_MAIN01/manifest.json --manifest-sha256 <ACTUAL_PREPARE_SHA> --report results/agent_p5_action_words_physical_20261028_v1_physics_MAIN01/report.json --report-sha256 <ACTUAL_REPORT_SHA>
```

CLI stdout 返回消费对象；MAIN 若需持久消费回执，使用其独占wrapper独占创建保存。
consume不解pickle、不加载权重、不运行神经/physics；完整inventory、branch/journal/command、
native7/public/transition、原prefix及资源计量均复验。partial 明确保留、退出2，不能冒充all12通过。
