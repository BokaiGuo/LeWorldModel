# P2 commoncontroller 四 returned 槽 full15 物理协议 v1

本版仅协议与成本，不是 driver、状态 wrapper、物理许可或科学准入。worker 只读源和小型 saved metadata；未导入模型、构造环境、运行 NN/physics/full strict。experiment-design 仅用于固定对照/预算；不做调参或按收益选择。此前冻结文件均不修改。

## 1. 固定问题与六槽分母

问题仅为：相同真实 root/history/goal 下，两个不同来源 latent target 经同一已冻结 controller 返回的动作，实际执行后如何。不是原三程序 imagined-recompile 的再次执行，也不是原 proposal 生成成本公平比较。

| slot | known root | target 来源 | 本次动作 |
|---|---|---|---|
| 0 | 0/off | prior FIRST | 本槽 actual returned mean 对应原 raw15 |
| 1 | 0/off | program 首 postcondition | 本槽 actual returned mean 对应原 raw15 |
| 2 | 0/off | arbitrary | missing；无环境/动作/伪终点 |
| 3 | 1/off | prior FIRST | 本槽 actual returned mean 对应原 raw15 |
| 4 | 1/off | program 首 postcondition | 本槽 actual returned mean 对应原 raw15 |
| 5 | 1/off | arbitrary | missing；无环境/动作/伪终点 |

六槽原序不压缩，执行序固定 0/1/3/4；执行分支即使相同动作也各自新环境，不继承旧 row。两 arbitrary 缺失不是失败/零收益。两 known roots 不是独立新根。prior FIRST 物理时间未知；program target 来源原 cp5，不代表共同 H3 controller 终态也在 cp5。两来源的 target 非同时间/同生成预算目标；不由本实验消除此限制。

## 2. 已核接口与固定来源

以下路径均相对 `/home/gbk/code/LeWorldModel/le-wm-main/`。SHA 为本次只读核对；不将尚未生成的 Pauli report/receipt 写成实际值。

| 来源 | 路径 | SHA256 |
|---|---|---|
| delivery | results/agent_p2_target_only_delivery_20261029_v1_prepare_MAIN01.json | 82b36fa3030f8ebb2da37bdc91ba3e8d582f129d6bc9e5c9f8e774b363ef3f62 |
| controller report | results/agent_p2_target_controller_cuda_20261029_v1_cuda_MAIN01/report.json | b1358eb3cec42cdddb30b0d138c2bc2e68773f195aa985977bf9c090527382cb |
| controller strict | results/agent_p2_target_controller_cuda_20261029_v1_consume_MAIN01.json | 34e0491ca05a1b381db3541997c6215c44f9927761d6123a559e604160f0be4a |
| controller prepare | results/agent_p2_target_controller_cuda_20261029_v1_prepare_MAIN01.json | 1a021116784dfaa5b4ce976f694fbfde377e6b282ce3553cb57d49a5e033c0aa |
| raw-domain | results/agent_p2_delivered_mean_raw_domain_20261029_v1_prepare_MAIN01.json | 9df31325d7d2bf149b60db0f6d21ce40e6284060c4bb3af29c1699e1ee056c56 |
| 原物理 prepare | results/agent_p2_three_program_physical_20261029_v1_prepare_MAIN01.json | a4fb18b5aedd53e1820717f827d4703774b8c41dd08b0fd3974cca58593ade93 |
| full15 原生模板 source | experiments/spectral_reachability/agent_p2_three_program_physical_20261029_v1.py | b0e7ae15d4810963a64916b1da16e34ae1017b189156a93b635d73520e2666e5 |
| commoncontroller source | experiments/spectral_reachability/agent_p2_target_controller_cuda_20261029_v1.py | 44b87adb98a6ac60e755eee3e5fc544a194ce27c7ddf367f7cfb834d586ad946 |
| 最底层 placement/binding source | experiments/spectral_reachability/p1_moving_target_overlay_source_20261027_v1.py | 768a45d127ad01a0e3ee8b55a6c7c535c92134211af10914e3e99bb4674e472b |

delivery 六 row 实际字段为 root/condition/arm/task_id/task_sha256/blind_sha256/history_sha256/action_history_sha256/controller_payload；**没有 slot 字段**，按固定位置与身份共同认证。controller report 六 row 则有 slot/root/arm/status，实际四 returned、两 missing。不能把两种 row 当同 schema。

controller 只收到 `{z:[1,3,192], a:[1,2,10], target:[1,192]}`。target 不等于原 task goal192，更不是 native7 goal/goal_pose；未来见证未入 controller。原 task goal、RGB、history、两条过去 action tokens 的绑定沿 delivery→parity/program→原 physical prepare 核实；不能把 target 写进 native goal setter。

## 3. 物理准备的硬前置及 Pauli 接缝

Pauli 当前源为 `experiments/spectral_reachability/agent_p2_delivered_mean_paid_replay_cuda_20261029_v1.py`，文档同 stem。当前观察到的 consume 调用为 `consume(manifest,mh,report,rh,output)`；此函数本身写出 receipt，不应冒充无副作用 getter。最终生产 ABI/source SHA 以其实际冻结交付为准。MAIN 执行完整 paid consume；新物理 prepare 显式接收该 manifest/report/receipt 的实际绝对路径和 SHA，再核实际 schema/完整内容及数组来源，不只看 status。若以后要重建 consume，使用其真实可调用验证接口或显式新输出路径，不覆盖原 receipt。

当前实现 receipt 提供 manifest_sha256/report_sha256、denominator/local_checks/module_resources 等，未据此猜造 physical_release/commands_released 等 flags。Pauli 的 raw_deployment_authorized=false 不能被改成 true，也不把其模型 replay receipt 当物理授权；另需 MAIN 对本新协议/driver 明确授权。最终 ABI 变化需显式适配并留失败记录，绝不 `.get(...,False)` 补证据。

放行必须同时满足：

1. **四槽整体屏障**：0/1/3/4 每槽实际 CUDA inverse 及 whole H3 + segmented[1,2] replay 全返回、完整 consume 认证，无 blocked/partial；两 missing 保原样。任一未满足则本批不启动任何物理 env，六槽保留，不能挑其余成功槽替代。
2. 本槽 original selected mean/native_return actions/native_return mean 的 byte 身份已认证；新 raw 为已封 `float32(mul)` 后 `float32(add)` 的 120 bytes，时间顺序 `[15,2]`。有限、闭域 [-1,1]、无 clip/tanh/重选/重采样。实际 CUDA raw 必须精确等于 raw-domain 已封 raw，包括 signed zero。
3. 付费 replay 的输入是实际 raw 经独立 float32 sub/div 得到的 tokens，不要求等于旧 mean；两条 incoming history 不偷换，whole/segmented 的输入/输出与模型 425 before/after/source closure 完整认证。不能复用旧 mean 成本冒充部署 token 成本。
4. 物理 prepare 只导入本槽已认证 raw bytes，不再 inverse/normalize；native 自身 relative action_scale=100 属原 step，不能额外乘除100。公共 mean/std 只是 MAIN 批准的 P2 工程坐标约定，Hi 训练 scaler 身份仍未认证。

## 4. 真实执行链与因果首动作 guard

新 driver 必须自有 schema。复用模板的 place/bind_goal、native_hooks、dispatch、capture、labels/metrics helpers，而不造三程序 `.plan`/edges/receipt，亦不 patch 旧 module globals。模板的 assemble/material 要求原 program plans，不能直接拿新 controller row 填入。

每执行槽固定：

1. 新建原安装 `swm/PushT-v1`，rgb_array、224、with_target=false；原 slot/reset_seed 做 seed reset，取原 placement native goal；用原 root_seed/plan 与真实 block.shapes 的 sample_state 配方，第二次 reset(seed=0, options=原 requested state/goal)。只允许这两个原生 reset 的 state setter，不允许 imagined restore 或额外 native 状态注入。
2. 原10条 warmup 原字节依次 native dispatch，捕获 warmup0..10 的 off/on/off；native warmup 早停就留失败/缺项，不 pad 出 root。goal 尚为 placement goal；每返回 native7/reward/terminated/truncated 按该 phase 的真实 goal 复核。
3. 核 warmup0/5/10 off 三帧精确等于原 packet rgb_history；真实 state7/root_hash/snapshot、可对应的 warm trace 和 renderer 必须匹配源。完整10条 actual/requested raw bits 等于原 warm10；用原 once-normalization 算术认证 incoming action2 等于 delivery/controller/replay 的 a。z/goal 的编码身份依赖原 sealed RGB→latent authority，物理阶段不重编码。
4. 只用 bind_goal 挂接原 task native7 goal 和原 goal RGB，snapshot/goal_pose/renderer 不变。捕获真实 plan cp0，核当前 off 图与原 packet current/history末帧、原 goal RGB/goal192 authority 及 task/source/blind/model 身份一致；不得把 latent target 当 native task goal。
5. 在 **plan step1 的 public_step intent/callback 之前**，持久化并 fsync `initial_history_guard` 接受证据，绑定 source、raw15、history、goal 和上述实际观察。未接受/写失败则绝无 plan step1 callback。还须保留终态再次核验，但不能用 finally 检查替代此因果 guard。原模板现有 episode 中的检查可复用，新的 guard 事件及 strict 时序仍需实现。
6. 原 raw15 一次执行，不根据实际/预测收益反馈重规划；每步保留 requested/actual bits、native7、snapshot、reward、termination、分轴 metric 与 off/on/off 三帧。cp0/5/10/15 取真实帧。原生提前终止立即停，不补末态、不重复上一帧充 cp；即使 cp15 missing 也保留六槽分母。任一步 task goal/renderer/action 身份变化即错误，不放宽完整性。

## 5. 预算、日志与严格消费要求

下表是**四槽均 full15 的新增物理上限**，非已执行计数。沿模板原生 callback hook 真实 attempted/completed 计账；失败不退款，未知不填0。

| 轴 | 每执行槽 | 四槽上限 |
|---|---:|---:|
| constructor / close | 1 / 1 | 4 / 4 |
| reset | 2 | 8 |
| public_step (warm + plan) | 10+15 | 100 |
| physics_step (reset + public) | 4+10×25 | 1016 |
| set_state / goal_setter | 4 / 3 | 16 / 12 |
| goal_render / goal_bind | 2 / 1 | 8 / 4 |
| explicit_render / PNG | 3×(11+16)=81 | 324 |
| 新 NN/encoder/predict/AD/训练 | 0 | 0 |

cp0 与 warmup10 是两次实际 triplet，不能为节约预算去重。wall900s；输出总64MiB含8MiB失败/终态 reserve，floor200MiB；完整 prepare metadata 上限128MiB。初始磁盘空间须同时留输出 cap+floor。guard 若加入 metadata 超出任一事前 cap，prepare 失败，不能执行后扩容掩盖。清理/越时/IO故障保留真实 partial；无法完整日志闭合的不签完整 receipt。

上游成本单独完整继承，不报本物理新调用：原四 controller cap48 predict API/216 samples；Pauli 全四合格付费 replay 计划24 predict API/24 samples、24 AE/24 low projection、96 AE tokens/72 predictor context tokens/72 projection rows，encoder/high/AD/J/VJP/physics均0。只把完整实际 ledger 加总为实际值；不能把 cap 或 unknown FLOPs 改成实际计数。新物理0 NN不代表整个研究0 NN，也不宣称 equal FLOPs/同生成成本。Pauli CUDA的512MiB输出/64MiB reserve、900s scope/180s每槽、GPU3/4GiB分开保留，不并入物理wall或当已付费完成。

完整物理 consumer 需核 own schema、全部事件白名单/hashchain/操作嵌套时序、首动作 guard在callback之前、每次实际 native labels 的 phase-goal、所有raw bits、source/task/receipt/sourceclosure、PNG/file/pixel/snapshot/renderer、文件全集与实际资源 census；两 missing 无任何 env/callback。先真实返回再报资源完成，不因早停丢槽。MAIN 将实际执行 full consume；worker不运行。

## 6. 固定结果读法与真正阻塞项

每 root 固定比较 program-target vs prior-target，cp0/5/10/15 和真实 terminal 分栏；native success、jointXY、wrapped angle、velocity、full7 reward/distance 各列保留，native termination/truncation 单列。成功门不含velocity但 full7 距离含velocity；两者不替换。缺 checkpoint 的定时对比为 null，earlystop terminal 不充 cp15。latent target attainment 若以后加入须单独真实 RGB 编码协议，不能伪造 latent 或用 native pose代填。不能据物理结果挑 target、动作或时刻。

| 当前阻塞 | 解阻所需实际输入/动作 | 不代表什么 |
|---|---|---|
| paid replay 尚无本次可核实际report/receipt（本协议写入时） | Pauli冻结最终ABI与sourceSHA，MAIN实际四槽运行+全consume，外部提供实际manifest/report/receipt SHA | 不是模型/物理负结果，也不臆测未来flags |
| 新 own-schema物理adapter尚未实现/MAIN验收 | 真实receipt字段逐一接线、RAM全生命周期及guard失配绝无step1回归、MAIN授权 | 不是拿原template rename即可运行 |
| same warm/history/current/goal的实际因果guard未执行 | MAIN新四env真实replay并认证step1之前的guard；不匹配即保留拒绝 | saved metadata相同不等于真实环境已匹配 |
| Hi training scaler / prior物理时距 / arbitrary target仍未知 | 原始训练来源或另行明确科学协议；本工程slice不补造 | 前两者限制相应科学解释；两missing禁止替代 |

工程执行满足前置后可由MAIN独立授权；所有 P1/P2 scientific admission、freshness/独立根确认保持 false，付费replay本身不发物理许可。不生成“ready”wrapper，也不在本协议提供不存在driver的CLI。后续最小实施就是自有schema的四槽full15适配器与guard/RAM tests；等待Pauli实际ABI与MAIN指令后再实施。本次仅此新文档，未生产prepare或receipt。
