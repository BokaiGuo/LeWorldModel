# Reacher D2P：原始冻结模型 RGB successor 排序复核

## Material Passport

- 对象：冻结视觉世界模型对实际 action packet 的 successor 排序，而非 qpos 分类或训练收益。
- 状态：`implemented / validated`（诊断代码和本页 RGB 回执）；证据桶 `recovery / exploratory`。
- 采样单位：初始 8 根对照与独立固定的 32 根 source-state/history 复核，每根 128 个固定 source packets。
- 主张边界：source-episode diagnostic，**不是 episode-held-out P1 calibration/outcome**；不自动打开 P2。
- 数据与计算：官方 SHA 绑定 HDF5；所有神经模型计算均使用主机 CUDA，不回退 CPU。

## 先纠正原结论

2026-10-26补充固定旧根因果证据：原六根margin校准实际使用joint-repair权重，
不是官方原始权重。同root/goal/动作/后继下，官方6/6、joint-repair4/6；仅恢复
joint-repair预测头BatchNorm运行统计即6/6。不能再将原两个反转概括成原始冻结
模型边界。该具体退化已定位并增加训练防护，但不等于P1机制通过。见
[预测归一化根因与修复](D2P_PREDICTION_NORMALIZATION_REPAIR_20261026.md)。

“Reacher 的冻结 checkpoint 无法稳定排序真实 RGB successor，所以是 checkpoint 的可靠性边界”
表述过宽。当前证据证明：在显式修正的 source-native root/history/goal 合同下，**原始发布权重**
无需训练、无需 qpos readout、无需 latent adapter，能够对这一诊断 panel 的真实 RGB successor
进行高一致性排序。该事实不支持把旧负结果概括成理论失效或无条件 checkpoint 失效。

同时，不能说“已找到解释所有历史失败的唯一 bug”。旧 panel 与本 panel 的 root 支持、history
和 goal 合同不同，多个修复共同改变了输入条件，尚无固定旧根的逐因素消融来分离它们的因果贡献。

## 已确认的代码和协议缺陷

1. 旧 runtime-support 审计曾误把 `_prepare()` 第二返回值 goal tensor 当成实际 successor。
   已改为执行真实 raw packet，并从 shifted context encode 实际 RGB；原数值不能解释模型 fidelity。
2. 从官方 HDF5 恢复 source-state/history 后，曾保留随机 reset root 的 goal。
   现以 source root 原子地重建 target qpos、目标图像和执行 snapshot。
3. goal 事务若在 history 渲染后才绑定，history 与 root/goal 的可见场景合同不同。
   当前 `official_source_state_history_v3_goal_bound_before_history_render` 在 history 渲染前绑定 goal。
4. “来自 source 的 action”不等于完整 state/history 来自 source 分布。旧 action-only warmup
   只满足 action marginal，不能用它的阴性去否定 source-native 条件下的冻结预测能力。

这些修复分别属于观测实现、任务实例绑定和输入支持合同；不能靠提高 margin 阈值替代。

## 同根、同动作、原始 RGB 度量的主机 CUDA 证据

审计程序：
`le-wm-main/experiments/spectral_reachability/audit_reacher_d2p_source_native_rgb_rank.py`。
回执：
`le-wm-main/results/reacher_d2p_source_native_rgb_rank_original_vs_repair_20261025_cuda.json`。

固定 roots `202621000..202621007`；每个候选恢复完全相同的 root/goal snapshot，执行相同 raw
packet。两套 checkpoint 共用真实 successor RGB，不按物理结果选择 bank 或调参。预测逐候选
使用 singleton 路径，实际 successor 由执行后的 shifted context 编码；两侧均以同一 encoder
与同一 goal RGB 的 latent MSE 计分。没有 qpos 读出，也没有训练或权重更新。

| 权重 | argmin 与实际 RGB 最优一致 | mean Spearman | min Spearman | 所选动作相对 zero 的方向一致 |
|---|---:|---:|---:|---:|
| 原始发布 Reacher LeWM | 8/8 | 0.971824 | 0.858863 | 8/8 |
| 既有 joint-repair checkpoint | 7/8 | 0.940073 | 0.846657 | 8/8 |

原始权重 SHA256：`eb70b1fd5409f8f81875d62f5ee5a20dd220a3128a477de66b5760f475f0f469`。
joint-repair 权重 SHA256：`d68e2da61d4b64fa01ddb5c476cc6fc975ffa4ff750a1efe4822a4516cdf009d`。
不能把既有 joint-repair + qpos-readout 的另一份 rho=0.976 回执误写成原始发布权重的原始 RGB 结果。

该结果的最窄表述是：**冻结的原始模型在当前 source-native 诊断合同下具有可用的 RGB 候选排序能力**。
不是“所有 live reset 都泛化”，也不是“非贪心重配置已胜过贪心”。

## 尚需验证的实际 P1 机制

最优动作的 argmin 准确，不保证通过 future reachability 选出的非贪心 prefix 也保序。
因此另执行原始权重、无 readout 的 source-native same-bank P1 diagnostic：
`le-wm-main/results/reacher_d2p_source_native_original_rgb_p1_samebank_20261025_cuda.json`。
实际得到 8/8 model-eligible roots，8/8 prefix 相对 same-bank greedy 在真实 RGB 和 evaluator
qpos 中均非贪心，两坐标的方向一致。但相对 raw-zero，真实 RGB 非贪心方向只过 6/8；
两种 comparator 不能混写。两段执行的 final qpos 在 8/8 根均劣于同库贪心，描述性平均差
`reconfiguration - greedy = +0.80349`。没有闭环收益通过。

### 新发现并修复：zero-only future witness 不代表重配置有用

逐 root 检查 model-side `proposal_rows` 表明：旧 selector 选择的重配置，其**模型自己预测**
的 bounded-DLS future cost 已比 same-bank greedy 高 `1.1271..2.0390`，却仍因胜过 zero-prefix
被标记为 witness。这是有用性参考合同缺口，不能将该选择后的劣势解释为理论反例。

新增 `reacher_d2p_reconfiguration_reference.py`，并把显式
`--future-reference same_bank_greedy` 接到 compiler、preflight、正式 P1 runner 和 source diagnostic。
它保留原非贪心/zero-prefix 约束，**额外**要求同一 bank 中 immediate-greedy prefix 的未来
优化代价也被严格改善。参考只来自冻结模型 scores，物理 qpos/RGB outcome 不进入选择。
底层 compiler API 和历史 diagnostic 保留 `zero_prefix` 兼容行为，供旧协议复现；正式
preflight / P1 runner 的 CLI 默认改为 `same_bank_greedy`，需复现旧协议时显式传入
`--future-reference zero_prefix`。不能默称历史结果采用了新合同。

新合同的主机 CUDA 回执：
`le-wm-main/results/reacher_d2p_source_native_original_rgb_p1_strong_reference_20261025_cuda.json`。
同一 8 根、同一权重、同一 bank 上为 **0/8 witness**：原非贪心约束范围内的候选全部被
正确拒绝，完整 model rows 与拒绝原因仍写入回执。这是避免错误 witness 的修复，不是 P1
成功，也不是已经证明所有 bank/所有 root family 均无重配置价值。

新 reference 字段绑定到正式 preflight/calibration/outcome 以及 selected-packet 训练合同。
`same_bank_greedy` 运行不能借用旧 `zero_prefix` 证据；缺少明确字段的旧 receipt 只按
legacy zero-prefix 解释。该协议变化必须从新 preflight 开始，不能就地开启 P2。

### 32-root 确认中发现的执行级问题

另外固定 `202623000..202623031`、只用原始权重的确认，在第 9 根 `202623008`、候选 index=62
两次中断于严格 RGB root identity 检查，而非 ranking gate。第二次运行将前 8 根逐根保存于
`le-wm-main/results/reacher_d2p_source_native_original_rgb_confirmation32_v2_20261025_cuda.roots/`，
失败现场保存在相邻 `.failure.json`；这些 partial roots 不能代替完整 32 根确认。

具体现场为 224×224×3 RGB 中仅 **1 个颜色分量相差 1 灰度**，而 qpos/qvel、旧 `get_state()`
向量、target qpos/marker 均为零差异。单根无模型调用的 128 次 restore/render 诊断未复现，
所以不能把它草率写成物理根重放错误或神经模型预测错误，也没有放宽像素阈值。

查 dm_control 实现进一步发现，`physics.get_state()` 默认实际只包含 qpos/qvel/act/plugin，
并不包含 time、ctrl、applied forces 与 solver warm-start。当前 snapshot 已改为为真实 MuJoCo
对象额外捕获/恢复 `mjSTATE_INTEGRATION`，包括临时 goal render 前后两次恢复，并保留旧向量
供诊断和历史回执兼容。该状态遗漏是实际代码合同缺陷；**尚不能仅凭发现遗漏就断言它造成了
那 1 灰度差异**，必须看同根 v3 重跑。

v3 重跑 output（仍在相同 root/candidate 处中断，integration state 也为零差异）：
`le-wm-main/results/reacher_d2p_source_native_original_rgb_confirmation32_v3_integration_20261025_cuda.json`。
该路径没有最终成功 report，现场在相邻 `.failure.json`；integration 修复未消除像素中断。

在此基础上修复 goal replay 事务：snapshot 现在额外保存原始 goal RGB 的压缩字节和 SHA，
恢复时验证 hash 并精确重放该不可变观测，不再在已经通过 root 检查后临时切到另一个物理姿态
重新渲染 goal，再切回 root。这使原始 goal 观测与 root 检验分离，**未放宽任何像素阈值**。
没有绑定 goal RGB 的历史 snapshot 仍走原显式重渲染路径供审计兼容。
v4 同根重跑：
`le-wm-main/results/reacher_d2p_source_native_original_rgb_confirmation32_v4_bound_goal_20261025_cuda.json`。
它仍在第 2 根、候选 index=33 的冗余第二次 root render 处中断，1 个颜色分量差 1，完整
integration/target 状态零差异。因此保存 goal RGB 本身也不是该中断的充分修复，失败现场保留。

v5 所尝试的执行路径使用**接力式精确 identity proof**：`restore_execution_snapshot()` 得到实际
root RGB 后，调用者仍必须先与封存 root **逐像素完全相等**；紧接着的
`advance_executed_context(..., verified_root_frame=实际已检查图像)` 不再二次渲染，而是新鲜
捕获并精确比较 integration vector/spec、完整编译 MuJoCo model binary SHA、target qpos 和
绑定 goal RGB。物理状态、相机、材质、几何任一变化都会拒绝；不完整的历史 snapshot 不允许
走该路径。这个变化消除冗余 OpenGL 数值依赖，既不增加像素容差，也不改变 RGB 排序度量。
默认不传该 proof 的旧 API 仍执行原严格 RGB 检查，不能凭旧截图绕过新鲜状态验证。

此路径已接入原始 RGB audit 与正式 P1 two-stage 的 method/zero 执行；formal caller 的第一次
实际 RGB 检查仍由 `_base_distance()` / `_verify_execution_identity()` 完成。
对应未完成时不得预报的确认输出：
`le-wm-main/results/reacher_d2p_source_native_original_rgb_confirmation32_v5_exact_root_proof_20261025_cuda.json`。
v2/v3/v4 失败和 partial roots 均保留，不能将它们重命名成 v5 成功证据。

v5 已实际失败：第 1 根 `202623000`、candidate=98，在**第一次**严格 root/goal RGB 比较处
中断，完整 snapshot 完全相等。因此“消除第二次渲染即可解决”的推断不成立。

v6 显式拆分 singleton CUDA prediction、物理 RGB 采集、observed RGB encoding 三个阶段，
仍在相同根、相同 candidate=98 失败，未完成 32 根。新增逐像素现场表明：goal 完全相等，
root 仅 `(56,167,0)` 分量差 1；不执行动作、不改变状态，再连续渲染 4 次均与原始根逐像素
完全相同。对应 `.failure_rgb.npz` 保存原始/异常/重复根和 goal，`.failure.json` 保存精确
状态及差异坐标。故“单纯推理/渲染交错造成中断”没有被该分阶段实验支持，不能再据此盲目重跑。

### 显式的新诊断合同：观测不等于反复测量

原始 root RGB 是已经真实测量并输入规划器的固定观测；同一物理状态的后续 GPU rasterization
不应被要求总能重现完全相同的量化字节。新增可选 `--root-identity sealed_observation_exact_state`：

- snapshot 额外绑定首次实际 root RGB 的压缩字节与 SHA；没有该字段的历史 snapshot 拒绝使用。
- 恢复前验证 sealed RGB hash；恢复后逐项**精确相等**核对完整 integration state/spec、完整
  compiled model SHA、目标 qpos、可见 marker 与绑定 goal RGB；任何遗漏或不匹配都拒绝执行。
- 仅在证明同一执行状态/场景后，复用已测量的同一根观测，不重新合成它。
- 每一个动作后的 successor RGB 仍由当前真实环境新鲜渲染、编码、计分，绝不以预测图或缓存
  successor 替代；排序、argmin、权重和 candidate bank 均不改变。

这是一项明确的 diagnostic protocol 变化，不是把像素阈值从 0 调到 1，也不是旧 `exact_render`
合同的成功复现。默认仍为旧严格渲染模式；正式 P1 链并未静默切换此新合同。v2–v6 中断全部保留。

固定 32 根、原始权重的 v7 已完整完成（主机 RTX 5060 Ti，CUDA 13.0，权重未改）：
`le-wm-main/results/reacher_d2p_source_native_original_rgb_confirmation32_v7_sealed_observation_20261025_cuda.json`。
根集合仍为事先固定的 `202623000..202623031`，无删根、无按结果选择候选；共 4096 个实际 RGB successors。

| 复核对象 | argmin 一致 | mean Spearman | min Spearman | 模型最优相对 zero 的方向一致 |
|---|---:|---:|---:|---:|
| 原始冻结权重，32-root source-native diagnostic | 28/32 | 0.972377 | 0.882811 | 32/32 |

注意最后一列检验的是模型最优动作，而不是 P1 刻意非贪心的 selected prefix；不能代换两者。
`completed` 表示这一显式诊断合同完整执行且 integrity gates 通过，**不是正式 P1 pass**。

旧 v2 严格渲染运行已完成的 8 根，与新 v7 按 root ID 逐一对照，root/goal hashes、source
episode/row、proposal receipt、raw packets、全部 successor RGB hashes，以及所有 singleton
predicted/observed costs **全部精确相同**，两种 cost 最大差异均为 0。
不可变等价回执：`le-wm-main/results/reacher_d2p_strict_vs_sealed_rgb_equivalence8_20261025.json`；
实现为 `audit_reacher_d2p_rgb_protocol_equivalence.py`，不按 JSON 数组顺序配对，拒绝缺根和重复根。
该等价证据只覆盖旧运行实际完成的 8 根，不追认其它中断根的旧严格渲染合同。

因此已解决本次 32-root 审计的伪中断，并独立确认原始 checkpoint 在当前输入合同下具有良好的
真实 RGB 候选排序能力；仍不能把该结果推广到任意 live reset、P1 selected prefix 的保序或重配置收益。

### 同 32 根继续核对 P1 编译器代码路径

新增并完成主机 CUDA 审计 `audit_reacher_d2p_rgb_compiler_alignment.py`；不可变回执：
`le-wm-main/results/reacher_d2p_original_rgb_compiler_alignment32_20261025_cuda.json`。
它逐 root 复原相同 source episode/row、root/goal 图像 hash、同一 bank/原始 raw packets，
不读取物理 outcome 来选择动作；全部 32 根、128 候选均纳入：

- singleton predicted costs 与 v7 参考逐项精确一致（max error=0）。
- P0 terminal 与 P1 prefix 推进的终态 latent 精确相同（max error=0）。
- 三段整段 rollout 与“prefix 后重编译两段 suffix”的终态 latent 精确相同（max error=0）。
- batch 与 singleton 成本存在浮点数值差异，最大 `7.152557373046875e-7`；但 32/32 根的 argmin
  不变，全部候选相对 zero 的方向翻转计数均为 0，不能将它归为当前 panel 的选错动作原因。
- 原始权重 hash 与参数 hash 均未改变，四项 alignment gates 全部通过。

这排除了该固定 panel 上的这几条具体实现路径，并不证明不存在其它实现错误，也不是对历史
6-root selected non-greedy panel 的复现，更不是“理论负结果”。P1 当前剩余问题仍是：在预先
封存的任务/根分布上，产生**相对同库贪心确有模型 future gain**的非贪心 witness，然后验证
其真实 successor 及闭环收益。不能拿本页最优动作的 32/32 zero 方向一致替换该要求。

本轮集中回归为 146 项逻辑测试通过，`git diff --check` 通过；逻辑测试与以上两项真实 CUDA
实验分别记账。代码调试技能要求在多次同类失败后改变定位方式，本轮因此保留 v2–v6 现场，
先做分阶段判别，再引入明确的新观测合同和逐根等价审计，没有通过放宽门槛或覆盖旧结果止错。

即使该 diagnostic 通过，正式 P1 仍需预先封存的 controllability-gap root family、与训练来源
相隔离的 calibration/outcome、同 bank greedy 与连续 D2P/CEM 强对照、匹配资源，以及真实闭环收益。
目前 production P1 仍默认 executed history；不能借本 source-native diagnostic 的结果，静默更换
其 root law 或挪用旧 preflight/calibration 回执。这里保留这个边界，而不是修改门槛强行开门。

## 后缀优化器与预算计数修复（2026-10-25）

继续代码审计找到了此前未排除的实际错误。因此不能再把“不是尚存代码错误”当作已经证明的
全局结论；已通过的时序、动作尺度和重放审计只排除了各自覆盖的具体路径。

### 错误一：无条件执行完整 DLS 步长

旧 `_reachability_value` 对零后缀做一次线性化，经 `tanh` 约束后，无论非线性模型目标是否
变差，都接受完整 DLS 更新。小阻尼下可能发生严重过冲；候选优于一个被过冲损坏的 greedy
对照，不是 reconfiguration 机制证据。

固定相同的 8 根 `202621000..202621007`、128 候选、原始冻结 checkpoint，事先固定
`{0.01, 0.1, 1, 10}` damping 网格，完成新的主机 CUDA 对照：

| damping | 旧完整更新的目标上升次数 / 1024 | 旧强 witness 根 | 修复后强 witness 根 |
|---|---:|---:|---:|
| 0.01 | 412 | 2/8 | 1/8 |
| 0.1 | 301 | 0/8 | 0/8 |
| 1 | 5 | 0/8 | 0/8 |
| 10 | 0 | 0/8 | 0/8 |

修复固定比较步长 `{1, 1/2, 1/4, 1/8, 1/16, 0}`，以同一非线性模型目标选最小值；零更新
精确复用线性化起点预测。返回与执行的参数也是被接受的缩步，不能只修改报告分数。所有
4096 个候选–damping 更新均不使目标上升；其中 181 次回退到零。逐根身份配对后，新代码
记录的 full-step 成本与旧结果全部精确相同，说明变化来自接受规则，而非换根、换模型或换分数。

具体地，根 `202621001`、greedy 候选 34、damping=0.01：旧 full-step 代价 `1.18344092`，
零更新 `0.57941973`，新接受值 `0.42562842`；候选 17 的 `0.47605896` 不再优于 greedy。
根 `202621004` 仍有 3 个模型侧 witness，最优候选 62；这只是后续物理诊断的对象。

不可变产物（以下均在 `le-wm-main/results/`）：

- `reacher_d2p_source_native_model_gap_damping_grid8_20261025_cuda.json`：旧网格与 `.implementation/` 原实现。
- `reacher_d2p_source_native_model_gap_backtracking_grid8_20261025_cuda.json`：新 CUDA 网格；附预封存 `.protocol.json` 与逐根 `.roots/`。
- `reacher_d2p_suffix_optimizer_repair_comparison8_20261025.json`：逐身份因果对照，full-step 最大误差 0、接受代价最大上升 0。

**归因边界：damping=10 的旧网格没有目标上升；因此这一优化器错误不能直接解释历史
damping=10 的六根 selected-prefix RGB 排序反转。** 它确实影响小阻尼候选选择，但“找到
代码错误”不等于“所有负结果均已解释”。这也不是 D2P 理论的反证或证明。

### 错误二：用奇异值数量替代反向计算行数，并漏计前向调用

原记录将 `len(singular_values)=20` 计为 VJP 行数，实际 reverse-mode Jacobian 的输出维度
为 192，低报 9.6 倍；这是逻辑 VJP 行数，不是声称物理 GPU kernel 调用 192 次。前向记录也
漏掉起点、Jacobian primal、候选重放/评分和线搜索。现逐次计 terminal primal，分开报告
sample-equivalent rollout、实际 `model.predict` 样本量与逻辑 VJP 行数。

新主机 CUDA 插桩审计 `reacher_d2p_compute_accounting_full128_20261025_cuda.json` 已通过：

- 固定首根完整 128 候选编译：实际 `model.predict` 调用 1921 次、预测样本 2048，与回执样本数精确一致。
- 单独两步后缀编译：14 次调用/14 个预测样本，与回执精确一致。
- 接受参数再次实际预测所得目标与回执一致；模型参数前后哈希不变。

该审计修复了计数，但**没有证明强基线与 D2P 已匹配计算资源**；正式 P1 仍需独立资源合同。

### 防止旧校准回执被新求解器复用

默认后缀求解器现为 `bounded_single_step_dls_backtracking_v2`。preflight、proposal-training
合同、margin selection 和 calibration/outcome 合同显式绑定该版本；正式 outcome 额外
核对选择器源文件 SHA。缺少版本、旧 full-step 版本或不同选择器 SHA 均拒绝，不把旧实验
追认为修复后的通过结果。主线仍只用原始冻结模型；训练 repair/readout 不进入这个主张。

这组代码错误使“尚存代码错误已全部排除”的措辞失效，但当前普通 Reacher root family
也尚未被证实满足提案要求的 controllability-gap 前提。必须继续区分实现修复、模型侧
候选存在、真实 successor 排序、以及等预算闭环效益，不能通过调阈值打开 P1。

### 修复后的真实执行与完整状态恢复问题

固定旧 8 根的主机 CUDA 成对执行已完成：
`reacher_d2p_source_native_backtracking_p1_damping001_diagnostic8_20261025_cuda.json`。
仅根 `202621004`、候选 62 model-eligible；其首段在真实 RGB 与 qpos 上均比零前缀和
同库贪心更非贪心，但两段 final-qpos 为 `0.1196762863`，优于 greedy `0.2610154684`
和 zero `0.5370733289`。这是一个真实非贪心收益案例，不是 P1 全面通过。

未产生 witness 的 7 根没有删除；若按原合同使用已实际执行的 neutral-prefix fallback，
整个 8-root 诊断策略相对 greedy 的平均 final-qpos 差为 `+0.4799199753`，7 负 1 胜。
因此只展示那 1 根会造成严重选择性报告。完整汇总为
`reacher_d2p_backtracking_physical_diagnostic8_summary_20261025.json`。

随后预封存 32 根 `202624000..202624031`、damping=.01、新 solver 的确认；它们来自
32 个不同来源 episode，均不与前述 8 根及原 RGB 确认 32 根的合计 40 个 episode 重叠。
这只代表与先前诊断 episode 隔离，不代表原 checkpoint 没见过官方预训练数据。

第一个确认在第 5 根 `202624004` 发生完整 integration-state 校验失败，前 4 根现场与
原始协议保留；这是执行代码中断，绝不能统计为理论失败或将 partial 直接称为 32-root 完成。
逐字段主机 CUDA 复现 `reacher_d2p_snapshot_state_seed202624004_before_20261025_cuda.json`
显示，恢复后的 root/goal RGB 均精确一致，唯一变化来自 `mjSTATE_WARMSTART`：
`-61.81826524185766` 变成 `-61.818265241857645`，差 `1.4210854715202004e-14`。

根因是 `restore_execution_snapshot` 在恢复完整 integration vector **之后**调用
`physics.forward()` 更新派生量，而 `mj_forward` 会改写因果求解器 warm-start。修复为
先 forward 更新派生量，再恢复完整 integration vector，不再次 forward；不放宽任何
状态阈值。已有 bound-goal 与旧临时 goal-render 两条分支都覆盖，回归测试显式模拟
forward 改写 warm-start。修复后同一根 4/4 次完整 snapshot、root/goal 均精确一致：
`reacher_d2p_snapshot_state_seed202624004_after_20261025_cuda.json`。

完整同 32 根以新的 `...confirmation32_v2_exact_state_20261025_cuda` 输出重跑；不只
重跑有利根，不合并旧 partial，不更换模型、margin、damping 或候选分布。新协议额外绑定
context-module SHA，保留实现快照。v2 已在主机 CUDA 完整完成全部 32 根，所有完整性 gate
通过，模型权重未改变；第 5 根也不再中断。旧中断前 4 根的执行动作、observed-first-chunk
评分、物理距离及执行身份与 v2 重测逐项精确一致。

最终回执：`reacher_d2p_source_native_backtracking_p1_damping001_confirmation32_v2_exact_state_20261025_cuda.json`；
全根汇总：`reacher_d2p_backtracking_physical_confirmation32_summary_20261025.json`。

| 32-root 确认指标 | 实际值 |
|---|---:|
| 模型侧强 witness / abstention | 6 / 26 |
| eligible 首段真实 RGB 非贪心：对 zero / 同库 greedy | 5/6 / 6/6 |
| eligible 首段真实 qpos 非贪心：对 zero / 同库 greedy | 1/6 / 5/6 |
| eligible 两段 final-qpos 同时优于两个对照 | 5/6 |
| 全部 32 根、按原 neutral fallback，相对 greedy 胜 / 负 | 5 / 27 |
| 全根 final-qpos 平均差：对 greedy / 对 zero | +0.465292 / -0.101706 |

因此已确认：非贪心两段收益不再局限于旧的单根；但 RGB 非贪心不等同于任务 qpos 非贪心，
且原 neutral fallback 的全根策略仍明显劣于 greedy。**不能只保留 6 根，或把 5 个受益案例
写成正式 P1 通过。** 当前还需独立审计任务目标度量与回退策略，并完成预声明
controllability-gap 任务、资源匹配的强基线和独立 calibration/outcome；不是已证明的理论失败。

补充审计还发现一个日志合同缺陷：上述旧诊断虽实际传入 source packet bank，嵌套
`model_selection.proposal_distribution` 却继承默认 `physical_uniform`，而外层协议和
`proposal_support` 正确绑定 source packets。现调用端显式传入 source distribution；提供
同一 bank 时动作选择完全不变的回归测试已覆盖。原始回执不改写，以上解读以真实 bank
和外层 source receipt 为准。后续正式 preflight/margin/calibration/outcome 同时绑定
context-module SHA，拒绝把精确状态修复前的校准直接挪用到修复后代码。

本轮最终集中回归为 173 项逻辑测试通过，`git diff --check` 通过；测试与上述真实
CUDA/物理结果分开记账。主线 P1 仍未解锁，整套研究目标仍未完成。

## 目标度量与回退调度的独立审计

### 精确目标未绑错，但 RGB 代价不等同于角度距离

`reacher_d2p_goal_metric_geometry32_20261025_cuda.json` 已在主机 CUDA 完成。
使用上节 32 个已知诊断根，仅检查目标接口和观测几何，不再选择动作或拟合任何 metric：

- 32/32 将物理关节设到评测器目标时，重新渲染的图像与绑定 goal RGB 逐字节一致；
  原始冻结 encoder 的 self-goal 代价全部精确为零。
- 固定 5 个半径、8 个角向，共 1280 个真实渲染观测；1024 个同角向相邻半径比较中，
  出现 5 次 RGB 代价径向反转，分布在 4 个根。物理 wrapped-qpos 距离实现已检查。
- 每根结束均恢复完整 integration/scene 状态；原模型参数不变。

这排除了这批根上的目标绑定错误，**没有证明 RGB 与 qpos 是同一个任务度量**。
这一几何差异直接存在于实际观测 encoder，而非模型未来预测中；不能把两坐标不一致一律
称作 successor 预测代码错误，也不能据此判定 D2P 理论失败。此审计复用已有 source SHA
回执及逐根 RGB 身份，明确没有重新计算整份 93GB HDF5 的 SHA。

### 找不到重配置，不应自动意味着停止进展

旧 neutral fallback 是原 P1 的显式对照协议，并非隐藏的随机执行错误。它作为完整 runtime
的默认调度策略却有明显缺陷：当 compiler 拒绝重配置时，仍有模型支持的 goal-progress
动作可执行。因此新增独立策略
`model_only_reconfiguration_else_same_bank_progress_v1`：存在强模型 witness 时保持原选择；
否则选择同一 bank 的 immediate-model-cost 最小动作，而不是 zero prefix。

选择只读取冻结模型 proposal rows，不读取 RGB/qpos outcome；不修改 witness 资格、margin、
旧 neutral 控制或 P1 gate。回执将 fallback 明确记作 `same_bank_goal_progress`，不冒称
reconfiguration。未知 metric、旧 suffix solver、错误 greedy 身份和无正向未来收益的
witness 均拒绝。

已完成逐身份离线轨迹重放：
`reacher_d2p_program_dispatch_trace_replay32_20261025.json`。
原 32 根的 6 个 witness 全部保留；26 个 abstention 改路由到已执行的 greedy branch：

| 相对同库贪心的全根终态 qpos 差（越小越好） | 均值 | 胜 / 负 / 平 |
|---|---:|---:|
| 原 neutral-fallback 策略 | +0.465292 | 5 / 27 / 0 |
| 新 progress-fallback 策略的轨迹重放 | -0.033304 | 5 / 1 / 26 |

两者差值为 -0.498596。**这只是开发后的同根分支重放，不是独立泛化、正式 P1 通过或
资源匹配收益。** 它定位了旧全根劣势的一个运行时因素，不声称解释所有 selected-prefix
排序反转，更不改写旧实验。

新主机 CUDA 诊断已完整完成固定 roots `202626000..202626031`、32 个不同来源 episode；
与此前 72 个 episode 均不重叠。原始权重、128 proposals、damping=.01 和 margin 均不变。
协议在推理前封存，逐根落盘，附实现快照；output 为
`reacher_d2p_program_dispatch_fresh32_20261025_cuda.json`。
每根实际执行 dispatcher、same-bank greedy、zero 和三段前视 CEM 首段，随后统一观察 RGB
并编译相同 DLS 后缀。CEM 固定 128 samples × 5 iterations × 3 horizon；前向工作接近但
VJP 资源不同，**不是完整等预算强基线**。完整回执及独立核账
`reacher_d2p_program_dispatch_fresh32_result_audit_20261025.json` 均已生成；全部完整性 gate
通过，无缺根，权重和封存实现哈希不变，25 个 fallback 的完整动作与 final distance 均与
greedy 控制精确一致。模型 eligibility 为 7/32，未因 outcome 而删除或重新定义。

| 新 32-root 全根对比（dispatch 减 reference） | final-qpos 均差 | 描述性 95% root-bootstrap CI | 胜 / 负 / 平 |
|---|---:|---:|---:|
| same-bank greedy | -0.013170 | [-0.048198, +0.020525] | 4 / 3 / 25 |
| zero-prefix control | -0.671371 | [-0.779314, -0.563683] | 32 / 0 / 0 |
| 三段前视 CEM 首段 + 相同后缀 | -0.253799 | [-0.344226, -0.168758] | 29 / 3 / 0 |
| 旧 neutral-fallback 策略的同根分支重放 | -0.557174 | [-0.687955, -0.424074] | 25 / 0 / 7 |

旧 neutral policy 在这批新根相对 greedy 的均差为 +0.544004；新 progress fallback 移除了
这部分全根劣势。该因果比较来自同根实际执行的 branches，不是跨 panel 拼接。但
**重配置相对同库贪心的增量 CI 跨零**，不能宣称其效益已确立。即便没有重配置，greedy
本身对这份 CEM 对照也有 -0.240629 的均差（30 胜、2 负）；因此不能把 dispatch 对 CEM
的全部差距归功于 reconfiguration，更不能以此替代强基线资源审计。

7 个 eligible 根的首段在真实 RGB 与 qpos 上，相对 greedy 均为 7/7 非贪心，相对 zero
均为 5/7。原 zero-reference P1 的全根机制条件依然未通过。例子 `202626026`：模型先验
认为 selected prefix 的未来代价比 greedy 低 0.010177；但执行 prefix 后，从真实 RGB
重新编译所得 suffix 预测代价已为 selected 0.601630、greedy 0.157256，最后 qpos 也更差。
该现场保留为 predicted-successor 与 observed-successor 之间的差异，不能隐去、靠改阈值
去掉，或未经新审计便宣布是理论失效。

实际计算核账（每根，含同一后缀；VJP 为逻辑行数，不是 GPU kernel 次数）：

| 方法 | predict 调用 | predict 样本 | VJP 行数 | mean planner wall seconds |
|---|---:|---:|---:|---:|
| dispatcher | 1935 | 2062 | 24768 | 16.3647 |
| same-bank greedy | 15 | 142 | 192 | 0.1363 |
| CEM-lookahead control | 29 | 1934 | 192 | 0.1933 |
| zero-prefix control | 14 | 14 | 192 | 0.1292 |

前向样本接近绝不代表等资源；dispatcher 的逐候选 Jacobian 明显昂贵，当前不能声称
计算效率或等预算优势。新结果独立证明回退策略缺陷及其修复，同时保留 RGB 目标几何、
successor fidelity、强基线资源匹配和独立 calibration/outcome 为未完成问题。

最新集中回归：125 项逻辑测试通过，`git diff --check` 通过。代码审计与实验技能的流程
要求促使本轮保留旧失败协议、另命名新 runtime，并把逻辑测试、开发重放和新 CUDA 证据
分开记账。没有训练 readout/adapter，没有 CPU 神经实验，也没有 Git commit/push。

## 继续溯源：历史 innovation 与动作限制任务

### 128 分支同根重放及过去观测校正

主机 CUDA 审计已完成全部 `202626000..202626031`，每根重放 dispatch、greedy、zero、
CEM 的固定首段，共 128 个分支，不重新选择动作：
`reacher_d2p_causal_innovation_fixed_branches32_20261025_cuda.json`。
全部实际 successor RGB、动作哈希、observed goal cost 与原回执精确一致；raw action
归一化往返最大误差为 0，singleton 模型分数相对原 batch 分数最大误差仅
`4.76837158203125e-7`，在推理前固定的 `2e-6` 范围内，没有改变两处排序反转。

同时检验一个明确不同于 raw frozen predictor 的、参数不训练的临时反馈校正：

`innovation = encode(observed_root) - predict(predecessor_RGB, executed_history)`。

完整 predecessor 包含尚未 root-shift 的三个 RGB 与三个已执行 action packets；校正量
在任何候选 successor 执行前计算，系数事先固定为 1，所有候选共用。没有未来标签、
系数拟合或 readout/adapter 训练。候选预测加同一 innovation，仅用于诊断，不修改主线。

| 固定分支诊断 | raw predictor | past-only unit innovation |
|---|---:|---:|
| 全部 128 分支的 mean latent L2 error | 1.155917 | 1.094625 |
| 全 32 根 candidate-vs-zero 严格排序一致 | 30/32 | 30/32 |
| 原 7 个 eligible 根严格排序一致 | 5/7 | 5/7 |

82/128 分支的 latent L2 变小，但不能以平均误差下降代替决策改善。该规则修正了
`202626027` 的反转，同时将原本正确的 `202626010` 错排；`202626021` 仍然错误：
raw margin +0.126867、校正 +0.023358、实际 -0.019503。因此不采用它开启 P1，也不再
根据这些已知 outcome 调大系数或选择校正窗口。

审计代码另发现 `pixels.shape[0] != history.shape[0] != 1` 的 Python 链式比较未真正
保证单根输入。已改为分别检查两个 batch 必须为 1，并覆盖 `(2,2)/(1,2)/(2,1)` 三种
非法输入。**此前实验实际都使用 batch=1，该校验漏洞不是这两个反转的已证实原因。**
这项修复不改变原始权重、selector 或执行代码。

### 固定的动作限制任务补查研究前提

普通全驱动 Reacher 的 qpos 目标不自动具备“当前难以直接控制、通过重配置后可达”前提。
新 7 个 witness 中相对初始 qpos 真正不进展的只有 3 个；这与对 zero/greedy 非贪心是
三个不同判据，不能混为一种机制证明。

因此按原提案的“一模型、多目标/horizon/动作限制”，事前固定三种任务：
`full_drive=(1,1)`、`weak_shoulder=(0.05,1)`、`weak_elbow=(1,0.05)`，数值为未来 raw
torque 的绝对上界。它们不是完全锁死执行器，也不能仅凭动作限制就自称已构造成功的
controllability-gap family。

新根 `202627000..202627007` 来自 8 个不同 episode，与此前 104 个 episode 不重叠。
三条件共用 root/goal/已执行 history 及 source-derived 参数坐标；限制只从 planning root
开始生效，所有候选、Jacobian、评分、执行和 suffix 使用同一受限 chart。经缩放后的实际
packets **不再声称是原始官方训练 packet**；未训练或改写模型。所有 24 个 root-task
单元预声明保留，独立有限差分及实际 raw action bounds 均需核验。

运行程序为 `audit_reacher_d2p_actuator_limit_tasks.py`，同 stem 的协议与完整 JSON
`reacher_d2p_actuator_limit_tasks_pilot8_20261025_cuda.json` 均已落盘；24 个单元全部完成，
执行、动作范围与预声明 FD gate 全部通过，最大 per-cell median FD error 约 0.00446。

| task | witness / 全根 | 对 greedy 终态距离均差 | 胜 / 负 / 平 | eligible RGB 对 zero 非贪心 | eligible qpos 对初始根不进展 |
|---|---:|---:|---:|---:|---:|
| full drive | 0/8 | 0 | 0 / 0 / 8 | 不适用 | 不适用 |
| weak shoulder | 2/8 | -0.024062 | 2 / 0 / 6 | 2/2 | 1/2 |
| weak elbow | 2/8 | -0.001586 | 1 / 1 / 6 | 2/2 | 1/2 |

shoulder 描述性 CI 为 [-0.055054, 0]；elbow 为 [-0.022720, +0.017962]。不能从两个
witness 或接触零的 bootstrap 区间推导正式 P1 成功。zero-prefix 平均 Jacobian norm
依次为 27.22013/20.58838/17.43825，但正交不可达残差比例几乎不变（约 0.366）。
列空间投影丢弃奇异值幅度，在可逆输入缩放下基本不变；因此该比例不能独自衡量有界
力矩下的真实控制能力。当前 selector 的 bounded suffix score 与该诊断不是同一量。

### 分段执行精确重放与后缀置零干预

主机 CUDA 回执 `reacher_d2p_actuator_suffix_ablation24_v2_20261025_cuda.json` 已完成。
24 个 root/task × dispatch/greedy/zero 共 72 条存储的完整 15-step raw action witness，
连续重放所得 final-qpos 距离与原“5 步—观测—恢复—10 步”执行**全部精确相等**。
这批负结果没有被验证为分段状态恢复误差；结论只覆盖这些轨迹。

成对干预固定前 5 步，将后 10 步替换为 literal raw zero。定义 suffix improvement 为
被动后缀距离减去优化后缀距离，正数表示控制后缀有用：

| eligible 单元 | dispatch 后缀改善 | greedy 后缀改善 | 被动 dispatch−greedy | 优化后 dispatch−greedy |
|---|---:|---:|---:|---:|
| shoulder / 202627000 | +0.294312 | -0.001128 | +0.185590 | -0.109850 |
| shoulder / 202627005 | -0.404659 | 约 0 | -0.487304 | -0.082645 |
| elbow / 202627002 | +0.318713 | 约 0 | +0.258126 | -0.060587 |
| elbow / 202627004 | +0.146775 | 约 0 | +0.194675 | +0.047900 |

前、第三个案例需要后缀控制才能反超 greedy。第二个案例的终态获胜主要来自被动运动，
后缀对物理目标反而有害，不能作为改善可控性的正例。第四个虽后缀效果更大，仍不足以
反超 greedy。这是已知根的因果诊断，不是未见根泛化、可控性定理或 P1 门槛放行。

首次新审计在图像预处理时因 `env.render()` 的负 stride 视图中断；v2 使用连续数组，
保持像素值/朝向不变，并加专门回归测试。首次协议和现场不覆盖。该错误只阻止新脚本
产生结果，不能解释早先 P1 的排序反转。此任务族检查仍不替代资源匹配强基线、正式
P1 calibration/outcome 或组合规划效益。

### 排序反转的精确代数归因

主机 CUDA 回执 `reacher_d2p_margin_causal_decomposition32_20261025_cuda.json` 完成同一
32 根 × 4 分支重放；原始 observed RGB/action/cost 精确复现，权重未改，全部 gate 通过。
另存 predicted/observed/goal latent vectors，使分解可复核。令 `p,p0` 为候选和 zero
预测，`o,o0` 为真实编码，`g` 为 goal，`dp=p-p0`、`do=o-o0`，以 latent 维度平均：

`实际 margin − 预测 margin = 2 mean[(o0-p0) dp] + 2 mean[(o0-g)(do-dp)] + mean[do²-dp²]`。

三项分别是 zero 后继的共同偏移、动作响应线性误差、响应二次项误差。分解在 float64
算术中核验恒等式，不更改 float32 模型前向、部署分数或原选择。两个反转根为：

| root | 预测 margin | 实际 margin | zero 偏移贡献 | 响应线性贡献 | 响应二次贡献 | 动作响应余弦 |
|---|---:|---:|---:|---:|---:|---:|
| 202626021 | +0.126867 | -0.019503 | -0.122297 | +0.002344 | -0.026417 | 0.996303 |
| 202626027 | +0.006538 | -0.033888 | +0.008249 | -0.048011 | -0.000665 | 0.996421 |

021 的共同偏差很大，但即便使用**未来真实 zero-successor**进行 oracle anchoring，
其 margin 仍为 +0.004570，与实际负号不符；027 的 oracle margin 为 +0.014788，也仍错。
010 原本正确的排序甚至会被该 oracle anchor 翻错。因此“所有动作共加一个偏移即可修复”
不是这批数据支持的方案，更不能用未来 counterfactual 数据接入部署或放行 P1。

两个 root 的动作响应方向其实很接近真实，但高余弦不能保证与目标残差投影后的代价
排序正确。这具体定位了差异的来源；它既不是 D2P 理论反证，也不证明所有代码问题
已排除。当前仍没有证据将这两个反转归因为新增的执行路径、数值减法或动作缩放错误。

### 全 10 维真实响应与自动微分的分离检查

`reacher_d2p_physical_response32_20261025_cuda.json` 在主机 CUDA 完整执行同一 32 根。
以参数原点为中心，完整 10 维首段动作各做正负干预，尺度事前固定为 0.1、0.25，
合计 1,280 个 5-step 物理干预，另逐根精确重现原 zero successor。使用已验证的
`sealed_observation_exact_state` diagnostic root 合同：实际测量的原根与参考逐 hash
相等，恢复完整 integration/scene state 后复用该根观测；所有 successor 仍真实新鲜
渲染，不缓存、不用预测替代。该合同没有静默改变正式 P1。

| 固定参数尺度 | 自动微分 vs 同模型 FD 相对 Frobenius 差的跨根均值 | 自动微分 vs 真实 RGB FD 的相对差 | 同一实际目标残差下斜率方向一致 |
|---|---:|---:|---:|
| 0.1 | 0.004032 | 0.184801 | 291/320 |
| 0.25 | 0.025049 | 0.131168 | 305/320 |

这些有限尺度点包含 `tanh` 与模型非线性曲率，真实 RGB 还包含 rasterization/量化，
不能将其当作无噪声的无穷小物理 Jacobian，也不根据较有利的尺度重新选 gate。
同模型差远小于物理响应差，未发现支持“自动微分实现错误造成当前反转”的证据。
小奇异值也不能直接解释为真实高阶控制方向，可能受到上述有限尺度和渲染影响。
没有把测得的物理 Jacobian、qpos 或 oracle successor 送入 planner。

本轮相关逻辑回归共 244 项通过，`git diff --check` 通过。完整原提案缺口清单见
[D2P 研究闭环审计](D2P_RESEARCH_CLOSURE_AUDIT_20261025.md)。

## 新增：官方 RGB / runtime 渲染的交叉干预

`reacher_d2p_source_render_crossover32_20261025_cuda.json` 已在主机 CUDA 完成。
固定上述全部 32 根，不筛掉两个反转根；使用相同 source episode 内的下一段官方
五动作，交叉替换模型输入 RGB 和实际 successor RGB 的来源，目标保持相同。

| 输入 / successor | 平均预测 latent L2 | 平均绝对 goal-cost 误差 |
|---|---:|---:|
| source / source | 1.112035 | 0.038771 |
| source / runtime | 1.108762 | 0.038313 |
| runtime / source | 1.112969 | 0.038974 |
| runtime / runtime | 1.108779 | 0.038516 |

输入变化引起的平均预测位移为 0.032216，实际输出 RGB 编码位移为 0.048859；物理
qpos/qvel 最大误差为 1.51e-8 / 1.16e-7，均在事前合同内。当前官方动作下，不支持
将剩余预测误差主要归于渲染迁移。但该审计没有替代 counterfactual prefix 排序实验，
也不构成理论负结果。沿用已核验的 source SHA 回执，未声称重新 hash 完整数据集。

## 新增：CEM 输出保留与搜索—执行后缀合同

### 历史最优候选保留不是单独的物理修复

新增 `reacher_d2p_cem_incumbent.py`，保留整个搜索过程中最佳候选；不改变 RNG、
proposal、elite 更新，不向候选注入动作，不额外调用模型，不用物理标签。
`reacher_d2p_cem_incumbent32_20261025_cuda.json` 完整重放 32 根的旧搜索 trace、
动作、observed costs、执行身份和真实终态，全部精确一致。20 根模型目标严格改善，
20 根首段改变，但实际终态对旧选择仅 10 胜 / 10 负 / 12 平，均差 +0.048462，
描述性 95% CI [-0.044151,+0.153798]。不能称其已经修复控制或 P1。

### 已确认的具体协议问题：搜索三段、执行时丢弃两段

原 `lookahead_cem` 实际优化 3 个 action packet，却仅执行第一个 packet，随后统一
换成另一种单次 DLS 后缀。这是合法的“首段选择 + 相同 controller”诊断，不是完整
CEM-MPC 基线。更好的三段候选不保证其首段搭配一个不同 controller 后仍更好。

`reacher_d2p_cem_execution_contract32_20261025_cuda.json` 固定全部 32 个已知根、
原搜索轨迹与原首段，直接执行保存的 CEM tail，或从相同真实 successor RGB 重新
用 CEM 搜索剩余两段。另对 dispatcher/同库 greedy 的固定首段做同样后缀对照。
完整性 gate 全通过；所有旧 DLS 终态、首段和 successor RGB 精确重放，保存 tail 的
分段/整段执行终态也精确相同。未训练或修改原 checkpoint。

| 固定首段策略 | 原单次 DLS 后缀：平均真实终态距离 | 保存 CEM tail | 真实 RGB 重搜 CEM |
|---|---:|---:|---:|
| CEM 最后一代输出 | 0.511204 | 0.053504 | 0.035000 |
| CEM 历史最优输出 | 0.559666 | 0.044471 | 0.032561 |
| D2P dispatcher | 0.257405 | 不适用 | 0.033826 |
| 同库 greedy | 0.270575 | 不适用 | 0.035803 |

两种 CEM 的保存 tail 均在 32/32 根优于原 DLS；相对原 dispatcher 均为 30 胜 / 2 负。
Dispatcher 的固定首段改为 observed-CEM 后缀为 31 胜 / 1 负，配对均差 -0.223579，
描述性 CI [-0.270093,-0.177739]。因此旧“dispatcher 胜过 CEM”的说法只能指这个
已受限的共享后缀对照，不能继续支持完整规划器性能主张。这是有干预证据的协议缺陷，
不是冻结模型不具备有效控制动力学的证明。

保存 tail **没有增加搜索**；observed-CEM 每根每分支额外耗费 1280 个 predict 样本、
0 VJP 行（128 proposals × 5 generations × 2 packets），必须单列资源。原共享 DLS
为 14 个 predict 样本和 192 VJP 行，不能称两者等预算。编码、真实评估和重放也不应
混入或隐藏为在线搜索。该对照使用已知根，只能作因果诊断，不能替代新根确认。

后缀替换没有改变首段；两个首段 RGB 排序反转仍是独立待解问题。P1 的非贪心及
重配置收益合同未因此通过，后续 P 阶段不自动打开。新的迭代 SVD 后缀将单独审计，
不静默更改原 selector/执行合同，也不使用物理 outcome 选择迭代数。

## 新增：单次 SVD 后缀的求解不足

`reacher_d2p_iterated_suffix32_20261025_cuda.json` 已完成同根 CUDA 对照。
新增独立的 `reacher_d2p_iterated_suffix.py`，每次在已接受动作参数处重新计算 SVD，
仍用原有 singleton 非线性模型代价接受规则。保留全部事前指定的 1/2/4/8 迭代结果，
不用实际终态选择迭代数。原 P1 selector/runner 保持不动，因此这不是静默更改 P1。

| 固定首段 | 1 次 SVD-DLS | 2 次 | 4 次 | 8 次 | observed-CEM 后缀 |
|---|---:|---:|---:|---:|---:|
| dispatcher | 0.257405 | 0.183632 | 0.133556 | 0.101150 | 0.033826 |
| 同库 greedy | 0.270575 | 0.198146 | 0.146228 | 0.101728 | 0.035803 |

表中是所有 32 根的平均真实终态距离。首次迭代的完整动作、终态和 successor RGB
均精确重现旧执行；各 checkpoint 的累计 predict 样本数与实际计数一致，VJP 另列。
Dispatcher 8 次相对原单次为 29 胜 / 1 负 / 2 平，均差 -0.156255，描述性 CI
[-0.204233,-0.110628]；相对同首段 observed-CEM 仍均差 +0.067324，不宣称已赶上。
普通 greedy 也同样大幅改善，故这首先是后缀优化能力，不是重配置独有收益。

原网格最小非零步长为 1/16；8 次版本有 5 个 dispatcher 分支在该网格无下降后提前停。
这不能被写作局部无可达进展或模型动力学不存在。仅在 coarse-stop 时加入 1/32 到
1/65536 步长的模型内 rescue 已独立完成；此变体独立封存，不改变前一回执。原始
solver / auditor 源文件已归档为 results 中相同 artifact 前缀的 `*_source.py`。

`reacher_d2p_fine_suffix32_20261025_cuda.json` 的全部完整性 gate 通过。5/5 个
dispatcher coarse-stop 分支仍有模型下降方向，fine rescue 均找到它；5 根真实终态
均改善，其他 27 根完整动作保持精确不变。这直接证实 coarse-grid 停止不能被解释为
局部不存在改进动作。8 次 fine-rescue 均距为 0.083214，对 coarse-8 均差 -0.017936，
描述性 CI [-0.043019,-0.000808]。同库 greedy 的 6 个停步分支均找到模型下降，但
物理结果为 5 改善 / 1 恶化，全根均距为 0.080737；不混写模型下降与真实单调改善。

Fine rescue 没有改变首段、非贪心阈值、目标或 checkpoint；未训练额外模块。它仍比
dispatcher 同首段 observed-CEM 均距高 0.049388，因此不能称求解问题已全部解决。
全部仍是开发根诊断，非 P1 开门。合并代码逻辑回归 286 项通过。随后已封存 8 个新根
`202628000..202628007`、8 个 source episode，与此前 112 个 episode 全部隔离，
在查看新 outcome 前固定 single/coarse-8/fine-8 三版本，并加入真正执行自身
observed-RGB CEM 后缀的 shrinking-horizon CEM-MPC（每阶段 128×3，合计 1920
predict 样本、0 VJP）。该确认仍非完整资源匹配或独立团队复现。

### 新 episode 确认已完整完成

`reacher_d2p_suffix_repair_fresh8_20261025_cuda.json` 在主机 CUDA 完整运行，8 根和全部
7 种方法保留，事前元数据 admission、完整 93 GB source SHA、所有执行/计数 gate
和冻结权重检查通过。模型不训练、不拟合 readout，不根据新物理 outcome 选版本。

| 新 8 根全根平均真实终态距离 | 原单次后缀 | coarse-8 | fine-rescue-8 |
|---|---:|---:|---:|
| dispatcher 首段 | 0.226112 | 0.118124 | 0.057419 |
| 同库 greedy 首段 | 0.254305 | 0.114660 | 0.053955 |

真正 CEM-MPC 为 0.032064。Fine-8 对原单次在两类首段均为 8 胜 / 0 负；dispatcher
配对均差 -0.168693，描述性 CI [-0.273343,-0.087561]。新根 202628000 的普通迭代
仍停在 0.545527，fine rescue 降至 0.059887，给过早停步修复提供了新根因果证据。
但 fine-8 相对 CEM-MPC 仍为 3 胜 / 5 负，均差 +0.025354，描述性 CI
[-0.006874,+0.076176]。不能因差异不显著就声称等效、胜过 CEM 或正式等预算。

重配置 witness 仅 1 个（202628002）；其 actual RGB 首段相对 zero 为非贪心，但
qpos 对 zero 反而略有进展，两种坐标不能混写。该根原单次后缀下 dispatcher 为
0.083226、greedy 为 0.308774，看似有利；充分求解后为 0.045109 对 0.017402，
重配置优势消失。因此当前证据支持“后缀数值优化不足与受限基线造成部分旧负结果/
旧优势误读”，不支持“已经证明了新的非贪心重配置能力”，也不支持理论失败。

新根平均在线部署计算（不把诊断重复执行当作在线开销）：dispatcher fine-8 为
2166 个 predict 样本、26112 逻辑 VJP 行；greedy fine-8 为 246 / 1536；CEM-MPC
为 1920 / 0。代价和首段选择都不完整匹配；greedy 同样获益且计算更少，不能把
后缀改进全部记为重配置或 Planning Compiler 的独有收益。

本批所有 CUDA 作业均完成，289 项相关代码逻辑回归通过，`git diff --check` 通过。
首段排序反转与原 P1 正式合同仍未解决；未放宽门槛、未打开 P2/P3/P5，也未据此
标记整套提案闭环。

## 新增：选择器与后缀求解器一致性的因果审计

`reacher_d2p_refined_selector40_20261025_cuda.json` 已完成：保留前 32 根及新 8 根的
全部旧 dispatch/greedy/zero 分支，先从**想象 successor** 计算 fine-8 代价并冻结
选择，再执行各分支真实 RGB 后缀。所有旧单次模型分数、原首段、实际首段 RGB、
fine-8 完整动作及真实终态精确重放，完整 93 GB source SHA 重新核验。原模型未变。

8 个旧 witness 中，模型内精化后 4 个不再优于 zero/greedy，且这 4 根实际也都比
greedy 差（根 202626005/010/026、202628002）。保留者为 202626011/021/022/027。
仅按模型分数重检，不使用实际距离或 RGB successor 参与选择；非贪心 margin 保持
1e-6。拒绝者不是删除根，而是在同一根执行原有 greedy fallback。

全 40 根重检策略相对原 dispatch+fine-8 为 4 胜 / 0 负 / 36 平，终态均差
-0.016085，描述性 CI [-0.037558,-0.000693]；相对 greedy+fine-8 为 3 胜 / 1 负 /
36 平、均差 -0.013410，CI [-0.040001,+0.001668]，仍不足以证明重配置增量收益。
两个旧首段排序反转 202626021/027 仍被保留，因此这项修复没有解决首段可靠性门。

这是**固定三个旧分支**的因果审计，不是完整 bank 搜索或新 efficacy。它支持的结论
是：旧选择器的部分优势来自单次局部后缀的数值求解差异，不能直接解释为可控能力改变。
为避免只重筛旧 witness，已新增完整 128-candidate 的 `reacher_d2p_refined_dispatch.py`：
所有满足原模型非贪心条件的候选和 zero/greedy 控制均使用同一 fine-8 求解器；执行时
在真实 successor 重新运行完全相同的 solver。没有按实际 outcome 预筛、没有跳过
不利候选、没有放宽原 margin。新的 8-root 运行与前 120 个 source episode 隔离，
协议、源码和根名单均在outcome前封存；该新运行现已完整完成，结果如下。

### 完整候选库的新鲜8根结果

`reacher_d2p_full_bank_refined_fresh8_20261025_cuda.json`：8个来源episode与此前120个
隔离，完整93GB数据SHA重新验证，原冻结权重哈希不变，所有实际调用核账、执行身份、
精确fallback及全根保留gate通过。6个模型witness首段在真实RGB中均对zero非贪心，
但wrapped-qpos只有3/6成立；不能把两种目标坐标混同。

| 全8根对照（dispatch减reference） | 终态均差 | 描述性95% root-bootstrap CI | 胜/负/平 |
|---|---:|---|---|
| 同库greedy+相同fine8 | -0.011168 | [-0.077904,+0.049450] | 3/3/2 |
| zero+相同fine8 | -0.303805 | [-0.573331,-0.104253] | 7/1/0 |
| 真正shrinking-horizon CEM-MPC | -0.012038 | [-0.239516,+0.148206] | 3/5/0 |

Dispatcher/greedy/CEM的平均终态距离为0.475862/0.487030/0.487899；平均predict样本与
VJP行为7923.75/71808、324/1536、1920/0，仍明显不等预算。因此一致求解修复不能
升级为重配置稳定收益、优于强基线、正式P1通过，亦不支持理论被否定。

### 后继物理控制几何：终点变好不自动等于控制能力改变

`reacher_d2p_post_prefix_geometry8_20261025_cuda.json`已完成全部8根×dispatch/greedy/
zero三分支。24条原完整动作见证的终态及首段qpos距离全部精确回放。每个5-step
首段后恢复精确integration/scene state，执行literal raw-zero的10步后缀，并以
±0.1、±0.25分别扰动全部20个raw torque坐标，用wrapped-qpos中央差分估计2×20
响应；重复动作见证只缓存相同干预，无根被删除。共22个不同首段、1760次带符号
后缀探针（另有zero和原优化轨迹回放）。模拟器执行与CUDA float64几何分析分开记账，
没有CPU神经模型或训练。

为避免把目标变近误判为响应增强，固定greedy/raw-zero-future的目标残差，对所有
分支使用同一残差，求解固定1000步的盒约束二次响应问题，阻尼0.01。6个witness中，
两个尺度下均只有2个的共同残差有界响应增益提高，3个最小奇异值提高。具体为：

| 根后缀 | 共同任务响应增益是否提高（两个尺度同号） | 被动终距差 | 优化终距差 |
|---|---|---:|---:|
| 001 | 是 | +0.552860 | -0.081478 |
| 002 | 否 | -0.035845 | -0.010749 |
| 003 | 否 | +0.511919 | -0.200719 |
| 004 | 否 | -0.167379 | +0.000424 |
| 006 | 否 | +0.460775 | +0.134243 |
| 007 | 是 | +0.236056 | +0.068933 |

001提供了真实首段非贪心、共同方向响应变强、后缀反超greedy三者同时出现的诊断例；
007响应提高却仍终态更差，003终态更好但该响应指标不提高。它们共同说明不同机制
不能用单一终点分数顶替。有限尺度局部响应既非全局可控性定理，也不能单独开放P1。
原source SHA回执在本诊断中复用，未声称又重算完整93GB哈希。qpos/Jacobian/探针
结果不作为规划输入。新auditor初次在写出协议前拒绝了旧float32反归一化1-ULP越界，
现与producer既有1e-6舍入合同一致、原动作不clip；该校验问题不是历史P1反转原因。

### 模型侧见证合同修正及同根因果重放（已完成）

独立命名`reacher_d2p_geometry_dispatch.py`仍完整精化原bank中的所有模型非贪心候选，
保留原fine8终点收益检查，但不再仅凭终点优势命名reconfiguration。对每个可能通过
终点检查的候选及两个对照，从冻结模型计算20坐标的后继action Jacobian，以同一
greedy-prefix/zero-suffix任务残差比较盒约束响应增益，要求严格优于两个对照。全部
候选覆盖、Jacobian额外调用/VJP以及原solver工作都计入预算；没有物理outcome输入。

新8根seed202631000..007与此前128个source episode隔离，协议和实现先封存，随后
已在主机CUDA完整完成。最初候选块202630000..007只按source元数据检查就因episode碰撞被拒绝，
未执行推理/真实outcome、未封存结果；下一固定块在同样前置检查通过后才启动。
这不是按结果重新抽根。产物为`reacher_d2p_common_geometry_fresh8_20261025_cuda.json`。
无候选通过共同响应资格，8根全部精确执行原greedy；均距0.174130，CEM为0.118045，
对CEM3胜/5负、均差+0.056086、CI[-0.005739,+0.154115]。平均predict/VJP为
7059.875/66768，greedy285/1536，CEM1920/0；无效率优势，不能将保守拒绝说成成功规划。

`reacher_d2p_geometry_qualification_causal8_20261025_cuda.json`已完成同根固定决定
重放：两种selector决定均已由原运行在outcome前封存，没有重新筛选。改变4个决定，
对仅看终点的策略3胜/1负/4平、均差-0.127437、描述性CI[-0.447880,+0.097605]。
202631005被拒绝的分支实际为0.092888，greedy却为0.447249，故不能说资格规则可靠。
全部旧动作/距离精确回放，原权重未变，完整source SHA再次核验。

### 新见证求解器的数值缺口与误差证书（已完成）

新1000步投影梯度的box-QP在真实latent上有0.0004--0.0017的投影残差，尚未收敛；
不能从“迭代次数已跑完”声称已正确评价响应能力。新增`bounded_response_qp.py`：
float64 active-set求解小型SPD盒约束二次问题；记录可行性、终止状态和由凸切平面给出的
目标duality-gap上界，候选的增益下界必须严格超过两个对照的增益上界才能放行。

同矩阵主机CUDA审计`reacher_d2p_response_qp_convergence8_20261025_cuda.json`已完成
全部8根/22个响应矩阵。原共同残差、zero终点和1000步旧增益逐项精确重现，排除换矩阵
或换坐标造成的比较差异。求解修复最多提高增益4.148158，最大目标误差证书上界仅
4.285461e-13，但8个选择决定完全不变，仍无witness。因而这是真实的新数值缺口，
却**不是这批零放行的原因**，更不是历史两个首段排序反转的原因。理论未被该审计否定。

### 固定三任务的前置机制检查（全部完成）

新增独立certified compiler及可重启的任务顺序入口：
`run_reacher_d2p_certified_task_matrix_cuda.sh`，固定使用已存在的full-drive、
weak-shoulder(0.05,1)、weak-elbow(1,0.05)，同一组8个新根×3条件，不按outcome挑任务。
新根202633000..007的8个source episode与前136个不重叠；202632000块在元数据检查
阶段因episode重复被拒绝，未推理或执行outcome。未来约束从planning root开始，
过去source history不改；弱驱动未来包为source参数坐标经受限action chart映射，
不冒充官方原始raw packet。每个任务重新执行FD、调用核账、精确状态回放、动作限制
和完整CEM-MPC对照，模型完全冻结。24个单元现已全部完成、所有完整性gate通过；
三任务identity-joined审计亦完成，不计作正式P1或等预算效益。
响应坐标明确为单位有界actuator command，避免把不同torque上限下的单位变化当成
动力学几何变化。后继物理探针也已支持对应的动作尺度，物理真值仍不进入选择。

| 固定任务 | witness | dispatch / greedy / CEM均距 | 对greedy胜/负/平 |
|---|---:|---|---|
| full drive | 0 | 0.015147 / 0.015147 / 0.031210 | 0/0/8 |
| weak shoulder | 1 | 0.535952 / 0.539206 / 0.534714 | 1/0/7 |
| weak elbow | 0 | 0.489182 / 0.489182 / 0.532249 | 0/0/8 |

full/weak-elbow相对CEM的差异来自greedy后缀，不能归于重配置。三任务计算不等：
dispatch平均predict/VJP分别为9404.375/83952、9820.75/83904、6841.875/66048，
CEM均为1920/0。弱肩对greedy描述性CI[-0.009761,0]仍触零，对CEM4胜4负、CI跨零。

三组后继物理几何也全部完成：每任务8根×3分支，完整动作/首段距离精确回放，
后10步全部20个torque坐标、双符号±0.1和±0.25，物理量只作评估。唯一模型witness
weak-shoulder/202633002首段距离0.966983，大于初始0.919239；优化终态
dispatch/greedy/zero/CEM为0.608673/0.634703/0.615497/0.616283。共同greedy残差下，
dispatch双尺度响应增益为0.042179/0.042181，greedy为0.039421/0.039408，zero却为
0.043765/0.043768：相对greedy有部分机制支持，相对zero未成立。不能将这一局部
指标当作一般可控性的必要条件，也不能凭单根打开P1。

证据：`reacher_d2p_certified_task_matrix_20261025_audit.json`及
`reacher_d2p_certified_task_{task}_geometry8_20261025_cuda.json`。没有仍在运行的上述作业。

### 阶段准入门的独立实现缺口（已修复，不解释历史负结果）

旧P0→P1标量准入使用`ci95.upper <= 0`，会把全平[0,0]或触零区间误判为通过，
且未结合完整性gate。新`d2p_stage_admission.py`要求严格负的有限有序双侧区间，
并要求六个P0实现gate全部为true；calibration-only不得准入。主运行器已接入，
新回执记录helper SHA。补丁在所有已封存CUDA作业完成后应用；历史产物不改写。
AST回归验证除准入与main外至少20个函数完全不变；广覆盖Reacher回归358项通过。
检查的四份历史P0正例原区间本就严格负、gate全通过，不由此漏洞造成。
这个漏洞可能造成误放行，不能解释两个旧首段反转202626021/027；它们仍未修复。

### 全根动作插值干预（2026-10-26，主机CUDA已完成）

`reacher_d2p_packet_interpolation32_20261026_cuda.json`保留原32根、四个固定分支，
从当前根的官方下一五步动作向原动作按alpha=0/.25/.5/.75/1插值，合计640次真实
五步执行。候选不重新选择，模型不更新，原始P0源码按旧SHA从不可变快照加载。
全部gate通过；alpha=1的原动作、successor RGB和observed cost精确复现，
原模型分数在原singleton/batch容差内重现。参数化路径最大latent逐坐标差
9.536743e-7，重新归一化路径7.152557e-7，raw动作桥最大5.960464e-8；
官方动作物理状态误差最大1.159646e-7。没有发现足以解释两处反转的动作桥或
参数化错误。复用原source SHA回执，未声称重新hash完整93GB。

| alpha | 原7个eligible根的严格排序一致/反转 | 全128分支平均latent L2误差 |
|---|---|---:|
| 0 | 0/0，全部同一动作、均为平局 | 1.108779 |
| .25 | 7/0 | 1.026803 |
| .5 | 5/2 | 1.011800 |
| .75 | 4/3 | 1.075745 |
| 1 | 5/2 | 1.155917 |

alpha=1仍为202626021/027反转；中间档位另出现其他根反转，误差并非简单单调。
这里比较的是两条都向同一官方动作插值的分支，alpha<1的参考已不是raw-zero。
而且.25档的7个候选有5个在模型中已成贪心动作；不能把这个档位选作新门槛、
宣称P1修好。官方下一动作仅用于干预诊断，不是部署时可用的动作oracle。

### RGB历史条件动作支持（2026-10-26，主机CUDA已完成）

`reacher_d2p_rgb_conditioned_packets32_v3_exact_control_20261026_cuda.json`进一步检验
动作边缘支持与当前RGB历史的联合支持是否有区别。固定4096个不同官方source
episode，每个取一个因果三帧历史及下一五步动作，排除此前全部144个已用episode。
只用冻结编码器的三帧有序特征进行检索，不训练readout，也不使用qpos、目标标签或
未来观测。检索与随机对照从同一索引取包，每库31个候选加同一旧neutral动作，
全部32根的候选在任何物理successor执行前封存；全2048个bank单元均保留，重复
动作只执行一次并显式复用同根测量。该检索另依赖官方档案，不是D2P贡献或新根效益。

| 全32根描述性指标 | 随机取包 | RGB历史检索 | 检索减随机的配对95%CI |
|---|---:|---:|---|
| 绝对goal-cost误差（小为好） | 0.037908 | 0.038460 | [-0.001132,+0.002252] |
| 相对neutral的margin误差（小为好） | 0.024919 | 0.024588 | [-0.003315,+0.001980] |
| 两两严格排序一致率（大为好） | 0.955078 | 0.956527 | [-0.005922,+0.009514] |

全部完整性gate通过，模型权重未变。三项描述性区间均跨零，不支持此检索规则稳定
修复排序；尤其不能凭微小均值差宣布P1修好。模型非贪心候选在根021/027中的实际
非贪心precision，检索为13/16与6/8，仍存在反转。没有根据这些outcome换索引或距离度量。

该新审计的第一次入口在执行物理候选前因汇总API名称错误主动停止；修正后v2在
根023被精确重放门拒绝，因为新脚本用字面raw零替换了旧回执的近零浮点动作。
两次现场/协议保留，不作为完整实验。v3严格使用旧neutral字节，并额外评估真正
raw零：32根只有023/030的RGB改变，goal-cost差为-6.437302e-6/+6.198883e-6。
原反转根021/027的后继图像和cost完全不变。故这一可复现的微小动作/渲染敏感性
不能解释原两处失败；修复的是新auditor的控制身份错误，不是假称修复历史P1。

### 阶段准入调用链的补漏（2026-10-26，已修复）

继续审计发现只修P0 producer尚不完整：P1 consumer原先只信`p1_unlocked`，
`all({}.values())`还会接受空完整性gate；旧true标志可绕过实际CI、缺失比较与
calibration-only约束。先用8个反例测试复现，再使consumer重新执行严格P0证据
检查，并在绑定中记录新helper SHA。同时修复P1→P2的`upper<=0`，要求所有预声明
对照CI严格负、完整机制/实现gate和最低eligible根数，缺失额外source-matched对照
不能放行。新回执记录helper SHA；未来完整候选runner亦封存这个依赖。

四份真实历史P0正例均通过重新校验，历史产物未改。AST回归验证P0、P1 selector与
runner除准入校验/main之外全部函数不变，因此这不是神经/控制算法修复。旧两个
反转仍未解决，P1未通过，P2/P3/P5不开放。最新广覆盖逻辑回归403项通过。

本轮两张全部根诊断图及PDF、LaTeX caption和输入SHA映射位于
`le-wm-main/results/reacher_d2p_packet_diagnostics_figures_20261026/`；已目视检查。
图1保留全部7个原eligible根的插值曲线，图2保留全部32根的三项配对差，区间注明
仅为均值的描述性不确定性。图表流程不执行神经模型，不是CPU替代实验。

两张已完成的all-root诊断图（PNG与vector PDF）在
`le-wm-main/results/reacher_d2p_runtime_geometry_figures_v2_20261025/`：Figure_1为逐根
终态对照，Figure_2为共同目标响应与终态收益的关系；两尺度与全部8根均保留。

### 自身残差与有限动作词的进一步排查（2026-10-26，均已完成）

新完整证据见[残差锚点与动作顺序实验](D2P_RESIDUAL_ANCHOR_AND_ACTION_WORDS_20261026.md)。
先保留全部三任务24单元，重算113个原模型响应，并按候选自身残差交换Jacobians。
原所有模型量与物理对照精确重放；新诊断未增加候选且错拒原唯一弱肩有益分支，
故没有替换现有策略。这说明不能仅凭一般代数反例把共同残差条件定为本批根因。

再实现原提案允许的有限动作顺序类型，使用冻结模型一包动作响应的前两个右奇异
方向，不依赖官方动作proposal库。预声明17个四包词，在8个新来源根上全部测量，
加平坦六包SVD/GN及真正六/二包CEM对照，共152条30步真实分支。第20步统一
重观察，逐五步包更新历史；模型六包与四加二分段输出一致，动作/FD/CUDA门通过。
6个模型非贪心witness仅3个保持实际RGB非贪心；对同库greedy的全根CI跨零，
对CEM为1胜/7负。128个非neutral词margin为92一致/36反转；64个固定顺序对
54一致/10反转。模型权重未动，全部根和动作保留，不据此调门、开放P1或否定理论。

## 复现

在 `le-wm-main/` 下、主机 CUDA 环境运行（output 必须是不存在的新路径）：

```bash
MUJOCO_GL=egl .venv/bin/python -m experiments.spectral_reachability.audit_reacher_d2p_source_native_rgb_rank \
  --policy /home/gbk/.stable_worldmodel/checkpoints/reacher/lewm \
  --policy results/reacher_d2p_joint_repair_v1_20261024_cuda/checkpoint \
  --adapter-artifact results/reacher_d2p_p1_distribution_repair_v1_20261024_cuda/reacher_training_action_stats.json \
  --output /absolute/new-rgb-rank-audit.json \
  --roots 8 --proposals 128 --seed 202621000 --device cuda
```

回归测试只验证代码合同，不是模型 efficacy。当前仍为 `pilot_only`，未形成完整研究闭环。

新调度策略的独立复现命令（同名输出存在时需使用全新的 output；不可覆盖旧回执）：

```bash
.venv/bin/python -u -m experiments.spectral_reachability.run_reacher_d2p_program_dispatch_diagnostic \
  --policy /home/gbk/.stable_worldmodel/checkpoints/reacher/lewm \
  --adapter-artifact results/reacher_d2p_p1_distribution_repair_v1_20261024_cuda/reacher_training_action_stats.json \
  --disjoint-reference results/reacher_d2p_source_native_backtracking_p1_damping001_diagnostic8_20261025_cuda.json \
  --disjoint-reference results/reacher_d2p_source_native_original_rgb_confirmation32_v7_sealed_observation_20261025_cuda.json \
  --disjoint-reference results/reacher_d2p_source_native_backtracking_p1_damping001_confirmation32_v2_exact_state_20261025_cuda.json \
  --output results/reacher_d2p_program_dispatch_fresh32_20261025_cuda.json \
  --seed 202626000 --roots 32

.venv/bin/python -m experiments.spectral_reachability.audit_reacher_d2p_program_dispatch_result \
  --report results/reacher_d2p_program_dispatch_fresh32_20261025_cuda.json \
  --output results/reacher_d2p_program_dispatch_fresh32_result_audit_20261025.json
```

首条命令必须由主机 CUDA 执行；第二条只审计已有标量/轨迹，不执行神经模型，
不属于 CPU 替代实验。

本轮新增因果审计的主机 CUDA 入口如下；`--output` 必须改为尚不存在的新路径。
以下给出实际已运行的证据名以便逐项核对，不要覆盖同名旧产物：

```bash
.venv/bin/python -u -m experiments.spectral_reachability.audit_reacher_d2p_cem_execution_contract \
  --reference results/reacher_d2p_cem_incumbent32_20261025_cuda.json \
  --original-runtime results/reacher_d2p_program_dispatch_fresh32_20261025_cuda.json \
  --policy /home/gbk/.stable_worldmodel/checkpoints/reacher/lewm \
  --adapter-artifact results/reacher_d2p_p1_distribution_repair_v1_20261024_cuda/reacher_training_action_stats.json \
  --output results/reacher_d2p_cem_execution_contract32_20261025_cuda.json

.venv/bin/python -u -m experiments.spectral_reachability.audit_reacher_d2p_iterated_suffix \
  --reference results/reacher_d2p_cem_incumbent32_20261025_cuda.json \
  --execution-reference results/reacher_d2p_cem_execution_contract32_20261025_cuda.json \
  --coarse-reference results/reacher_d2p_iterated_suffix32_20261025_cuda.json --fine-rescue \
  --policy /home/gbk/.stable_worldmodel/checkpoints/reacher/lewm \
  --adapter-artifact results/reacher_d2p_p1_distribution_repair_v1_20261024_cuda/reacher_training_action_stats.json \
  --output results/reacher_d2p_fine_suffix32_20261025_cuda.json

.venv/bin/python -u -m experiments.spectral_reachability.run_reacher_d2p_suffix_confirmation \
  --policy /home/gbk/.stable_worldmodel/checkpoints/reacher/lewm \
  --adapter-artifact results/reacher_d2p_p1_distribution_repair_v1_20261024_cuda/reacher_training_action_stats.json \
  --disjoint-reference results/reacher_d2p_program_dispatch_fresh32_20261025_cuda.json \
  --disjoint-reference results/reacher_d2p_actuator_limit_tasks_pilot8_20261025_cuda.json \
  --output results/reacher_d2p_suffix_repair_fresh8_20261025_cuda.json --seed 202628000 --roots 8
```

不传 `--fine-rescue` 和 `--coarse-reference` 即运行普通 1/2/4/8 重线性化对照。
确认脚本会封存 implementation 源码，并重新验证完整官方数据 SHA。
