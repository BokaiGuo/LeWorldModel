# Actual-mid safeguarded suffix v2 — MAIN 审读后独占 HOST 执行

只新增本 stem source/tests/innerdoc。所有 v1、MAIN 结果和共享文档不改。固定原2 known roots × off/on × joint80/conditional40，共8计划，prefix5/goal/Q/normalizer/原任务 success 不变。不是 fresh/P1/P2 成立，不自动放行物理；worker 不运行模型、CUDA、物理或网络。

## 精确根因，不扩大解释

冻结 `moving_history_observed_suffix_cuda_20261027_v1.evaluate` 260/265行在 zero 求 nominal/J，270行 zero-centered bounded ridge，276行后才评估旧 suffix，288行无条件替换。`pusht_full_packet_compiler.bounded_ridge` 28–57行确实解零中心有限球正则最小二乘，不是以旧点为基准的非线性防劣选择。v1 协议本来就是 unconditional，不能在旧结果上补称它已 safeguarded。

MAIN 独立 saved arithmetic authority：`results/main_suffix_linearization_audit_20261028_v1_MAIN01.json`，SHA `44e558fe97f43c694beae6519167c7c43c9d0f6181878536f56a6b2fa58072db`。MAIN 报告8/8局部正则目标相对 zero 改善、8/8非线性模型 Q 相对 zero 改善，但3/8相对 OLD incumbent 变差。狭义原因是**弱 zero 参照 + 无条件替换**；这不证明符号错误，也不证明曲率导致比 zero 更差。worker 没有复跑该分析，模型选择不读取其数值或任何物理 outcome。

## 输入与精确动作域

复用冻结 v1 `build` 的完整 RGB strict、原 planner strict v2、物理输入和原 goal/model 闭包，从原 winner candidate_id 唯一取保存的 full20 **原始 float64 logits**。原实际 RGB report SHA `b57fa03560696c3c0b75fe842ec8263d6ec3961341d6736a4f6e782d1f0e92ea`，strict SHA `3d3d020c113959aec25f4e310b8512b814f7a78976f2bd7d5e58e0e435027f4b`。不需要 v1 recompile 或其物理 outcome 作为 v2 编译输入。

唯一候选 chart：`np.tanh(np.asarray(logits,np.float64)).astype(np.float32)`。必须与原 prefix+incumbent rawfloat32 bits 完全相同。joint20 cap 仍为 `.75*sqrt(10)`，suffix cap 仍为 `sqrt(cap²-prefix_logits@prefix_logits)`。检查 full joint norm、suffix norm、finite、raw 严格在(-1,1)，全部不加 epsilon。即使只超一个 ulp，也保留域数据、prepare=`blocked_illegal_incumbent`，编译在开输出/加载模型前拒绝。绝不 `atanh(rounded raw)`、clip、扩大球、修复旧 logits 或改固定槽。原边界浮点判定导致 blocked 是工程/域边界，不是科学负结果。

全部8 branches 和 domain check 永远保留在 prepare；任一非法不跑子集。compiler branch whitelist 只有实际 mid latent history、实际 raw10/tokens、原 prefix/old suffix、原 logits、goal192、identity Q、normalizer/domain/身份。native state、reward/success、真实 terminal latent 和控制结果不进入 `evaluate/proposals/choose`。完整 provenance/control metrics 在外层保存，只在计划封存后用于配对报告。

## 固定有限 model-only safeguarded search

每臂恰好：

1. 从原 actual-mid 输入执行 literal incumbent raw5，记录真实模型 endpoint/Q。
2. 在**同一个 literal rawfloat32 incumbent**求192×10 raw Jacobian（vectorized AD），AD forward 必须逐值等于第1步；不是在 float32 tanh(u32) 的另一个 raw 点求导。
3. float64 CPU saved arithmetic 计算 `g=2*(Jraw*sech²(u_old)).T@(z_old-goal)/192`。这是连续 float64 tanh 的链式**proposal surrogate**，不是声称可微分 float32 量化 chart。原 Q 不变，无 ridge/Q/readout 拟合。
4. 固定半径内点 `v=-0.5*remaining_radius*g/||g||`，方向 `d=v-u_old`；仅当 `g@d<0` 采用，否则 d=0。这是从旧 incumbent 出发的一步有界下降提案，不是从 zero 求 ridge。`g=0` 同样 d=0。固定 alpha `[1,1/2,1/4,1/8]`，全部生成/封存后再评估，逐个检查实际 logits/fulljoint/raw 合法性，不做 inverse 或边界修复。
5. 固定5候选为 incumbent index0 +4个步长，不提前停止、不去重。全部在相同 actual-mid/context/goal 上真实预测；`stable argmin(cost,index)`，相等保旧。复用 MAIN 冻结 `main_suffix_incumbent_guard_20261028_v1.choose`，source SHA `1a33cd589efe44acf71a316ceef5670f3ea0b551f4a154c8b589414ca9b2fbc0`；按候选index依次与当前best比较，等价全5 stable argmin，并保存4次guard比较cost/选择/SHA。只传 contextSHA/goal192/commands/predicted192/domain_verified/prediction_verified 精确字段，任何物理标签不传入。调用方完成实际输入/域/模块输出验证，guard不认证provenance。候选非线性变差会保留在全表，不删除。
6. 封存选中 index 后独立预测 replay 同一 raw，必须 context/Q 完全相同，且 **`selected_Q <= incumbent_Q`**，零 epsilon；失败不释放、不重试。

这是新的探索性 bounded line-search 算法，不声称原策略已有此 safeguard，不按 MAIN 报告的3个劣化索引分支，也不使用物理标签。仅保证本次同模型/输入上的已评估成本不变差；不保证全局最优、真实 Q、任务效能、P1/P2 或安全。

## 计量、strict 与 native handoff

每臂7 predict/AE/pred_proj（incumbent、AD forward、4 candidate、replay），1 Jacobian，0 SVD；AD API jacobian1/grad1，192 mathematical VJP。全8为56 predict、8 Jacobian、1536 VJP，新 visual encoder/projector=0。predict API 是 inclusive 另一轴，不能与内部模块重复相加。原编译1328predict、actual RGB24encode/8predict、旧物理160public/1632physics等继承另列；总 FLOPs/上游未知费用 null，不称等FLOPs。候选或 pending 故障不拿零成本替代。

与 v1 一样用真实 Modules 输入/输出 NPY 和预测/AD/math attempt→input-save→return durable ledger。新 Modules/PredictAPI 只扩展为固定7 stages，没有猴补旧 globals。strict 独立重算 proposal、所有5 Q、stable选择、合法性与 replay，并逐 stage 校验 full3 实际 latent/actions→AE→predictor→pred_proj→endpoint；所有未知 event 拒绝、AD 嵌套返回次序保留，missing/partial 不可标 completed。

计划保存 parent_*，删除泛称旧 `predicted_Q/witness`；新 `new_suffix_Q/context/witness` 绑定所选真实预测、完整 row/report digest，附 safeguard incumbent_Q/selected_Q/index/domain。无论保旧还是换新，部署 digest 都是本次模型选择见证，不能冒作旧选择器身份。whole10 是精确旧 prefix5+所选 suffix5。

复用原 `physical.execute`/native episode、warmup10、task goal bind、full10动作计量、off/on渲染及 `physical.assemble`。新 ReplayGateStore 直接接实际 native emit：step5同condition PNG bytes/pixel SHA/state/snapshot 必须与已编码 mid 精确一致，才可进入step6；不设状态、不近似恢复、不新增一次观测制造 gate。原success包含agent+object，与object-only coverage分栏，warmup旧goal reward不能混当前goal。

物理 strict 重新 consume 全编译和所有 native 原数组/PNG/raw/latest_action/goal/终止/计量，保存全部8 control/intervention 指标及有符号 delta；early terminal/missing按原native合同保留，不补成full10。编译和物理各900s（入口验证前计时，非抢占watchdog）、64MiB输出/final2MiB/磁盘floor200MiB。单臂 Sink另有原8MiB cap。验证阶段若先失败不加载CUDA；实际调用失败保8 rows和已用账，无自动重试。

## MAIN 命令

cwd `/home/gbk/code/LeWorldModel/le-wm-main`。所有输出必须是不存在的新路径。`<PREP_SHA>/<COMPILE_SHA>/<PHYS_SHA>` 只填实际产出，不预填未来SHA。

```bash
.venv/bin/python -m pytest -q tests/test_agent_p2_contextual_midfeedback_recompile_20261028_v2.py

.venv/bin/python -m experiments.spectral_reachability.agent_p2_contextual_midfeedback_recompile_20261028_v2 prepare --rgb-output results/agent_p2_contextual_physical_rgb_encode_20261028_v1_cuda_MAIN01 --rgb-receipt results/agent_p2_contextual_physical_rgb_encode_20261028_v1_consume_MAIN01.json --output results/agent_p2_contextual_midfeedback_recompile_20261028_v2_prepare_MAIN01.json

.venv/bin/python -m experiments.spectral_reachability.agent_p2_contextual_midfeedback_recompile_20261028_v2 verify --prepared results/agent_p2_contextual_midfeedback_recompile_20261028_v2_prepare_MAIN01.json --prepare-sha <PREP_SHA>

# MAIN 单独授权 HOST CUDA，worker 不执行：
.venv/bin/python -m experiments.spectral_reachability.agent_p2_contextual_midfeedback_recompile_20261028_v2 compile --prepared results/agent_p2_contextual_midfeedback_recompile_20261028_v2_prepare_MAIN01.json --prepare-sha <PREP_SHA> --output results/agent_p2_contextual_midfeedback_recompile_20261028_v2_cuda_MAIN01 --authorize-host-cuda

.venv/bin/python -m experiments.spectral_reachability.agent_p2_contextual_midfeedback_recompile_20261028_v2 consume_compile --compile-output results/agent_p2_contextual_midfeedback_recompile_20261028_v2_cuda_MAIN01 --compile-sha <COMPILE_SHA> --receipt results/agent_p2_contextual_midfeedback_recompile_20261028_v2_compile_consume_MAIN01.json

# MAIN 审读完整 strict 后另授权真实物理：
.venv/bin/python -m experiments.spectral_reachability.agent_p2_contextual_midfeedback_recompile_20261028_v2 run_physical --compile-output results/agent_p2_contextual_midfeedback_recompile_20261028_v2_cuda_MAIN01 --compile-sha <COMPILE_SHA> --output results/agent_p2_contextual_midfeedback_recompile_20261028_v2_physical_MAIN01 --authorize-physics

.venv/bin/python -m experiments.spectral_reachability.agent_p2_contextual_midfeedback_recompile_20261028_v2 consume_physical --output results/agent_p2_contextual_midfeedback_recompile_20261028_v2_physical_MAIN01 --report-sha <PHYS_SHA> --receipt results/agent_p2_contextual_midfeedback_recompile_20261028_v2_physical_consume_MAIN01.json
```

CLI只返回小路径/SHA，完整计划/指标写独占receipt。终态 RGB 重新编码需要另由 MAIN 授权；现 terminal_rgb v1绑定v1 compiler，不可直接冒用为v2 strict authority。

## 实际验证与未决

首轮新 RAM tests：38 passed in11.90s；后加严格suffix bits检查的联合回归出现3 failed/100passed，根因是误复用只接受full10的bits helper，已新增严格5×2 helper并补signed-zero/拒绝隐式rounding测试，保留失败说明。随后新增MAIN guard接线与见证变异测试。没有真实 model/torch/物理执行，使用 NumPy native-interface doubles。覆盖真实7 stage 模块/事件 ABI、全量 buffers/费用/AD、strict变异、旧点更优回退、zero ties保旧、精确chart差异、非法/一ulp域拒绝、new witness/parent分离和原native runner实际 emit schema接线。RAM不是科学证据。

新增 MAIN guard 后联合回归117 passed in21.20s。另已实际 saved-only 域 smoke：读取固定旧 prepare（SHA `a415fafcf82f9f7f93aa8a9a1c68d3f205b6b4c27e5763a2f4b346947580516e`）以及原4个slot result中的8个唯一winner logits；全8 incumbent 在原chart、fulljoint和精确remaining radius下均合法，无反解/修复。该只读接缝已加入单测（缺真实资产时明确skip而非造fixture）。这不是新v2完整prepare验证。还新增直接测试 `run_physical→physical.execute` 的所选计划/callback/binding/AUTH 接线，未运行真实物理。

最终实际回归：上述新v2 tests + 冻结recompile v1 tests + terminal_rgb v1 tests + MAIN guard tests，**119 passed in24.54s**（其中新v2 43项，saved已存在资产接缝实际执行、未skip）。模块 `--help` 退出0，六个CLI均已接入；旧terminal/v1/MAIN guard source SHA再次核验不变。三件套至此冻结，SHA由交付消息给出。

未执行生产 prepare/verify/CUDA/physical；MAIN须重建全闭包并确认全8域检查，通过后才能运行。没有fresh认证或独立P1/P2科学放行；真实效能/终态latent仍未知。
