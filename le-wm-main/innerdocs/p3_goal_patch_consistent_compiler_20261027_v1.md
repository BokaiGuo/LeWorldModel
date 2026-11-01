# Q 一致 conditional full10 / fixed-H2 compiler

本模块是 **P1 机制的 H2 可运行前置组件**。保留最初约定的 p3 文件名；不是已完成的 P3 dynamic，也不是 P1/P3/P4 准入。仅新增源码、非神经测试及 prepare，真实 CUDA/physics 由 main 独占执行。既有四根仍是 calibration，独立新根数为零。

## 路线与源码审查

已完整审查 `p3_horizon_residual_budget_20261027_v4.py`、`pusht_full_packet_compiler_20261027.py` 和 `p3p4_goal_patch_metric_20261027_v1.py`。v4 的 `compile_cuda` 计算 joint H2/H3 Jacobian，仅用于 residual/budget signals；`search` 用固定随机 logit-ball pool，不调用谱 bank 或优化动作。metric 模块的 `select_common_bank` 仅重评分；`compile_action_response` 是数组桥，不独立生成非线性模型证据。两者均不能包装为本次 compiler。

本模块实际调用原 full-packet compiler 的 `spectral_bank`、`bounded_ridge`、`make_transition`；依据其 `compile_arm(full)` 的 conditional suffix 结构实现新的带 durable journal 编译流程。v4 仅复用输入认证、纯 IO 和 neural hook 恢复机制，**不调用其 search/pool/allocate/compile_cuda 或旧计划出口**。

已阅读外层 `docs/p3_goal_patch_consistent_compiler_spec_20261027_v1.md`。该 spec 是 jointH2/H3 root ridge + 随机尾部的四臂方案，不是本合同。本轮按 main 明确选择继续 conditional full10 固定 H2 两臂；不静默混入该 spec 的候选、共享 AD、预算或 terminal-only selection。P3 dynamic 全科学范围保留，须独立后续实现。

## 精确闭环

每根有 `old_Q`、`goal_patch_Q` 两个独立真实 compiler；按 root index 奇偶交替先后顺序。各臂均从原三帧 native latent `[3,192]`、两段 normalized action `[2,10]` 的相同 context、相同原始独立 goal0 出发。

1. fresh H1 zero nominal 和 `192×10` raw-logit Jacobian。原 `spectral_bank` 使用 `QJ` 与 `Q(goal0−nominal)`：zero、bounded ridge progress、前两条右奇异向量的双符号/双半径，共 10 prefix。继承原 sign fix 和 SVD 次序，重根子空间无跨平台规范唯一性主张；同 host/backend，保存所有向量与谱。
2. 每个 prefix 真正执行一个五步 packet，然后从该 predicted context 独立 fresh suffix zero nominal、fresh `192×10` conditional Jacobian。prefix 作为固定条件，与原 compile_arm 一致；不冒充 joint H2 全链 Jacobian。
3. 调用原 `bounded_ridge(QJ, Q(goal0−nominal))`，精确求解 `min ||Ku−r||² + .002||u||²`，约束 `||u||≤.75 sqrt(5)`。未对 Gram/rhs 除以 192 后保留同 damping；只把展示代价除以 192。这是真实 action-space 线性化 trust-ball 优化，不是迭代非线性 GD/MPC。
4. 实际执行优化 suffix，再从原 root 独立完整 H2 replay；要求整个 latent/action history context 逐字一致，不仅比较 endpoint 或 allclose。所有 10 条（含重复）均执行并保存。
5. 每候选 `stage=||Q(z_prefix−goal0)||²/192`；`reach=(||Q(goal0−nominal)−QJ u||²+.002||u||²)/192`。按原 compiler 的 `stage+reach`、再 bank index 选优。真实模型 nonlinear terminal Q cost 是同 Q 诊断，不事后换为 terminal argmin。
6. winner 再执行整段 H2 和独立两段 replay，均与候选完整 context 精确一致。无视觉反馈/真实状态插入，没有任何旧 endpoint/candidate 导入。

旧/新 Q **在各自 bank、suffix solve、stage、reach、terminal、selection 和验证中全程一致**。每 packet 十维 time-major logits 转 float32 后由 NumPy `tanh` 映为五个二维 raw controls；模型端由封存 `make_transition` 的 public action mean/std 正常归一化，不 clip、不增加缩放。此 raw 值同时传实际执行与独立 replay；不声称与旧 CUDA `tanh` 逐比特重演相同动作。每 packet 的 ball 分别约束，非 spec 的 joint-H ball。

## 固定预算：新旧 Q 均独立支付

| 项目 | 每臂 | 每根两臂 | 四根八臂 |
|---|---:|---:|---:|
| 根 nominal + ADforward | 2 | 4 | 16 |
| 10×(prefix + suffix nominal + ADforward + optimized suffix + H2 replay) | 60 | 120 | 480 |
| winner 整段及逐段 replay | 4 | 8 | 32 |
| predict attempts cap | **66** | **132** | **528** |
| action-encoder attempts cap（独立账） | **66** | **132** | **528** |
| fresh Jacobian materializations / AD API | 11 | 22 | 88 |
| 数学输出 VJP 方向 | 2112 | 4224 | 16896 |
| SVD（bank 两次 + 每 suffix 一次） | 12 | 24 | 96 |
| planner visual encoder | 0 | 0 | 0 |

完整成功必须恰好这些次数；失败/中断按 attempted/completed 原样留账，不退款、不补齐、不自动 retry。AD forward 已包含在 predict 里；VJP/backward 另列，实际 engine backward API 次数未知，不能拿数学方向冒充 engine 调用。predict 与 action encoder 各自 B=1、T=3，不称 FLOPs 或完整 pipeline 等预算。两臂不共享 raw J 或历史导数，即使动作/根 nominal 恰好相同也重新执行。

Qnew 构造继承已封存 80 encoder +80 existing-projector forwards、8 metric SVD；旧 context/goal 和模型训练等历史成本未知，不是零。本 prepare 不再运行这些计算。prepare/verify 只读哈希、数组和 projector 检查，不做模型/AD/physics；没有 SVD 调用。`selected_actions` 重算 8×12=96 次 saved-array SVD，明确是出口验证 CPU 数组成本，不计为新 CUDA 或 fresh Jacobian。

## 输入和封存

prepare 严格绑定 v4 原 prepare、goal-patch metric prepare/report 与独立 array-audit report；四根通过 `root_sha256` join，root/index/原 goal0/冻结 model bindings 必须一致。原 report 的 Q 数组原样使用，与 audit 的 `.old_Q.npy`/`.new_Q.npy` 哈希和数值核验，不重新估计或后验调 cutoff。

`source_input_sha256` 保留旧 packet 身份；`active_metric_ids` 绑定 root、原 goal0、实际 Q、model bindings、metric report/prepare/catalog/audit hash、arm；新的 `input_sha256` 绑定两臂源输入。metric report 中的旧 selection/candidate/outcome 字段不会进入新协议或选优。只读取已验证 Q，不拿旧模型/物理终点替代新预测。

源码 closure 包括新源码/测试/本文、所有静态可达 Python 实现（含 lazy runtime imports）及已安装模型包，并分批 source-seal。执行前/后复查；模型字节执行前核查，模型 tensor state 执行前后比较。缓存 protocol 中模型配置/normalizer 来源与实际执行保持同一 sealed closure。

每个 neural entry、AD、实际 helper SVD 都有先 intent 后 completion/error 的 fsync journal。SVD 对原 helper 的 NumPy 调用进行单线程临时 instrumentation，finally 恢复；禁止并发线程编译。prefix、suffix J/nominal、solve/actions、预测、replay context 随阶段写入，不等整根成功才留证。report 保留全部 8 slots、所有 duplicate 和失败，未开始臂标 blocked；首错与后续 source-drift 分开保留。

## main 执行接口

从 `le-wm-main` 使用 `.venv/bin/python -m experiments.spectral_reachability.p3_goal_patch_consistent_compiler_20261027_v1`：

```bash
.venv/bin/python -m experiments.spectral_reachability.p3_goal_patch_consistent_compiler_20261027_v1 --verify --prepared results/p3_goal_patch_consistent_compiler_20261027_v1_prepared --prepare-sha256 PREPARE_SHA
.venv/bin/python -m experiments.spectral_reachability.p3_goal_patch_consistent_compiler_20261027_v1 --execute-cuda --prepared results/p3_goal_patch_consistent_compiler_20261027_v1_prepared --prepare-sha256 PREPARE_SHA --output results/p3_goal_patch_consistent_compiler_20261027_v1_cuda_main_v1
```

`PREPARE_SHA` 在交付消息中给出，执行输出必须不存在。仅 main review 后允许 CUDA；这里没有运行上述 execute 命令。

下游接口 `selected_actions(cuda_output)` 返回八条新程序，每条包含 `root_index/root_sha256/arm/horizon=2/actions[10,2]`、原 root context/goal、active Q 与 metric ID、model bindings、source/active input hash、预测完整 context/Q cost、protocol/report/selection hash。出口重建所有 bank/conditional solves/动作/成本，校验 journal 和 528/88/96 账，不复用任何旧 exporter/schema。出口本身只认证保存证据内部一致，不能重演神经模型或证明物理收益。

## 验证与剩余缺口

非神经测试包括 Q 改变动作生成、原 bank 等价、ridge ball/KKT 既有测试、零响应重复保留、同 Q 全角色、完整 context mismatch、pre-dispatch 双 cap/失败不退款、源输入身份、保存数组篡改拒绝、假后端八臂执行/出口及失败 census、prepare 源漂移。mock 是代数/协议测试，不是 CUDA 预测证据。

仍需 main：review + 真实 CUDA 验证 88 个 Jacobian/80 个新候选及 exact replay；另独占新 physical consumer（8×10=80 command 上限，guard/partial 全保留，可另预算 8 terminal encodes）及实际运行，不能复用旧终点。P1 独立确认性根/任务、P3 dynamic/H3 matched frontier 仍未完成。不同 Q 定义不同代价，旧 Q 与新 Q scalar objective 不跨臂相减当效用；未来物理对照按同 root、同任务、固定 H2、同反馈合同保留全部 partial。没有训练、下载、后验阈值或任何新效能证据。
