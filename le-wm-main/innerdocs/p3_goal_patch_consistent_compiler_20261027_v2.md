# Conditional compiler v2：R1–R3 独立修复合同

日期：2026-10-27。本版本仅修失败留证和纯数组 history/layout 核验；不是算法升级或新的效果结果。v1 源码、测试、封存 prepare 和已完成结果均不修改。main 报告 v1 实际 CUDA 已成功、528/88/96 匹配且源码/模型 unchanged；该结果属于 v1，不视为 v2 的真实执行认证，也不自动关闭异常路径缺陷。

## 身份与不变量

独立模块/测试/本文 stem 为 `p3_goal_patch_consistent_compiler_20261027_v2`，独立 `.protocol`、`.report` schema。生产代码完整复制并局部修复，不导入 conditional v1 执行器，不猴补 v1 NAME、contract 或函数 globals。测试可以只读调用 v1 做代数等价对照。

保留同四 calibration roots、原 goal0、原三帧 192 维 native projected CLS、两段 normalized action history、同冻结 model 与 goal-only Q 来源。`source_inputs()` 与 v1 输入完全相同，旧 source input SHA、active metric IDs、新 input SHA 不变；版本/执行身份由新 schema、contract、source closure、prepare/protocol SHA 隔离。

算法仍为两臂 `{old_Q,goal_patch_Q}` 的 fixed-H2 conditional full10，不是四臂 P3 dynamic/H3：

1. 每臂 fresh 根 H1 nominal/J；原 `spectral_bank` 构造 zero、root ridge、两个 mode 的双符号/双半径，十候选，不去重。
2. 每 prefix 实际预测后，从其自己的完整 predicted context fresh nominal/J；原 `bounded_ridge(QJ,Q(goal0−nominal))` 求 suffix。
3. 每 packet 十维 time-major logits，float32 NumPy tanh 得五个二维 raw commands；不 clip、不额外 `/100`。model transition/replay 仍调用原 public mean/std normalization。
4. `.002` damping、per-packet cap `.75 sqrt(5)`、原半径、bank 次序、Q stage/own-reach/nonlinear terminal 成本不变。ridge 原优化问题不除 192，展示 score 除 192。
5. selection 仍是 `stage+own_reach_value`、stable bank index tie；不是 nonlinear terminal argmin。每候选完整 H2 独立 replay；winner whole/segmented exact full-context replay 不变。

## R2：每阶段立即保存，不等后续 AD/SVD

Backend 新接口显式拆为 `nominal(context)` 与 `jacobian(context,operation)`；前者只做原 nominal forward，后者只做原 AD-forward/API。未增加模型调用。

Journal 序列：

```text
root nominal 完成 -> nominal 数组 fsync
root AD 完成      -> derivative J 数组 fsync -> bank SVD
prefix 完成       -> prefix 完整 context fsync
suffix nominal    -> nominal 数组 fsync
suffix AD         -> derivative J 数组 fsync -> ridge SVD
solve/actions     -> candidate 阶段写盘 -> optimized suffix -> replay
```

每个 neural entry、AD/SVD 仍是先 intent 再 completion/error。`selected_actions` 不仅核验原 candidate/operation/replay receipts，还逐项核对新增 11 nominal、11 derivative、10 prefix 和 bank witness 与 compilation 数组一致。

J/nominal 成功后下一项计算失败不再丢掉已保存数组。数组尚未从设备成功取出、进程被杀、永久磁盘故障等情况不能声称已持久化；intent/计账与可保存 partials 分开记录，不凭空补结果。

## R1：首错、secondary 和 best-effort report

`FailureLog` 保存异常链中的首因、每项 secondary、phase 与 traceback；持有异常引用避免 id 重用误去重。completion journal 写入失败不覆盖活跃计算异常，而附加为 secondary；hook install/restore 显式跟踪各属性，并独立尝试恢复其余 hooks。

执行器在每次 journal write/flush/fsync **前**保存深拷贝 RAM mirror。失败后仍独立尝试 model-state hash、journal hash、journal read/一致性检查、source revalidation、各 root-record write 和 report write，不允许其中单项异常跳过剩余收尾。全部八个 row slots 保留，未开始标 blocked；已完成候选/臂不伪称未执行。predict/action_encoder attempted/completed 来自非退款 receipt，J/SVD 计数来自 pre-write RAM events，不依赖故障后还能完整读 journal。

失败时 root report 可包含 `recovery_events`，保存 partial arrays/intent；`durable_journal_complete` 单列，不能把 RAM mirror 冒充已封存磁盘日志。主 report 写失败时，仅尝试新的 `failure_report.json`，不覆盖残缺主 report；两者都失败则 `IncompleteCompilation.report` 仍携完整内存报告供 main 捕获。此机制是 best effort，不承诺永久 ENOSPC、SIGKILL 或机器崩溃后恢复内存。任何 failure/secondary 均使 report incomplete、export 拒绝；没有自动重试、padding/refund。

## R3：纯数组严格 history/layout

`validate_context` 严格检查 layout 字符串及 `context_length=3,latent_dim=192,action_dim=10` 的类型/值，并要求有限 `[3,192]` latent 与 `[2,10]` action history。

`normalized_history` 用独立 NumPy float32 表达式镜像封存 normalizer：

```text
raw.astype(float32).reshape(-1,10)
token = (raw_flat − tile(float32([−.0087,.0068]),5))
        / tile(float32([.2019,.2002]),5)
```

subtract/divide 分开，不做 `/100`，不 repeat-interleave，不允许 raw 越界或隐式 clipping。prepare/verify 以 AST 只读核对 public authority `experiments/run_pusht_counterfactual.py` 的统计量字面值；该文件及 normalizer/transition/replay 在 static source closure 中封存。无需导入 torch 或模拟器。

runtime prefix/suffix 以及 saved-array verifier 都用 `validate_advance` 重建应有 normalized token chronology，strict `array_equal`，不接受 allclose；H2 完整 history 必须正好是两段 commands 的 tokens，H1 prefix 保留原 history 最后一段再追加 token。非 winner history 污染、winner 和其重复 context 同步污染、layout 污染都拒绝。latent history 仍逐帧检查 root→prefix→terminal chronology，whole/segmented/replay 仍严格相等。

纯数组 float32 镜像不代表已实测 v2 的真实 CUDA 算子/AD；真实后端是否逐 bit 一致必须由 main 后续独立执行验证。

## 不变预算

| 项目 | 每臂 | 每根两臂 | 四根八臂 |
|---|---:|---:|---:|
| 根 nominal + ADforward | 2 | 4 | 16 |
| 十候选全部编译及独立 replay | 60 | 120 | 480 |
| winner whole+segmented replay | 4 | 8 | 32 |
| predict cap / action_encoder cap（独立） | 66 / 66 | 132 / 132 | 528 / 528 |
| fresh Jacobian API/materialization | 11 | 22 | 88 |
| 实际 helper SVD | 12 | 24 | 96 |
| 数学输出 VJP 方向 | 2112 | 4224 | 16896 |
| planner visual encoder | 0 | 0 | 0 |

Backward engine API 次数未知；predict 已包含 AD-forward；不称 FLOPs/完整 pipeline 等预算。Q 构造历史成本仍单列 80 encoder、80 existing-projector、8 SVD；旧 context 成本未知。export 数组重建额外 96 SVD，不属于 fresh CUDA。新增 fsync、RAM copy、AST/hash/history 校验的 CPU/IO 成本真实存在，不声称 v1/v2 总耗时匹配。

## 测试、封存与 main 接口

纯代数/IO fault injection 覆盖：nominal 后 AD 失败、prefix 后 suffix AD 失败、J 后首/次 bank SVD 和 suffix SVD 失败、hash/read/cleanup/root-write/report-write/source/model-state secondary、主/备 report 都失败、journal fsync 二次失败；正常八臂 mock runner/export、Q 改变生成、own-conditional J、原算法等价、zero/duplicate/tie、预 dispatch 非退款双 caps、视觉入口禁止、history/layout 污染、prepare 源漂移和 seal coverage。mock/数组结果不是科学效果证据。

最终封存前四文件联合测试：**85 passed in 12.46s**，其中 v2 33 项，v1 与两参照文件合计 52 项。另补测 hook 部分安装回滚、多项 restore 失败逐项保留，以及 fsync 失败后即使日志能读回也必须标 `durable_journal_complete=false`。这些均未加载真实神经模型或物理环境。

所有源文件完成最终测试后才 prepare。新 prepare 会冻结新源码/测试/本文和 AST 可达的 experiments/stable model/pretraining Python closure，包含 lazy imports；verify 重新计算 closure 集合、检查 current source bytes、每 seal 的 protocol args 与精确文件覆盖。静态封存不等于证明执行了这些字节，也不认证 compiled binary/runtime libraries。

从 `le-wm-main`：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_p3_goal_patch_consistent_compiler_20261027_v2.py tests/test_p3_goal_patch_consistent_compiler_20261027_v1.py tests/test_pusht_full_packet_compiler_20261027.py tests/test_p3p4_goal_patch_metric_20261027_v1.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m experiments.spectral_reachability.p3_goal_patch_consistent_compiler_20261027_v2 --prepare --output results/p3_goal_patch_consistent_compiler_20261027_v2_prepared
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m experiments.spectral_reachability.p3_goal_patch_consistent_compiler_20261027_v2 --verify --prepared results/p3_goal_patch_consistent_compiler_20261027_v2_prepared --prepare-sha256 PREPARE_SHA
```

prepare 路径必须不存在，禁止重用 v1；精确 prepare/source SHA 在交付消息给出，避免把自身摘要写进被封存本文形成循环。这里只执行纯测试、prepare/verify，不执行 `--execute-cuda`。

main review 后若另行授权 v2 CUDA，必须使用新的独占 output、v2 module 和本 prepare 的 SHA。v1 CUDA/物理结果不能改 schema 后冒充 v2，v1 consumer 不自动适配 v2 report。P1/P3/P4 admission、physical_quality_measured 保持 false；独立新根数为零。
