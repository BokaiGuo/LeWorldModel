# GRASP selected-plan physical adapter v1

## 范围与冻结接口

新独占 stem：`p1_constructive_grasp_physical_20261027_v1`。
仅把已冻结 GRASP producer 的 own `consume` 输出接到已冻结 generic
`p1_constructive_plan_physical_20261027_v1.execute`，不实现新控制器、模型或物理循环。
不调用旧 gradient producer，不修改任何旧源码/结果。两根、一个固定 arm
`GRASP_H3_bounded`、每根10条 own raw commands，不能按成本或物理结果重选。
相同命令也不合并根。不读取 witness 命令，不把证书作为候选控制器。

固定 compiler source SHA：
`ae9d2dd59bf613bc10c81858cb767dad1d92dcf9b7b7d103bedb018e0bd1fa28`。
固定 generic executor SHA：
`9dea58dfc6418dd01bf9ab28de6d712189ca1154a5cf29b165efa4b8ef65ea96`。
固定实际 compiler prepare SHA：
`4ee8a2a6526d4ecdf534a54b790150117022b53737323b0d78f0c603a4d710ff`。
固定实际 report SHA：
`501651be13c6dd6b6082bd20513d464c0ffc8aed053bb7a40f1c4ee9cd9388c5`。
report 路径允许显式传入（可迁移相同字节），不能换用不同 report SHA。
MAIN报告 actual consume 已通过；本适配器实现者仅取得单文件SHA，没有自行运行
模型、CUDA、实际 producer consume 或物理。测试中的 report 是RAM synthetic fixture。

## Schema / 权限隔离

`selection_from(consumed, report_path, report_sha)` 是纯映射。
consume schema 必须为 GRASP `.consume`（该接口没有 status 字段），
`official_equivalence=False, P1_admission=False, new_physics=0`。
逐行核对 source_slot 的类型/顺序、固定 blind SHA/task ID/export SHA、arm、
no-target/no-physics 声明、完整 plan digest、全10条 float32-exact raw commands。
不用 prediction/cost 决定是否纳入某臂。模型动作球验证沿用冻结 own consume，
物理端不 clip/round/pad/re-normalize。

生成 generic `.selection`，两行分别映射实际 report 的
`/rows/{0,1}/{commands,source_slot,blind_sha256,arm}`。
每行 `selection_digest` 使用 generic `source.digest` 的默认空格JSON，
不是 GRASP compact canonical；测试直接调用真实 generic.selection 验证兼容性。

`build` 先核 source/report/prepare SHA，再调用 own consume，核 generic report pointers，
合并冻结 compiler closure 和新 source/tests/doc。此阶段不加载 sourcepack/certificate。
`material` 先重建并精确比较整个 manifest，之后才授权 generic evaluator 加载原 source
recipe/state/goal；本 adapter 不读取 witness。输出独立 `.prepare` schema，保留 generic
contract、plans、source slots，加 adapter/report/prepare seals；verify 完整重建，不调用
只认识原 schema 的 generic CLI verify。小 metadata 文件独占创建、fsync 文件和父目录，
每份小于256KiB，不复制大型模型或闭包。

`run` 首先检查 MAIN physics authorization，再 verify，创建独占 Store，写
`execution_binding.json`，调用一次原 generic.execute。实际 report 仍为 generic `.report`；
绑定文件指向本 adapter prepare。任何 partial report 的 CLI exit code 为2。

## 原样物理合同

每臂从原 seed → sample_state → placement 重建，不能使用 saved poststate reset。
10步 raw-zero warmup 复现 blind 的0/5/10三帧及 state/dynamics；这10步是真实新增物理。
evaluator 绑定原构造 goal 与 goalRGB，无 physics/render/reset/setter，`with_target=False`。
之后只执行该臂10条命令，遇 native terminal 或 guard 失败立即停止，不替补、不填满。

两根完整预算（上限及 attempted/completed 均由 generic计账）：

| 项目 | 两臂完整值 |
| --- | ---: |
| reset | 4 |
| public_step | 40（20 warmup + 20 own） |
| explicit_render | 44 |
| goal_render | 4 |
| set_state | 8 |
| physics_step hook | 408 |
| goal_bind | 2 |
| PNG | 44（每臂 warmup0..10 和 post-attach plan0..10） |

wall上限600秒；输出≤8MiB，另留200MiB磁盘安全底线。
沿用 generic 的独占output、预置全分母、hashchain、原始轨迹、native/object-only指标及失败计账。
fresh capture 先保存 RGB/state/snapshot/receipt，再记录 guard、停止；RAM反例覆盖这个顺序。
如果 `step` 内部抛异常，不能保证故障后新帧，旧 last_observation 不代表故障现场。
native terminal 始终优先尊重；native角误差未mod可能为负，wrapped诊断单独保留，不改环境放行。
无 predict/AD/encoder/projector/训练调用；已知两根不是新独立样本，不给予科学P1准入。
GRASP自身为显式 H3/bounded extension，物理执行不把它升级为官方数值等价或等算力基线。

## MAIN命令（从 le-wm-main，依次执行）

以下命令由MAIN执行，不是实现者已执行记录。先使用MAIN审定的实际 report；selection 会再次
走冻结 own consume，失败就不生成 manifest。所有 output 必须尚不存在。

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_p1_constructive_grasp_physical_20261027_v1.py

.venv/bin/python -m experiments.spectral_reachability.p1_constructive_grasp_physical_20261027_v1 selection --report results/p1_constructive_grasp_baseline_20261027_v1_cuda_main01/report.json --report-sha 501651be13c6dd6b6082bd20513d464c0ffc8aed053bb7a40f1c4ee9cd9388c5 --output results/p1_constructive_grasp_physical_20261027_v1_selection01.json
```

从 selection 回执读取 SHA，设置 `GRASP_SELECTION_SHA` 为真实值；从 prepare 回执读取 SHA，
设置 `GRASP_PHYSICAL_PREPARE_SHA` 为真实值。不预设或伪造尚未创建的文件SHA。

```bash
.venv/bin/python -m experiments.spectral_reachability.p1_constructive_grasp_physical_20261027_v1 prepare --manifest results/p1_constructive_grasp_physical_20261027_v1_selection01.json --manifest-sha "$GRASP_SELECTION_SHA" --output results/p1_constructive_grasp_physical_20261027_v1_prepare01.json

.venv/bin/python -m experiments.spectral_reachability.p1_constructive_grasp_physical_20261027_v1 verify --prepared results/p1_constructive_grasp_physical_20261027_v1_prepare01.json --prepare-sha "$GRASP_PHYSICAL_PREPARE_SHA"

.venv/bin/python -m experiments.spectral_reachability.p1_constructive_grasp_physical_20261027_v1 run --prepared results/p1_constructive_grasp_physical_20261027_v1_prepare01.json --prepare-sha "$GRASP_PHYSICAL_PREPARE_SHA" --output results/p1_constructive_grasp_physical_20261027_v1_main01 --authorize-physics
```

## 验证界限

本stem测试只做RAM mocked dispatch/metadata与纯数组环境，不创建临时结果、不调用模拟器/神经。
复用冻结 generic 的测试 ArrayEnv/MemoryStore 来验证继承合同，不冒充独立实际物理证据。
覆盖两臂不去重、逐raw/pointer/digest绑定、无status的consume接线、错误源/闭包/外部SHA拒绝、
modified manifest在evaluator前拒绝、budget漂移拒绝、授权先行、完整44PNG/计数、失败帧保存后停止。
实际结果是否成功由MAIN运行所得原生记录决定；模型计划、mock通过或物理完成都不自动构成P1成立。
