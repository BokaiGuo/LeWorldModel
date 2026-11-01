# Moving baseline selected-plan physical consumer v1

仅新 source/test/innerdoc；不改旧 source/renderer/planner，不运行物理、CUDA 或 CPU 神经。
MAIN 审读、prepare、实际执行及科学判断。本实现不是独立审计器。
固定原 moving slots0/1，不是新独立根，不因源 witness reachable 认定 planner 成功。

## 真实接口与 authority

只准入 MAIN 冻结的三源：

|authority|module stem|source SHA|own consume 返回|
|---|---|---|---|
|sampling|p1_moving_sampling_baselines_cuda_20261027_v1|f30a1c5ac5ef618cfe3add37326aae29513278fe7c4319b5a743d9f6422300e6|rows|
|gradient|p1_moving_gradient_baseline_cuda_20261027_v1|3224d6a094e74b3845c2c695c6c85fe29c6522cd4a88c91c57f2cc03d127a70e|rows|
|kinodynamic|p1_moving_kinodynamic_baseline_cuda_20261027_v1|66c090cb98ad8a954bae5c94e39a8cbb707b5a2e8dc82027e63fa00f2583ae6d|plans|

三者调用都是 `consume(output_directory, external_report_sha)`，纯 saved，无模型执行。
未知 authority/自定义模块路径拒绝，源 SHA 核对在动态 import 前，消费后绑定 prepare、全部
artifacts、原模型/配置/normalizer/import闭包。不是使用用户提供的绿色 handoff 绕开 ownconsume。

arms 只能为 `CEM`、`kernel_faithful_mppi_nominal`、`moving_gradient_open_loop`、
`fixed_H2_kinodynamic_beam16x4`；prepare 前一次冻结非空有序子集，run 不接受覆盖。
支持只 gradient 的两个分支，无需等其他 actual。sampling 若 ownconsume 含两 arm，允许提前声明
仅执行其中之一，仍完整核验其 ownconsume；不得按其 cost/物理结果临时改 manifest。
authorities 固定使用顺序 sampling→gradient→kinodynamic，仅列本 manifest 用到者。

## 精确新 manifest

外部 SHA 必须由 MAIN 提供；以下是 gradient-only 字段示例，不包含虚构 actual SHA。
所有路径必须绝对路径；ROOT_SHA 来自原 sourcepack 对应 slot，非新 overlay source。

```json
{
  "schema":"p1_moving_baseline_physical_20261027_v1.selection",
  "sourcepack":"/ABS/p1_constructive_moving_history_source_20261027_v1_main01/sourcepack.json",
  "sourcepack_sha256":"c2d8d573f94884ffc081163f25f2bfa269f98ffe8155887704109219910ffec6",
  "source_prepared":"/ABS/ORIGINAL_MOVING_SOURCE_PREPARE.json",
  "source_prepare_sha256":"MAIN_ACTUAL_SOURCE_PREPARE_SHA",
  "roots":[{"root_slot":0,"root_sha256":"SOURCE_ROOT_SHA0"},{"root_slot":1,"root_sha256":"SOURCE_ROOT_SHA1"}],
  "arms":["moving_gradient_open_loop"],
  "authorities":[{"authority":"gradient","output":"/ABS/ACTUAL_GRADIENT_CUDA_DIR","report_sha256":"MAIN_ACTUAL_REPORT_SHA"}],
  "selection_fixed_before_physics":true,
  "physical_outcome_selection":false,
  "wall_seconds":600
}
```

无需手填 commands 或 JSON pointers。prepare 从成功 ownconsume literal raw10x2 生成全部
slot-major/arm-order plans。每 plan 自带 task/blind/compiler report/own plan digest/selection_digest。
命令必须有限、恰可 float32 表达且在[-1,1]，否则 prepare 拒绝、不 clip、不 dispatch。
task bridge 使用原 moving encoder `load_task` 完整出口核验，并将 task_contract 的 RGB3/goal/raw
字节 hash 与 evaluator 原 source refs 对接；root/goal/context 不凭 opaque slot 名推断。

## 物理 kernel 与预算

只复用 prior moving `load_source`、`branch`、`native_hooks`、`capture`、Ops 数学/操作核；
不调用 prior execute/material/selection，不 patch 旧 globals。自身 schema、manifest、execute、Store。
完整重做原双 reset/placement、真实 float32 raw warmup 前5[.1,0]后5[0,.1]；旧0/5/10 RGB、
state、完整 exported body snapshot（含 bit 字符串）逐位一致后才 attach 原独立 derived goal，
再执行本 arm 的 literal10 commands。goal attachment 不改 goal_pose/动力学/相机；始终 with_targetFalse。
不是 paired off/on overlay 分支，不复用 witness 作为候选，不继承旧 endpoint。

|资源|gradient-only两分支|全四arm八分支|
|---|---:|---:|
|reset / goal_render 各|4|16|
|set_state|8|32|
|public_step (warmup+selected)|40|160|
|internal physics|408|1632|
|goal_bind|2|8|
|explicit_render / PNG|44|176|

每完整 branch22PNG（warmup0..10 + post-bind plan0..10）；plan0/5/10在 checkpoints。
fresh frame/state/snapshot 先 durable 后 guard，保留失败图、所有 attempted/completed、固定分母。
selected 原生 terminated/truncated 立即停，合法 native partial 按实际步数核资源，不补步/重试。
继承旧 kernel 的 warmup premature terminal 是 reconstruction failure、未释放 selected commands；
保留实际 warmup 和零 selected receipt，不将其称完整同 root 物理比较。
physics=4*branches+10*实际public；reset/setter/goalrender/bind与每 branch 前置一致；
非完整或异常调用不伪造 fullcap。关闭/restore 错误保留首错，final hash/journal/fsync失败不报完成。
snapshot 不覆盖未导出的 solver 内部cache。new neural/训练均0；此前模型搜索/视觉成本仅继承，
不算本次实际新调用，也不声称不同算法总算力公平。

prepare<256KiB；run全目录<=16MiB（含partial、journal、final），final reserve512KiB；
启动需216MiB，逐写保留200MiB；wall<=600s。实际超cap必须 partial/fail-closed，不能省掉receipt。
运行输出自有 `.report` schema，`moving_baseline_physical_recorded` 只表示记录/计账完整，
不是全部full10/native成功/P1通过。`fixed_branch_denominator`、`full_horizons`、全部 rows 分开呈现。

## MAIN 命令

cwd `/home/gbk/code/LeWorldModel/le-wm-main`，只使用 NEW prepare/run 路径：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_moving_baseline_physical_20261027_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.p1_moving_baseline_physical_20261027_v1 prepare --manifest /ABS/NEW_SELECTION.json --manifest-sha ACTUAL_MANIFEST_SHA --output /ABS/NEW_PREPARE.json
.venv/bin/python -B -m experiments.spectral_reachability.p1_moving_baseline_physical_20261027_v1 verify --prepared /ABS/NEW_PREPARE.json --prepare-sha ACTUAL_PREPARE_SHA
.venv/bin/python -B -m experiments.spectral_reachability.p1_moving_baseline_physical_20261027_v1 run --prepared /ABS/NEW_PREPARE.json --prepare-sha ACTUAL_PREPARE_SHA --output /ABS/NEW_PHYSICAL_DIR --authorize-physics
```

执行方未 prepare/actualbuild/运行物理；RAM fixtures不是实际有效性证据。
剩余前置：MAIN实际 baseline report SHA、事前 fixed manifest、源码审读、prepare/verify及物理执行。
三源 whitelist 是本轮已批准固定 authority，不接未来未冻结模块、不回填success、不另写审计器。

## 本地实现验证

上述禁pyc/cache RAM命令实际 **39 passed**。包括 gradient-only 两分支与全部八分支完整mock、
三 authority 的真实签名/返回key、白名单源 hash 前置拒绝、slot/arm/原始命令/重复缺失反例、
history/goal像素桥、exact root重建、selected native early stop、失败PNG先存、首错/cleanup、
final journal/report/fsync失败、资源predispatch与floor、prepare重建反例、静态本地源码闭包。
八分支全部176张PNG在RAM实际PNG编码、完整journal序列化及report合计验证小于16MiB，
final report小于512KiB；不薄化任何mock receipt。实际纹理/绑定/错误可能增大产物，真实cap仍硬拒绝。
此处未生成prepare/result/tmpfixture，未启动模型/物理，未声称实际八臂已完成或科学准入。
