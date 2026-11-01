# Moving-history paired overlay fidelity CUDA v1

三文件实现交付，以最终消息 SHA 为冻结标识。本 stem 只实现 evaluator，不执行 producer、不重渲染、不规划、不训练。
MAIN 负责全部源码审查、prepare 和主机 CUDA。实现 goal 与科学 P1 分开；科学 P1=false。

## 已对齐的 acquisition 边界

Arendt producer schema 为 `p1_moving_target_overlay_source_20261027_v1`，Jason plans
schema 为 `p1_moving_target_overlay_plans_physical_20261027_v1`，不冒充旧 tag。
实际 source/prepare/report 文件 SHA 必须由 MAIN 显式提供，尚未生成时不能填未来 hash。
现 MAIN 已实际提供完整 source186PNG 与 plans112PNG，以下命令绑定其真实 SHA；本实现侧未实际读取 material/prepare/运行。
无可用 agent-message 工具，接口通过此文档及 MAIN 传递，不声称已直接发消息。

source 的 blind NPZ 严格三字段：`rgb_history:uint8[3,224,224,3]`、
`raw_action_history:float32[10,2]`、`goal_rgb:uint8[224,224,3]`。
history 来自 replay0 真正 global0/5/10；raw 前5[.1,0]后5[0,.1]，不再 /100。
off 必须逐位等于旧 moving blind packet；on 来自相同 body 真 render。
off goal 是原始图；on goal 是 construction global20 真 render，不由消费者解 privileged state 合成。
native default outline 的 goal_pose **不是**构造 task goal；不得称 marker 已与任务目标对齐。

Phase A `factual_witness` 必须 source replay0 的 global15/20 对应 root-relative5/10，
各自 off/on 真实标签，并绑定 witness commit、实际 raw returns 和 journal。
producer 仅 triplets0/5/10/20 时，缺 on15，本 phase 在模型加载/输出创建前拒绝。
Arendt 已封存并由 MAIN 实际完成0/5/10/15/20（186renders），缺口已由真实 acquisition 补齐；
消费者仍按实际合同及 receipts 检查，不造 on15，不拿 construction goal 充当 replay 标签。

Phase B `fixed_planner_terminal` 消费 Jason 固定2slots×GN/D2P 的真实 plan5/10 off/on，
经 `(slot,arm,selection_digest,root identity,commands)` join，绝不按图片序号猜角色。
actual plan0 的 off/on 必须与 source history 最后一帧逐位一致；goal 从 source 来，
Jason 的 `source_goal_on_certified=false` 不能被改名成 source goal 证据。

Jason 的 `fresh_observation` 先于派生 `old_off_match` journal 写入；该字段存在于后续
`guard_receipt` 与 row checkpoint receipt，不能对 journal observation 索取该字段。
消费者按 `(phase,step,mode,png)` 一一 join guard，要求 accepted、纯渲染与 old-off 五项匹配；
checkpoint 核所有 observation 共有字段及 guard 派生字段，不做不成立的 whole-dict equality。
该工程接线修复不修改 sealed producer，不是 CUDA/科学负结果。

## 固定神经合同

两 phase 分开独占 run，无跨 phase 隐含缓存。A=20 encoder/projector +16 predict/action_encoder/
predictor/pred_proj：每根每条件 history3+actual未来2，H1、freshH2首段、freshH2末段、
同条件真实 prefix teacherforcedH1。B=32 encoder/projector +24上述 forward：
每根每条件 history3+goal1+每 arm actualprefix/terminal2；各 arm freshH2两段及 teacherH1。
不新增 F/Z/M，不重做旧 zero-history。

两段 action tokens 分别由 `[warmup0:10,commands0:5]`、
`[warmup5:10,commands0:10]` 以原 float32 mean/std 重建。
open H2 第二段 latent 为 `[real5,real10,predicted15]`；teacher 为同条件同 arm actual15。
goal/Q/labels 不进入 predictor；所有 forward 原数组持久化后才评分。
B 原 identity192 Q，报告 predicted/actual same-condition terminal Q 与 GN-minus-D2P rank（保留 ties），
以及 native192 successor MSE、两种 persistence 和同时间 off/on latent 距离。
不根据任何效果选根、改 Q、调阈值或替换动作。

复用 MAIN 已修封存 archive helper `f536cf5fe9baa4c652b17888f6ff0112bccc5aa241c567581a034b5ec8ccc31c`
的完整 NativeTrace/Store/save_array，不修改其源。完整 native 模组输入输出及 ViT patch/CLS 均原精度
NPY lossless gzip；file/npy/raw 三层 SHA 分列，不丢数组过 cap。CUDA-only、MATH、no TF32、no AD，
before/after tensor hash，attempt/return、全分母、失败部分数组保留，禁止重试。
600s、3GiB allocated/4GiB reserved、8MiB output、512KiB final reserve、200MiB floor；
prepare 限256KiB，只元数据不复制图像/Q矩阵，超过则报设计错误，不自行调高。

## 科学边界

MAIN 实际 material 核验修正：物理 plans 的 `compiler_report_SHA` 绑定命令 handoff
`52c475ed499b077f261517bc592252b208926919909674cebf68fa4029cc8310`，
不是原始 CUDA report 的 `cd843…`。后者位于 handoff 的 projection/whole_run 字段。
错误混用这两层哈希导致 phaseB 工程准入失败，不构成科学负结果。保留旧 prepare，修复后另建 prepare。

原 roots/旧固定动作的 paired rendering observation-effect 校准，非独立确认、非 P1 成功，
不证明 training lineage 或排序已修。旧 archive 六 ratio 下降不外推此 moving task；
旧 zero-history 3/4 absolute MSE 改善且 persistence 不总改善也不混为本轮结果。
真实效果必须等 MAIN 的新实际报告；RAM mock 只验证接口/算术/失败协议。

## 实现测试与输出估算

65项 RAM tests 通过；不创建临时目录/results，不导入/执行真实模型或模拟器。
覆盖两个真实 schema、186/112 完整 census、四元 guard join、原 off/goal/动作位绑定、
缺 on15/partial 拒绝、同条件 teacher chronology、原 identity Q、固定完整分母、失败部分数组、
数组 file/NPY/raw 三层哈希、CLI verify→material A/B 端到端 RAM mock。
完整 synthetic native tensors（含全部 encoder 输入、ViT patch/pooler、projector、AE、predictor、pred_proj）
A 压缩3,402,888字节/原NPY18,599,264，B压缩5,412,003/原NPY29,714,912。
这不是对真实 compressibility 的保证；实际仍硬限8MiB，包括全部输出及512KiB终态预留。
模型/来源/limits/哈希/final-journal 检查分别保留错误，首 execution 错不由 finally 检查覆盖。
不裁原精度数组过cap，不删任何既有资产。

## MAIN 命令（实现侧未执行 prepare/build/CUDA）

cwd `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_moving_target_overlay_fidelity_cuda_20261027_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.p1_moving_target_overlay_fidelity_cuda_20261027_v1 --help
.venv/bin/python -B -m experiments.spectral_reachability.p1_moving_target_overlay_fidelity_cuda_20261027_v1 prepare \
  --phase factual_witness \
  --source-report results/p1_moving_target_overlay_source_20261027_v1_main01/report.json \
  --source-sha 85ca1df3de910628bfd6c5bd1a0a482f61880a2ee76bfd404732a77dd8c7666a \
  --producer-source-sha 768a45d127ad01a0e3ee8b55a6c7c535c92134211af10914e3e99bb4674e472b \
  --producer-prepare-sha 88957b9d251595e5e35ed46ba4a9aa003eb937bbfff0f58ae12540418dcb39a1 \
  --output results/p1_moving_target_overlay_fidelity_cuda_20261027_v1_factual_prepare01.json
.venv/bin/python -B -m experiments.spectral_reachability.p1_moving_target_overlay_fidelity_cuda_20261027_v1 prepare \
  --phase fixed_planner_terminal \
  --source-report results/p1_moving_target_overlay_source_20261027_v1_main01/report.json \
  --source-sha 85ca1df3de910628bfd6c5bd1a0a482f61880a2ee76bfd404732a77dd8c7666a \
  --producer-source-sha 768a45d127ad01a0e3ee8b55a6c7c535c92134211af10914e3e99bb4674e472b \
  --producer-prepare-sha 88957b9d251595e5e35ed46ba4a9aa003eb937bbfff0f58ae12540418dcb39a1 \
  --plans-report results/p1_moving_target_overlay_plans_physical_20261027_v1_main01/report.json \
  --plans-sha d40c32e8c59df7c5552937cf76ee43525ae2aae0b595340cd325662b8ea686c0 \
  --plans-source-sha ec5c9d383b09f9db2475710645f680946c6ebabf966d8785405e96751a2f200b \
  --plans-prepare-sha 72c10ab251ab426c2e4e6d1b2a7ef8a19d697075c4e2e2d4b8bd76314885d048 \
  --output results/p1_moving_target_overlay_fidelity_cuda_20261027_v1_plans_prepare01.json
```

各 prepare 返回真实 `prepare_sha256`，MAIN 独立验证后分别执行，目录必须全新：

```bash
.venv/bin/python -B -m experiments.spectral_reachability.p1_moving_target_overlay_fidelity_cuda_20261027_v1 verify --prepared ACTUAL_PREPARE_PATH --prepare-sha ACTUAL_PREPARE_SHA
.venv/bin/python -B -m experiments.spectral_reachability.p1_moving_target_overlay_fidelity_cuda_20261027_v1 run --prepared ACTUAL_PREPARE_PATH --prepare-sha ACTUAL_PREPARE_SHA --output NEW_EXCLUSIVE_PHASE_OUTPUT
```

`material(**same_arguments)` 是 MAIN 的纯 saved 入口，返回 `(own_prepare,typed_roots)`；
不导入 torch、不执行环境、不创建 output。`prepare()`在 material 完整通过后才独占写入。
`verify()`重建全部闭包与原始标签，`run()`验证后再创建目录并加载 CUDA 模型。
当前实际端到端 material、prepare字节数和主机CUDA结果仍待 MAIN 验证；本侧不能据 mock 宣称实际完成。
