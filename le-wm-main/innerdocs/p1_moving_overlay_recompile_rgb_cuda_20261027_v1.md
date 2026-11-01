# Moving overlay8：真实 RGB CUDA evaluator（独占实现交付）

本文件只说明新 evaluator，不替代 MAIN 审读，不修改共享协议、progress、dispatch、closure 或任何旧实现。仅 MAIN 审读后在 HOST CUDA 运行。实现者只执行 RAM/NumPy mock tests 与 CLI help；没有 prepare、神经执行、物理、训练、数据集或联网。

## 输入与边界

消费 Bohr `p1_moving_overlay_recompile_physical_20261027_v1.report`，必须由 MAIN 提供**已经存在**的真实 report 路径及独立 SHA256。不能预填尚未生成的 SHA。prepare 绑定真实 report、execution_binding、physical prepare、journal 文件 SHA/hashchain、原 paired source、Nash saved compiler consume、checkpoint PNG、四份 blind NPZ、原 weights/config 和实现代码闭包。verify 从这些保存输入重建 prepare，不执行 planner/physics。

原 paired source report SHA 固定为 `85ca1df3de910628bfd6c5bd1a0a482f61880a2ee76bfd404732a77dd8c7666a`。fixed rows 为 slot0/1 × off/on × `line_search_GN`/`full10_conditional_D2P`，顺序固定，8 臂不可过滤。每臂 step0/5/10，缺失保持 `latent=null,Q=null` 和 missing 状态；不补图、不重试、不置零。

Bohr checkpoint 六字段：`step,png,file_sha256,pixel_sha256,state,snapshot`。只选择 journal `fresh_observation` 中 phase=plan、off 的 overlay_index0 或 on 的 overlay_index1；要求同一 branch/slot/condition/arm/selection_digest，state/snapshot 与真实 PNG path/fileSHA/pixelSHA 绑定。physical report 无 contract 字段，读取绑定 prepare 的 contract。未完成物理行仍保留，其物理 status/errors/counters 单独保存；嵌套 public_step/physics_step 等计数分别报告，不相加成额外独立调用。

每个 slot/condition 的 goal 来自原 NPZ 的**实际 goal_rgb**，并验证完整 packet 的已绑定 history/goal/raw SHA。默认 renderer goal_pose 不是构造的任务 goal，不据其重画、修补或改 goal 图片。四个目标分别编码；即使像素相等也不去重。NPZ 只允许三个原字段，文件≤2MiB、展开≤1MiB、成员≤512KiB、allow_pickle=False。

本次 moving history 严格为 float32 `(10,2)`：前五步 `[0.1,0]`，后五步 `[0,0.1]`，以 `np.array_equal` 匹配固定 float32 数组。**不是旧 zero-history**；全零、交换时序、任何尺度改变或 float64 都拒绝，不放宽为任意 history，不作第二次除100。

## 原模型链与封存

原 weights SHA `48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607`；config SHA `2564086e961e7b5c7c04dffc451091115b389a590645ff19653c64fd0bc16e09`；运行前后冻结 tensor SHA `4b12c01f5859f4b46bea478f3a8d5e2d3f516df456a622a3feec31bde7e33546`。

沿用原 `image_transform`/`_image_tensor` 和 `model.encode`，模型 eval/requires_grad=False，CUDA-only，batch1，float32，TF32 matmul/cudnn 均关闭，MATH SDP，inference_mode，无 autocast。禁止 predict/predictor/action_encoder/pred_proj 和 AD API；没有新控制、physics 或训练。prepare/help/tests 不导入 torch。运行 lazy import 原 preflight 模块，但不构造任何 env。

从绑定 config 的 image_size224 / patch_size14 推导 `(224/14)^2+1=257` tokens，width 来自 projector.input_dim192。保存并验证：原 RGB、原 PNG/NPZ bytes、实际 transformed `[1,1,3,224,224]`、encoder input `[1,3,224,224]`、**完整 hidden `[1,257,192]`**、实际 CLS/projector input `[1,192]`、projector output `[1,192]`、原 emb `[192]`。逐元素验证 transformed reshape=encoder input、hidden[:,0]=projector input、projector output[0]=emb。197-token/patch16、CLS 断链或 emb 不一致均拒绝。

每次真实 module 调用有独立 attempt/completed/failed 原始 receipt，记录实际输入输出；失败尝试不会当成功返回。正常上限24 checkpoint+4 goals=28 encoder/28 projector。4 goals先编码，再 row-major checkpoints；重复 checkpoint 也显式执行，不按 pixel/hash 去重。

所有原始数组、module receipts 与 emb 先 fsync 到 NEW 独占目录，写 `encoding_seal.json` 后才做 Q 算术。encoding-only 不产生新 prediction，`prediction_status=not_recomputed_encoding_only`。Q 为 float64 的192维平方差均值，与自身 slot/condition 的真实 goal latent 比较，不拟合。任意编码或 final model/source 验证失败时不产生 metrics，Q 全 null。输出总量≤32MiB（含 raw/final/error），普通写入预留2MiB，free floor200MiB；600s deadline 覆盖 verify/load/model/encode/final reverify，并在 Q 前再次检查。deadline 耗尽后只允许有限 final/error 持久化，不再发起模型调用或 Q；无强杀已经进入的单次底层 CUDA/文件调用。磁盘已不可写时无法保证新日志落盘，已保存文件不删除。

## MAIN 精确 CLI（本交付没有执行）

从 `/home/gbk/code/LeWorldModel/le-wm-main` 运行。先由 MAIN 设置 PHYSICAL_REPORT 为实际 report 的绝对路径、PHYSICAL_SHA 为其独立复算 SHA；不要把示例/未来 SHA 当证据。prepare 和输出路径必须从未存在；若已存在，选择新后缀，不覆盖旧结果。

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
.venv/bin/python -m experiments.spectral_reachability.p1_moving_overlay_recompile_rgb_cuda_20261027_v1 prepare \
  --physical-report "${PHYSICAL_REPORT:?MAIN actual existing report absolute path}" \
  --physical-sha "${PHYSICAL_SHA:?MAIN independently recomputed report SHA256}" \
  --output results/p1_moving_overlay_recompile_rgb_cuda_20261027_v1_MAIN01.prepare.json
```

将 prepare 打印的真实 prepare_sha256 存入 PREPARE_SHA，MAIN 审读 prepare 后：

```bash
.venv/bin/python -m experiments.spectral_reachability.p1_moving_overlay_recompile_rgb_cuda_20261027_v1 verify \
  --prepared results/p1_moving_overlay_recompile_rgb_cuda_20261027_v1_MAIN01.prepare.json \
  --prepare-sha "${PREPARE_SHA:?actual generated prepare SHA256}"
.venv/bin/python -m experiments.spectral_reachability.p1_moving_overlay_recompile_rgb_cuda_20261027_v1 run \
  --prepared results/p1_moving_overlay_recompile_rgb_cuda_20261027_v1_MAIN01.prepare.json \
  --prepare-sha "${PREPARE_SHA:?actual generated prepare SHA256}" \
  --authorization MAIN_MOVING_OVERLAY_RECOMPILE_RGB_CUDA_ONLY \
  --output results/p1_moving_overlay_recompile_rgb_cuda_20261027_v1_main01
```

运行结束不代表科学通过。复算 report SHA 后，MAIN/Poincare 可选 saved scoring bridge：

```bash
.venv/bin/python -m experiments.spectral_reachability.p1_moving_overlay_recompile_rgb_cuda_20261027_v1 consume \
  --output results/p1_moving_overlay_recompile_rgb_cuda_20261027_v1_main01 \
  --report-sha "${RGB_REPORT_SHA:?independently recomputed actual RGB report SHA256}"
```

Python 稳定入口 `consume(output, report_sha256)` 返回固定8行：

```python
{"root_slot": 0, "condition": "off", "arm": "line_search_GN",
 "selection_digest": "actual compiler selection digest",
 "checkpoints": [{"step": 0, "latent": receipt_or_none, "Q": float_or_none,
                  "status": status, "image": index_or_none}, ...]}  # steps 0,5,10
```

latent receipt 为 `{file,sha256,array_sha256,shape,dtype}`，原数组为 gzip NPY；consume 验证文件/raw SHA、config-derived shapes、原模块链、封存关系并重算 saved Q，拒绝 RAM cuda_verified=False。此环境无跨线程消息工具：以本接口供 MAIN 直接转交 Poincare，不声称已私下通知 Bohr/Poincare。Bohr 对接按其实际落盘新源码 schema，不凭文件名推断。

## 测试与限制

```bash
.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_p1_moving_overlay_recompile_rgb_cuda_20261027_v1.py
.venv/bin/python -m experiments.spectral_reachability.p1_moving_overlay_recompile_rgb_cuda_20261027_v1 --help
sha256sum experiments/spectral_reachability/p1_moving_overlay_recompile_rgb_cuda_20261027_v1.py tests/test_p1_moving_overlay_recompile_rgb_cuda_20261027_v1.py innerdocs/p1_moving_overlay_recompile_rgb_cuda_20261027_v1.md
```

47 RAM tests 覆盖固定分母/condition/root/selection/snapshot/journal 关联、缺帧和失败无替换、无去重、seal先于Q、模型漂移阻断、预算/授权/不伪造未来SHA、257完整链、saved consume schema、非零moving history接受及zero/时序/尺度/dtype变更拒绝。带 cuda_verified=True 的 fixture 只是注入式 schema测试，绝非CUDA运行凭证或科学证据。

本任务没有实际读取 MAIN 的新 physical report，也没有执行 prepare/CUDA/physics。真实 CUDA 接线、真实文件大小和终态由 MAIN 审读后运行验证。既有16阻断、training stats来源、物理root equivalence、完整reset bound/global freshness均未因此清除；未知仍未知，不制造官方export。`P1_admission=False` 始终保留。遵循 experiment-code 的执行/记录边界，但按用户限制不生成额外图、文件、CPU神经实验或训练脚本。
