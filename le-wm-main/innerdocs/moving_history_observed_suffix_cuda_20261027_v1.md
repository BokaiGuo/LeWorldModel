# 固定 moving-history prefix 的 Execute → Observe → Recompile 校准

实现文件：`experiments/spectral_reachability/moving_history_observed_suffix_cuda_20261027_v1.py`。
测试：`tests/test_moving_history_observed_suffix_cuda_20261027_v1.py`。
本交付仅实现与 RAM/NumPy mock 测试；未 prepare、未运行模型或物理。
MAIN 独占后续执行。它不是独立任务成功、非贪心证据或 P1/P4 准入。

## 输入与无选择反馈

固定两根 × `line_search_GN` / `full10_conditional_D2P`，全分母四分支，不补根、不重试。
原 moving planner 源 SHA `f8f1d5507e2474fbbbfa56cae98f7bfa99f5929829946a2d7760c2c30ccc03c1`。
实际 RGB report 为
`results/p1_constructive_moving_history_rgb_cuda_20261027_v1_cuda_main01/report.json`，
外部 SHA `7ec03c0a6a6e8411ff0dd0115044a872d099106edc95e4529b97c38c2b9b8f71`；
RGB 源 SHA `ed9b80bebb212434d60ddd55d3a3cba01e5cc55306170a0ea6f9a167bb156469`；
原 prepare SHA `5b5de956f6a6bc45f9569e40d85546f809efd06334f6a8fc5aeaf2fb66947d7a`。

先复用 RGB 自有 `verify` 与 moving planner 自有 `consume/load_export` 验证封存链，
不用旧 zero-only interface，不替换任何任务 module globals。
NPY receipt 使用 `file`（相对 RGB output），不是 `path`。
按 source_slot、arm、task_id/export/SHA、planner selection、physical selection、
`role=prefix5`、`checkpoint.phase=plan,step=5` 联结，不猜 image 数组索引。
完整八图的十项 artifact（binding、journal、八 NPY）都核哈希；
只解析四个 prefix5 的 float32 native192 NPY，terminal10 只核字节 SHA，不加载为目标或选法。

新 context 的三段 latent 恰为原 task `latent_history[1:] + actual_prefix192`。
两段历史动作恰为原 warmup 最后五步 `[0,.1]` 与本 arm 的原 selected prefix 五步，
以冻结 mean/std 执行 float32 normalization，保持 time-major 5×2 → 10 token。
不 `/100`、不裁剪、不反向排列，不使用目标 privileged state/witness 生成新动作。
原 native goal192、identity Q192、normalizer、packet logit ball 保持不变。
已观察旧 suffix 失败或 prefix 误差大小不改变此协议、根、参数或派发。

## 固定计算合同

每分支固定以下顺序，H1 = 五个 raw commands：

1. 独立零 raw nominal：1 predict + 1 action_encoder。
2. 同 context 的零 logit Jacobian：1 AD-forward predict + 1 action_encoder；J 为 192×10。
3. `compiler.bounded_ridge(Q@J, Q@(goal-nominal), damping=.002, cap=.75*sqrt(5))`，一次 SVD。
4. 原 suffix 诊断：1 predict + 1 action_encoder。
5. 新 suffix：1 predict + 1 action_encoder。
6. 同实际 context、独立 replay 新 suffix：1 predict + 1 action_encoder，完整 successor context 必须精确相等。

新 raw 为 `tanh(float32(solve.params)).reshape(5,2)`，保持原 ball 求解，不加投影或重新优化。
旧 suffix 仅作同 Q 对照；无论旧/新预测谁更低，都固定派发新 suffix。
新动作在任何旧/新候选评分前写入 durable journal。

每分支 **5 predict / 5 action_encoder / 5 pred_proj / 1 Jacobian / 1 SVD**；
四分支 **20 / 20 / 20 / 4 / 4**。
每 J 一次 `torch.autograd.functional.jacobian(vectorize=True)` API，
内部一次 `autograd.grad(is_grads_batched=True)` API，对应 192 个数学 VJP；总 768，
不是 768 次 API，也不把 AD-forward漏出 predict 账。
原 source、planner、physical、RGB8 成本继承但单列，不能声称总算力相等。
复用冻结 `d2p.TorchBackend` 与 bounded-ridge kernel；现有 `with_svd_receipts` 仅作用域统计实际 SVD，
不替换任务数据。没有额外预测用于 post-hook、计量或消费验证。
仅 CUDA 冻结 eval 模型；MATH SDP，matmul/cuDNN TF32 false；无 CPU neural、训练、Q fit 或新物理。

## 保存、失败与 release

原 prefix NPY 原始字节直接存档，另外保存 observed history、raw commands、goal、
nominal、**原返回精度** J（float64 容器、源 native float32 数值）、solve params、新 raw 与三次评估 context。
实际 action_encoder、predictor、pred_proj 的完整输入/输出都逐次保存原 float32 NPY：
每 forward 的三段 latent、三段 normalized token、action embeddings、predictor outputs、projected outputs。
模块 pre-hook 在调用前登记 attempt 并写入输入；输出在同步后记录 completed。
Jacobian/SVD/AD API 同样预记 attempt。nominal/J/new raw 完成即写 journal，
不等 root-finally；失败保留数组、部分账目、四分母与首错，不重试、不替换。
hook 移除、最终模型 hash、source verify、journal close、artifact hash 各自留存错误，
不覆盖原执行错误。若存储本身失效只能承诺已落盘证据与 attempt 下界，不能伪造完整 receipt。

`consume` 无模型调用；强核 artifact census/SHA、完整 gzip journal、四 branch seals、
原 prefix 字节、result/plan 一致性、原始数组、五次实际模块链、math/AD attempts 与 returns、
真实 token 算术、完整 context replay 和动作 self-seal。失败报告不可 release。
AD 内部 grad 先返回、外部 Jacobian 后返回，消费端按此真实嵌套顺序核验。

公开 handoff：

```json
{
  "schema": "moving_history_observed_suffix_cuda_20261027_v1.consume",
  "status": "observed_suffix_commands_verified_not_admitted",
  "errors": [],
  "plans": ["四个按 source_slot × arm 固定排序的 model_plan"],
  "new_physics": 0,
  "P1_admission": false,
  "P4_admission": false
}
```

每 plan schema 为 `moving_history_observed_suffix_cuda_20261027_v1.model_plan`。
JSON pointers：`/plans/k/blind_sha256`（原 branch 不变）、`/plans/k/commands`（原 prefix5 + 新 suffix5）、
`/plans/k/prefix_commands`、`/plans/k/suffix_commands`、`/plans/k/source_slot`、`/plans/k/arm`、
`/plans/k/parent_task_id`、`/plans/k/planner_selection_sha256`、`/plans/k/observed_context_sha256`、
`/plans/k/branch_id`、`/plans/k/selection_sha256`、`/plans/k/whole_run_report_sha256`。
`whole_run_report_sha256` 由 consume 添加，不包含在自身 selection digest 中。
physical 必须 NEW，原 prefix 的环境/RGB 重现由该 consumer 自己验证，不继承旧 endpoint。

## MAIN 命令（本 agent 未执行）

在 `/home/gbk/code/LeWorldModel/le-wm-main` 下，先完整 review 三文件，再创建独占 prepare 文件：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_moving_history_observed_suffix_cuda_20261027_v1.py

.venv/bin/python -m experiments.spectral_reachability.moving_history_observed_suffix_cuda_20261027_v1 prepare \
  --rgb-report /home/gbk/code/LeWorldModel/le-wm-main/results/p1_constructive_moving_history_rgb_cuda_20261027_v1_cuda_main01/report.json \
  --rgb-sha 7ec03c0a6a6e8411ff0dd0115044a872d099106edc95e4529b97c38c2b9b8f71 \
  --rgb-source-sha ed9b80bebb212434d60ddd55d3a3cba01e5cc55306170a0ea6f9a167bb156469 \
  --output results/moving_history_observed_suffix_cuda_20261027_v1_prepare_main01.json

.venv/bin/python -m experiments.spectral_reachability.moving_history_observed_suffix_cuda_20261027_v1 verify \
  --prepared results/moving_history_observed_suffix_cuda_20261027_v1_prepare_main01.json --prepare-sha PREPARE_SHA

.venv/bin/python -m experiments.spectral_reachability.moving_history_observed_suffix_cuda_20261027_v1 run \
  --prepared results/moving_history_observed_suffix_cuda_20261027_v1_prepare_main01.json --prepare-sha PREPARE_SHA \
  --output results/moving_history_observed_suffix_cuda_20261027_v1_cuda_main01

.venv/bin/python -m experiments.spectral_reachability.moving_history_observed_suffix_cuda_20261027_v1 consume \
  --output results/moving_history_observed_suffix_cuda_20261027_v1_cuda_main01 --report-sha ACTUAL_REPORT_SHA
```

占位 SHA 必须由 MAIN 对刚生成的实际单文件独立 sha256 后替换，不追认旧结果。
prepare 显式包含四个 Q192 矩阵，单文件上限 2MiB（不是错误的 256KiB）；
run 总产物硬上限 8MiB，含 journal、partial、报告，最终报告预留 256KiB。
正常产物预计数 MiB；实际字节取决于压缩，不承诺固定体积。每次写入维持 200MiB floor，
run 启动还要求 floor + 8MiB；不复制上游大报告或模型、不删除任何资产。
测试 fixture、NPY、journal 均驻 RAM，无 pytest tmp 研究产物。
