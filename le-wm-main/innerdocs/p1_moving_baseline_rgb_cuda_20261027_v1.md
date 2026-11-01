# Moving 强基线真实 RGB 评价适配器

仅实现 source/tests/innerdoc；MAIN完整读代码、审计、prepare与实际HOST CUDA。
实现agent不运行神经、物理、prepare或actual saved build。旧overlay三文件保持冻结。
使用experiment-code的固定实验计划、计账、失败保留合同；不训练、不拟合Q，不按效果迭代选根。

## cohort 与边界

原target-free moving两根、原goal、原rootlatent、原action normalizer不变。支持在physical
manifest准备前固定非空arms子集：优先 `moving_gradient_open_loop`（2分支、4图、4encoder/
4projector）。MAIN本轮事前声明gradient＋CEM＋nominalMPPI，同两根6分支、12图、
12encoder/12projector；不等待kinodynamic、不按结果择支。接口也覆盖全部四基线
（8分支、16图、16encoder/16projector），但那是另行事前固定的cohort。
四arm名称以Arendt实际接口为准：`CEM`、`kernel_faithful_mppi_nominal`、
`moving_gradient_open_loop`、`fixed_H2_kinodynamic_beam16x4`。
RGB arms必须和已事前固定的physical manifest完全一致，不得读完物理结果再从全八分支挑子集。
每分支必须真实full10且有prefix5/terminal10；合法native早停是physical有效记录，但不能把
早停帧冒充这里的terminal10，缺任何required checkpoint在CUDA前封闭，不补图不缩分母。

新stem：`p1_moving_baseline_rgb_cuda_20261027_v1`；prepare/report各自schema，完成状态
`moving_baseline_rgb_encoded_not_admitted`。knownroot校准而非独立样本或P1/P2准入。
不同planner仅继承此前各自已付预算，不声称全部算力匹配。

## 已确认的真实接线

读取Arendt已落地的 `p1_moving_baseline_physical_20261027_v1.py` 和innerdoc后实现，
不猜旧GN/zero接口。当前工具无send_input，未声称已向另一agent发送消息。

- 输入显式physical report path/SHA、prepare SHA、producer source SHA；实际未知值由MAIN
  完成物理后提供，代码不伪造成功或SHA。source SHA与已import模块文件核对，全文closure绑定。
- physical `.report` / `moving_baseline_physical_recorded`，`fixed_branch_denominator`与rows。
  按 `(root_slot,arm,selection_digest)`关联，无source_slot假设。
- `.prepare`经execution_binding读取；manifest含roots/arms/authorities，不假设prepare有source_slots。
  plans含task_export/task_sha256/task_id/blind_sha256/commands/compiler_authority/
  compiler_report_sha256/own_plan_sha256/selection_digest。
- 每个authority调用physical白名单的真正 `ownconsume(output,reportSHA)`：sampling/gradient返回
  rows，kinodynamic返回plans。其内部按已封存producer SHA导入，复核完整原模型报告与保存链。
  本adapter重新构造每个physical plan并要求完整等于prepare；不使用自报commands替代权威。
- gradient绑定MAIN明确actual report SHA
  `96c140caee14bba43bf766d981c21be1724a36c0047fde4abcc2892c24bf5b7d`。
- sampling绑定MAIN实际ownconsume通过的report SHA
  `9ac889d6be76bf3241f85c9bfcac13936784dcf80171b3fe48abf5c9cfcd319c`，
  目录 `results/p1_moving_sampling_baselines_cuda_20261027_v1_cuda_main01`。
  本轮三臂依此及gradient actual接线；未知实际physical SHA仍封闭，不降级严格核验。
- 原moving encoder `load_task`，只接受原task00/01的实际封存SHA
  `a42bfbb1ea692782355e2572a5df2cd048b72cf3e6c41fee40eb12503674959b` /
  `101add21a7a40591167a6ab8e8b22c3c62cb1fd700c741fd36cd2073e2d5df74`。
  不调用zero-only loader、不读取witness证书供选择，source/evaluator依赖仅hash。
- 全physical journal链、实际嵌套operation census、全部22帧/branch的guards及所有PNG SHA/像素
  核验。actual moving warmup raw10及RGB3绑定原task。全own10commands、goal attachment、
  root+goal identity与checkpoints逐一闭合。不是仅相信report成功标签。

## CUDA 与输出

仅编码prefix5/terminal10，固定完整cohort，无dedup，无新root/goal编码。
冻结native float32 projectedCLS192；CUDA MATH SDPA、TF32off、eval/inference/frozen model，
原模型tensor SHA前后相同。每张实际encoder/projector模块guard和ledger；禁止predict、action
encoder、pred projector、AD API；newpredict/AD/physics全部0。无CPU神经fallback。

16MiB整目录cap、200MiB floor、512KiB final reserve，启动需216MiB，300秒encoding wallcap。
保存原native192 NPY与fileSHA、image_attempt/module_attempt/module_return/image_saved事件、
全部输入/源码/模型/NPY/事件冻结hash、actual attempted/completed、首错与secondary错误。
部分失败保留所有预置图行，未开始/失败/已编码分开；无重试/替补。最后来源/模型/产物检查
失败不能报完成。

指标独立为 `mean((actual192-goal192)^2)`、原rootQ、Q变化、actual-vs-root MSE。
本适配不新增想象rollout，也不从异构planner schema猜prefix预测；predicted_Q/MSE明确null、
`unknown_not_recomputed`，不能把未知预测称0误差或冒称理论负结论。
每张回调只接真实PNG数组，不接物理state/witness，state只用于离线provenance对齐。

## MAIN命令

cwd `/home/gbk/code/LeWorldModel/le-wm-main`。实现agent只运行下面RAM pytest；其余MAIN执行。
必须用实际physical SHA替换占位符，未知时不得prepare/run：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_moving_baseline_rgb_cuda_20261027_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.p1_moving_baseline_rgb_cuda_20261027_v1 prepare --physical-report /ABS/ACTUAL_PHYSICAL/report.json --physical-sha ACTUAL_PHYSICAL_REPORT_SHA --physical-prepare-sha ACTUAL_PHYSICAL_PREPARE_SHA --physical-source-sha FROZEN_PHYSICAL_SOURCE_SHA --arms moving_gradient_open_loop --output results/p1_moving_baseline_rgb_cuda_20261027_v1_gradient_prepare01.json
.venv/bin/python -B -m experiments.spectral_reachability.p1_moving_baseline_rgb_cuda_20261027_v1 verify --prepared results/p1_moving_baseline_rgb_cuda_20261027_v1_gradient_prepare01.json --prepare-sha ACTUAL_RGB_PREPARE_SHA
.venv/bin/python -B -m experiments.spectral_reachability.p1_moving_baseline_rgb_cuda_20261027_v1 run --prepared results/p1_moving_baseline_rgb_cuda_20261027_v1_gradient_prepare01.json --prepare-sha ACTUAL_RGB_PREPARE_SHA --output results/p1_moving_baseline_rgb_cuda_20261027_v1_gradient_cuda_main01
```

本轮三臂命令将 `--arms` 改为 `moving_gradient_open_loop CEM kernel_faithful_mppi_nominal`，
并使用MAIN同顺序三臂physical的实际report/prepare/source SHA和独占新prepare/output名称。
不得对gradient-only physical报告临时增加arm；也不得从四臂physical报告运行后挑三臂。
全部四arm必须另行事前固定physical manifest和新RGB prepare/output，不能覆盖已有结果。
RAM mock不是实际物理/CUDA成功。MAIN独占真实运行、源码审计与科学结论。

本地验证：上述RAM命令42 passed（5.57秒），覆盖gradient4图、本轮三臂12图、
全部四臂16图的数组合同与实际mock module ledger、lossless NPY、错误ownconsume/commands/
task/root/goal/完整history/guard/PNG checkpoint计账拒绝、模块部分失败无重试及输出cap/floor。
未运行prepare、actual saved build、神经或物理；尚需MAIN实际physical report/prepare/source SHA。
