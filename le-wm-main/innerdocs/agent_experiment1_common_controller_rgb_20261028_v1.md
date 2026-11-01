# Common-controller RGB8 tracking（calibration only）

固定两根 × arbitrary_latent_waypoint / dynamics_compiled_witness 四工程分支，每支实际 prefix5、terminal10 两帧，共八图。六科学槽仍包含两个 existing_learned_prior unknown；未知不是失败率，不删除，也不假造可编码端点。known-root 校准不是完整 E1、官方 learned-prior 对比、独立样本或 P1 准入。

## 输入权威与 strict bridge

只接真实 physical MAIN02 report SHA `a75d97bfd5e6c19bf942ce5510ad7689ad3fd79f7bb3539b2ca96a6cda9fec6c`，prepare SHA `160517a999680baf8bc79809e36d01e0d5ed8ade69b54706ff48edb6b5786586`。绑定 full physical strict receipt `results/agent_experiment1_common_controller_physical_20261028_v1_consume_MAIN02.json`，SHA `3926373a79e5c0eb9a6295dadd4ce1403dfca13d9baad51639cd7012c913c12b`，重跑已封存 saved consumer 并与该文件全量精确相等，不接受摘要替代。consumer 闭包包含原 CUDA report `bc3cd74c85f878cdfc7b170d0b5b555827e0dc4a59cdaea6fb5781e71f03767f` 与 compiler strict receipt。

复用 frozen moving load_task，不走 zero-history padding loader。原非零 raw 历史前五 [.1,0]、后五 [0,.1] 保留，原 goal/root/context、normalizer、任务合同、模型权威全部绑定。各分支 own raw10/selection/root/task 必须与原物理及 compiler 对齐；原 journal、88 PNG、native7、reset/render/step/physics 账由已封存 full consumer 验证。缺端点/early stop 不能补帧，prepare 拒绝；不能按结果缩四分支。

## CUDA 执行与封存

只八张实际 PNG，按 root-major/arm-major/step5,10 顺序，不按像素去重。goal/root 使用原任务已冻结 latent，不重复编码；继承成本与新成本分开。原 native 224RGB transform / encoder / projected CLS192，hidden shape 从冻结 config 导出并严格检查 [1,257,192]，不是197。

实际模块 encoder / projector 各八 API、八 samples、八 frames，batch1；0 predictor/action_encoder/pred_proj/AD/physics。MAIN 必须显式 HOSTCUDA 授权；无 CPU 神经 fallback、训练、拟合、重试或选根。MATH、float32、TF32/autocast off，冻结模型 tensor SHA 前后相等，完整 source hash 前后验证，非法模块由 guard 拒绝。每模块 intent/return、全部预处理/输入/hidden/CLS/output NPY、原 PNG/RGB NPY 与最终 latent 均保存。

所有八图 encoding_seal.json 先落盘，再允许任何 Q/tracking 算术；模块不完整或 finalize 失败保留四分支/八图/六槽和错误，指标不生成。report 的 actual attempts/completions 不退款。32MiB 输出总 cap（含2MiB final reserve）、200MiB floor；CUDA 前要求整个 cap+floor 可用，600s bounded-check wall 从 run 的 Budget 创建开始（包含 verify/load，非抢占式）。prepare 排他 <=2MiB；不复制大模型/原闭包。不能把测试通过当科学通过。

## 指标（float64 算术，mean192 identity）

每端点分别记录：Q=actual latent 到原 goal；intermediate_target_MSE=actual 到该 arm 已封中间 target；actual_tracking_MSE 在5用中间 target、10用原 goal；model_Q、model_tracking_MSE 使用原 sealed own prediction；prediction_MSE/maxabs/bitwise 是同端点预测与实际编码误差。任意 latent target 原精度保留，不强制 float32 截断。actual_tracking_cost=.5 prefix target MSE +.5 terminal Q，模型对应值另存。native7 初终态和原 success 原样分栏，RGB Q 不是任务成功，object-only 误差也不是完整任务判据。

consume 不调用 score_rows/evaluate；读取保存 NPY，独立 sum/192 重算全部指标、完整模块链/intent/return、encoding seal、每个 artifact SHA 和严格目录 census，最终再次核外部闭包。它验证保存产物而非独立重跑编码或证明模型机制。

## MAIN 命令（inner le-wm-main）

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_experiment1_common_controller_rgb_20261028_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.agent_experiment1_common_controller_rgb_20261028_v1 prepare --physical-report results/agent_experiment1_common_controller_physical_20261028_v1_physics_MAIN02/report.json --physical-sha a75d97bfd5e6c19bf942ce5510ad7689ad3fd79f7bb3539b2ca96a6cda9fec6c --output results/agent_experiment1_common_controller_rgb_20261028_v1_prepare_MAIN01.json
.venv/bin/python -B -m experiments.spectral_reachability.agent_experiment1_common_controller_rgb_20261028_v1 verify --prepared results/agent_experiment1_common_controller_rgb_20261028_v1_prepare_MAIN01.json --prepare-sha ACTUAL_PREPARE_SHA
.venv/bin/python -B -m experiments.spectral_reachability.agent_experiment1_common_controller_rgb_20261028_v1 run --prepared results/agent_experiment1_common_controller_rgb_20261028_v1_prepare_MAIN01.json --prepare-sha ACTUAL_PREPARE_SHA --output results/agent_experiment1_common_controller_rgb_20261028_v1_cuda_MAIN01 --authorization MAIN_COMMON_CONTROLLER_RGB8_CUDA_ONLY
.venv/bin/python -B -m experiments.spectral_reachability.agent_experiment1_common_controller_rgb_20261028_v1 consume --output results/agent_experiment1_common_controller_rgb_20261028_v1_cuda_MAIN01 --report-sha ACTUAL_RGB_REPORT_SHA
```

ACTUAL placeholders 仅由 MAIN 使用各前序命令真实输出替换。实现者仅 RAM tests 和 saved build smoke，不 prepare、不运行模型或物理。封存后任何修复用新版本，不原地编辑。
