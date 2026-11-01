# P2 safeguard 两因素 paired ablation 准备

仅新增同stem source/tests/innerdoc，不改旧结果。此交付是 **saved 资产表 + 缺失 forward 的 literal prepare/verify**，不提供 CUDA/physics run，不声称科学准入。固定原8分支、2 known roots、off/on及joint/conditional身份；不选有利行。

## 既有资产能回答什么

绑定原 recompile v1 CUDA report SHA `8146692ccc9a3c2c6c09ff3c99419274e2e8a8ff86bb3ac0465af01dc6401eea`、v2 SHA `363925ff3714c6c9a5fbd2ec6c59e86dbcb63fe685cde1945d292ec1a9196117`，调用两版原 `consume_compile` 重建全源码/模型/prepare/数组/module/API/AD/event闭包。两版 actual-mid context/history10/prefix/raw incumbent/goal/Q/normalizer/remaining cap逐项相同才可配对。

原 v1 为 zero nominal/J → bounded ridge → 无条件替换，v2 为 incumbent raw J → half-radius gradient direction → 四alpha模型选择+guard。因此直接将v1、v2叫“只改变线性化点”的2×2是混淆solver/候选集的。代码保留每分支4格 `legacy_confounded_reconstruction`：v1原新候选无guard / 与旧incumbent比较后guard；v2四提案不含incumbent / 加入incumbent。模型分数确可saved重算，但不称其为干净因素实验，物理分支不能自动补造。

## 新的固定可识别合同

统一使用 v2 原提案公式、原goal及identity192 MSE、float64 tanh→float32 raw chart、原fulljoint20与remaining suffix logit cap（无epsilon、inverse、clip或修复）。因素A只改 anchor：zero 或 literal old incumbent。沿用原冻结J：v1在float32 tanh logits=0求导，局部tanh'=1，故该点等于raw J；v2是literal incumbent raw J。新候选生成仅NumPy算术，不再求AD。两种anchor都用连续float64 tanh proposal surrogate，不声称量化chart本身可微。

每anchor：`g=2*(Jraw*sech²(u_anchor)).T@(z_anchor-goal)/192`，half-radius negative-gradient内点，方向为该内点减anchor；非下降方向置0。固定alpha `[1,.5,.25,.125]`，全部4词保留不去重。因素B：无guard=四提案stable argmin(Q,alpha_index)；有guard=同四提案加原incumbent index0，ties保旧。**无guard不是强制alpha1，也不是原ridge策略**。该定义事前固定，仅依据模型Q，不读取真实终态、成功或收益。此合同不能追认是原算法实际部署行为。

全8×2anchor×4alpha=64提案槽、8 incumbent、32cell均保留。任一必需预测缺失，该anchor两cell selection=null；不取可用子集。逐rawbits/context/goal精确查找既有预测，重复必须完整context/Q一致；原incumbent侧应完全复用v2候选。zero侧同公式候选与旧ridge不同，缺项生成真实module输入 `(3,192)` latent / `(3,10)` normalized actions、raw5、logits/domain、原history10/goal/mean/std与dispatch ABI。prepare不会把待运行预测填成0或赋予假成本。

## 物理资产与合法复用边界

引用MAIN已封存strict物理receipt：v1 `c191b8076469f78fb7e02c793e0b061bcf6706266b1af6e90f3944748db2e11d`；v2 `7e71cf504691cb1b132c437267319f3291a62b0f6c9f839f0652b42a974f8de8`。验证实际SHA、compile SHA、8分母、每row task/selection/compiler digests与原计划身份；只对同分支精确full10 bits匹配旧control/v1/v2已执行计划。这里输出 **sealed strict receipt reference**，不是再次执行PNG/native审计，不复制物理标签到选择器，也不自动physical release。表内native_metrics=null；某cell没有对应实际分支明确missing，缺模型选择则unknown。所有新选中动作若不完全命中这些分支，后续需要MAIN另行真实物理授权，不能拿别的动作outcome代替。

## 预算与未决实现

继承两版完整原predict/module/AD资源计数单列，API inclusive与内模块不相加。缺失请求数量写 `new_compute`，phase明确为planned_not_executed，`actual_new_compute=null`、worker_neural_calls=0；最大64 native forward，各action_encoder/predict/pred_proj同数，新增AD/图像/physics=0，total_FLOPs=null，不称等FLOPs。未来执行上限预声明900s、64MiB output（2MiB final reserve）、磁盘floor200MiB；当前prepare/verify为saved检查，不声称已经对未来调用实施这些运行限制。

下一最小新增实验是仅执行所有缺失候选的固定原native suffix forward（预计zero侧32次），无需重跑原8编译/AD/图像/物理。**还缺reviewed durable CUDA dispatcher与strict returned-array/module-entry/return消费接线**；prepare明确run_available=false，不提供callback假运行receipt。MAIN须把这些literal请求接入冻结 `F.d2p.TorchBackend.step(deepcopy(input_context), rawfloat32[5,2])`，真实完成并严格验收全部候选后，才按本合同封存32cell选择。物理分支缺项再另算，不能预知其数量/费用。known-root calibration非P1、非fresh、非P2成立。

## MAIN saved CLI

cwd `/home/gbk/code/LeWorldModel/le-wm-main`。只写不存在的本stem输出。

```bash
.venv/bin/python -m pytest -q tests/test_agent_p2_safeguard_factorial_20261028_v1.py
.venv/bin/python -m experiments.spectral_reachability.agent_p2_safeguard_factorial_20261028_v1 prepare --zero-output results/agent_p2_contextual_midfeedback_recompile_20261028_v1_cuda_MAIN01 --incumbent-output results/agent_p2_contextual_midfeedback_recompile_20261028_v2_cuda_MAIN01 --zero-physical-receipt results/agent_p2_contextual_midfeedback_recompile_20261028_v1_physical_consume_MAIN01.json --incumbent-physical-receipt results/agent_p2_contextual_midfeedback_recompile_20261028_v2_physical_consume_MAIN01.json --output results/agent_p2_safeguard_factorial_20261028_v1_prepare_MAIN01.json
.venv/bin/python -m experiments.spectral_reachability.agent_p2_safeguard_factorial_20261028_v1 verify --prepared results/agent_p2_safeguard_factorial_20261028_v1_prepare_MAIN01.json --prepare-sha <ACTUAL_PREPARE_SHA>
```

实际新测试11 passed in3.41s；新tests+冻结v2 compiler43项+MAIN guard11项联合65 passed in16.53s；CLI help退出0。真实saved schema接缝未skip：全8、32cells、64提案保留，incumbent侧32提案全有预测，zero侧32提案全缺，准备请求32条均合法，原module输入维度精确接通。另实际执行完整两版consume_compile和本build的只读内存smoke，退出0：status=blocked_missing_model_candidates、rows8、requests32；incumbent侧16cell全部有精确full10 receipt引用，zero侧16cell selection/physical均unknown。没有写生产prepare或新saved结果。smoke启动后补充的最终改动仅为same-parent-plans显式相等断言（实际saved另核True）及planned/actual预算标签；没有改候选或配对逻辑，新11项回归已再次通过。MAIN需依据最终冻结三件套重新prepare/verify。

这些检查只读原JSON/NPY并重算，不调用模型/物理；不等于未来forward已完成，也不是科学证据。worker未运行神经、物理或网络。
