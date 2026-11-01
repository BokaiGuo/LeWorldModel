# Moving-history observed-suffix terminal RGB adapter v1

实现任务，非独立审计。只新增本stem source/tests/本doc，不改任何旧seal、物理runner或共享文档。
实现者只运行RAM/mock测试，不actual consume/build/prepare/CUDA/物理；MAIN完整审读后执行。
使用experiment-code的固定预算、可执行入口和失败记录原则，不因实际结果调整固定分支。

## 最小设计

固定两根 × `line_search_GN` / `full10_conditional_D2P`，按root-major顺序仅编码四张
新physical的真实 `plan/terminal10` PNG。0个新prefix编码；prefix来自已冻结旧moving RGB
的对应prefix5 NPY。不得按收益/成功、误差或latent值删支、替图、重试。缺图/partial一律拒绝。
MAIN给出的native距离变化不进入任何筛选条件，也不预设理论正/负结论。

不是复用旧planner report冒充feedback结果：NEW compiler自己的consume/verify验证动作与prepare，
NEW result的 `evaluations[1]` 必须是`new_suffix`，与`new_suffix_replay` context精确相同，
其terminal native192是本次预测比较对象。旧suffix只作为已支付的compiler诊断，不参与派发选择。
目标和Q仍来自原moving task的goal/native192 identity mean-Q，不训练、不拟合Q、不新rollout。

## 实际输入绑定

新compiler report：
`results/moving_history_observed_suffix_cuda_20261027_v1_cuda_main01/report.json`
SHA `79f07b28722991ce40a01cdfa2c40fa0e51f149803db00e1617d1ccecb0966fc`。
新physical report：
`results/moving_history_observed_suffix_physical_20261027_v1_main01/report.json`
SHA `3376eda7d0a88b31e8fb56a5f3b2cc19ef4854ef5b12f35ae196da0744c71f0e`。
对应实际physical execution_binding给出的prepare SHA：
`5bcd47b14a02a07162640349b26d315e11b7ee7b016628556eef42162f760972`。

原RGB report仍独立绑定 `7ec03c0a6a6e8411ff0dd0115044a872d099106edc95e4529b97c38c2b9b8f71`。
原source/任务/模型/normalizer及原prefix字节从NEW compiler prepare闭包继承；不改变旧任务。

源码pins：

- NEW compiler `3e72cef9375c5fa5b010593e2285e8eeb8d6b51b71310b70663b7f7406a7eb56`
- NEW physical `07215161b9ea620efe83ff7ca7ae2838d2f27b40ca16f5cc3aa969626813b2ce`
- 原moving RGB `ed9b80bebb212434d60ddd55d3a3cba01e5cc55306170a0ea6f9a167bb156469`

以上是输入身份/程序约束，不是本agent独立确认实际结论；实现者未运行actual saved审计。

## NEW schema与身份链

物理schema `moving_history_observed_suffix_physical_20261027_v1.report`，
status `moving_history_observed_suffix_physical_recorded`；不能用旧moving physical tag替代。
准备schema也必须是NEW physical `.prepare`，核合同、原sourcepack和original_physical_report SHA。

MAIN执行build时先NEW ownconsume、NEW verify并绑定全部artifact和prepare闭包；
以 `(source_slot,arm)` 连接四个compiler branches/plans/results，通过branch_id、parent_task_id、
planner_selection_sha、observed_context_sha、prefix/suffix原始命令和model_plan self-seal再逐项确认。
同一原task必须显式经moving encoder `load_task`装载，绝不走zero-only旧loader。

新physical report→execution_binding→固定外部prepare SHA→manifest SHA→compiler_authority
的source/report/own-consume三SHA；manifest中的own-consume文件必须与NEW consume返回完整相等。
显式JSON pointers匹配slot/arm/blind/commands，不依赖物理rows里不存在的source_slot。
物理以 `(root_slot,arm,selection_digest)` 与plan/branch_started/branch_closed关联，不按效果选择。
原sourcepack/certificate等evaluator-only文件只hash bytes，不解析状态或witness选根/选命令。

全operation attempt/completed栈必须闭合：8reset、80public、88render、8goalrender、16setter、
816physics、4goalbind。四臂完整10步、initial_exact、prefix_exact、prefix_goal_exact、无error，
原journal首项prepare/末项status/counts/errors/tail全部一致；任何部分失败不编码。
每臂22fresh observation/guard完整、先persist后guard、state/obs/snapshot对齐，全部own10命令一致。
17个paired_prefix_exact覆盖warmup0..10及plan0..5，file/pixel SHA一致；必须在step6 dispatch前
存在真实suffix_admitted_after_exact_prefix，并核prefix snapshot和goal。四臂分母永不减少。
plan0/5/10 checkpoint与相应journal receipt逐项相等，terminal是实际step10而非旧endpoint。

原prefix NPY只读取不重编码。entry路径/原始NPY文件SHA/shape/dtype/latent、对应旧RGB收据、
planner selection和新physical prefix PNG file/pixel/state/snapshot/observation_state必须一致。
prefix像素不因重放改名而改身份。完整input SHA闭包和新三文件都进入新prepare，最终再次复验。

## 编码、指标、预算与故障

MAIN-only HOST CUDA，原冻结模型、MATH SDPA、matmul/cuDNN TF32 off、inference_mode，
GPU同步和module guard；encoder/projector batch1，原生float32[192] NPY，前后tensor hash。
只有4encoder/4projector，0predict/actionencoder/AD/新physics/训练/Qfit；无CPU神经fallback。
重复PNG也不去重。guard阻止额外predictor/pred_proj/actionencoder/AD执行。
原compiler和physical/旧RGB8成本另记，不退款、不称全算力相等。

metrics：actual_Q、root_Q、actual_Q_change、actual_vs_root_mse、NEW predicted_Q、
predicted_vs_real_mse、actual_minus_predicted_Q、改进方向符号反转；另保留原prefix_Q、
actual_vs_prefix_mse、actual_Q_change_from_prefix。全部是原goal的mean192平方误差。
预测来自NEW observed-context rollout封存结果，不是旧model terminal、虚拟目标或真实state拟合。
原prefix只作为对齐过的观测基线，四个新terminal指标不代表任务准入或新增独立样本。

独立Store词法绑定8MiB总cap、256KiB final预留、200MiB磁盘floor，启动至少floor+cap。
不修改原generic/parent的2MiB全局常量。输出仅4NPY、events.jsonl、binding.json、report.json，
不复制PNG/权重/大闭包；输出目录独占，新prepare<256KiB。
四行先预置，durable image/module attempt-return-saved，partial保留已保存证据，不重试/替代。
最终model/source/hash/listing/report分别尽力记录；secondary不覆盖首错，模型未加载时unchanged=null。
输出schema为本stem `.report`，完成状态 `observed_suffix_terminal_rgb_encoded_not_admitted`；
P1/P4 admission=false，无新独立样本、科学成功或理论结论声明。

## MAIN最小可执行命令

cwd `/home/gbk/code/LeWorldModel/le-wm-main`；均是MAIN后续命令，不是已运行证据：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_moving_history_observed_suffix_rgb_cuda_20261027_v1.py

.venv/bin/python -B -m experiments.spectral_reachability.moving_history_observed_suffix_rgb_cuda_20261027_v1 prepare --output results/moving_history_observed_suffix_rgb_cuda_20261027_v1_prepare_main01.json
```

设置 `MOVING_SUFFIX_RGB_PREPARE_SHA` 为上述真实回执SHA后执行：

```bash
.venv/bin/python -B -m experiments.spectral_reachability.moving_history_observed_suffix_rgb_cuda_20261027_v1 verify --prepared results/moving_history_observed_suffix_rgb_cuda_20261027_v1_prepare_main01.json --prepare-sha "$MOVING_SUFFIX_RGB_PREPARE_SHA"

.venv/bin/python -B -m experiments.spectral_reachability.moving_history_observed_suffix_rgb_cuda_20261027_v1 run --prepared results/moving_history_observed_suffix_rgb_cuda_20261027_v1_prepare_main01.json --prepare-sha "$MOVING_SUFFIX_RGB_PREPARE_SHA" --output results/moving_history_observed_suffix_rgb_cuda_20261027_v1_cuda_main01
```

## RAM验证边界

测试使用封存physical测试的纯数组ArrayEnv，不执行native物理或神经。NEW schema、full10、
prefix/journal/NPY、goal、source/authority/observed context与model prediction故障均拒绝；
四terminal计量和float32 NPY精确、partial不推进、8MiB/200MiB门与最终重建覆盖。
无临时结果、无actual prepare/build、无审计agent。mock通过不等于真实RGB或科学结论。
