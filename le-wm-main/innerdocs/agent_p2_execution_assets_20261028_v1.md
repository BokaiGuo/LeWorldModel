# P2 sealed action-library execution assets v1

本次仅新增本stem source/tests/本文；不修改任何封存源、结果或共享总账。MAIN负责审读及真实physics；worker仅RAM fixture与saved metadata build。无后代，无神经/GPU/physics执行、下载、训练、拟合或阈值改变。旧blocked个人goal占用，不覆盖或虚标完成。本交付有界条件是：源链join、prepare/verify、MAIN-only calibration runner、RAM测试及SHA。

## 真正复用哪份动作库

原四根（不是moving两个根）`agent_p2_runtime_guard_20261028_v1_known_MAIN01/report.json`，SHA `0cfbe37a7783c6df331305e95e3c058a9b006f72bb1ca5b59d24fae047ee23a0`；原wrapper prepare SHA `02269ae5d08d1f645e9bcf2473a7c7cbb2a16f3125c3fad62c23859380cc6c48`。已封存MAIN saved consume `results/agent_p2_guard_saved_consume_20261028_v1_MAIN02.json`，SHA `c7d183f5b18d1bfacd81f855025efdbbb403f97e5442650970080766db17088b`。直接信任这份MAIN验收receipt的外部SHA，并重验其所有574bindings/consumer_sources；不伪称本runner新完成全部独立数组审计，不重复CUDA规划。

每根原joint_h2_no_composition与conditional_suffix_composition各一winner，分别来自原80/40全候选，保存计费1328 predict、480selection；继承资源不冒充物理费用。每条同原root/context/Q/goal/normalizer/raw10完全绑定：join候选ID、完整best digest、receipt SHA、原winner witness和动作、单full10边vs双5边；原库不支持广义独立预编译程序库/任意深度组合。

原 `p2_composition_predict_frontier_physical_20261027_v1.planning_material/prepare/execute_physics` 硬绑定另一份report，故**完全不调用**这些高层接口。不猴补globals。只复用其公开的低层 `PhysicalMeter`、`physical_branch`、`verify_branch`、`DiskGuardEnv` 与原reset/helper函数。内部 `execution_plan` 使用该helper旧plan schema作为明确ABI，不是声称原frontier-v1 producer生成此计划；外层独立schema、source_report_sha256、source_selection_digest、candidate_id、handoff_sha256全部新绑定。

## 可执行接线

`build()` / `prepare()` 无模型环境加载：验证原wrapper/native prepare、MAIN saved receipt全部binding、原4root compilation、历史RGB/goalRGB/source_Q/projection/model paths与旧physical source preparation。自己三文件及原物理helper完整source closure哈希，不复制大闭包。新prepare单文件≤2MiB，floor200MiB，exclusive/fsync，不覆盖。`verify()` 外部prepare SHA后重建完整metadata，拒绝任何漂移/缺臂/错误命令。

`run-calibration` 是MAIN显式独占物理入口，不是formal authorization。自己orchestrate八臂，even root按原arm order、odd反序，动作从封存winner原样执行。复用原collector seed202753010/oversample4生成同4root，root fingerprint/reference精确失败即停止；**不声称这是fresh root，也不新编有限历史reset上界**。原collector reset本身无本runner新有限cap，attempt/return逐项记账；内部physics/reset内部工作未知null。每臂原reset+15 public warmup，真实RGB3、goalRGB和normalized history严格核验；两臂状态身份相同才执行。选中命令10步/臂，上限80；warmup上限120，root collection public step cap0。native terminal可停止实际步数，但固定4root/8arm分母不减少；后续工程错误使剩余臂not_started，不重试/替根。

fresh RoyalBlue guard按原helper每次命令执行；遵守原installed native success/reward（包含agent+object）与原object-pose诊断，不改goal。step内部异常后可能没有新frame，不把旧last observation称故障现场。原native eval若与独立wrapped诊断不一致，原official_checked拒绝，保留工程错误，不更改env角度协议。

输出NEW100MiB额度+200MiB floor，900s为branch/public-call边界检查，不能终止阻塞native调用。每branch render仍沿原helper更严格的16MiB磁盘余量门；总输出cap在branch/public调用边界检查，单个render/write可能在下一个检查前增加字节，**不是逐byte硬cap**。预留1MiB最终report；完整attempt/return journal、每步PNG、逐branch结果、外层固定八行、错误和输入/产物SHA。close/source/hash/report分别尽力保存，磁盘故障不保证final report写成。source/goal/model全部SHA；不载模型，newpredict/AD/encoder/projector=0。可后续MAIN新增真实endpoint RGB消费者，本runner不伪造latent。

## CLI

在 `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p2_execution_assets_20261028_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_execution_assets_20261028_v1 prepare --output results/agent_p2_execution_assets_20261028_v1_prepare_MAIN01.json
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_execution_assets_20261028_v1 verify --prepared results/agent_p2_execution_assets_20261028_v1_prepare_MAIN01.json --prepare-sha ACTUAL_PREPARE_SHA
# 仅MAIN完成源/测试审读并显式授权calibration physics后：
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_execution_assets_20261028_v1 run-calibration --prepared results/agent_p2_execution_assets_20261028_v1_prepare_MAIN01.json --prepare-sha ACTUAL_PREPARE_SHA --main-review-sha ACTUAL_MAIN_REVIEW_SHA --output results/agent_p2_execution_assets_20261028_v1_physical_MAIN01
```

review SHA是MAIN审读记录引用，不是密码、P1 authority或科学证明。prepare/verify/build不执行物理，不需要P1通过；run也仅known-root校准，formal_execution_authorized/P1_admission/P2_admission始终false。

未完成：MAIN真实prepare/verify/physics、真实endpoint编码和原任务比较；freshness/独立样本、P1/P2科学准入、全硬件成本等价均未解决。测试不证明效能、真实复现或P2。保留零、负、earlystop及工程失败，不从结果选择根/臂。
