# Moving scheduler comparison + actual RGB：新独立saved consumer

只新增本stem source/tests/innerdoc，不修改旧comparison/RGB/physical/scheduler或任何封存结果。worker只运行RAM/mock测试；MAIN独占实际saved执行、独立复核与准入。不训练、不加载模型、不调用神经/物理、不拟合Q、不挑策略或根。按experiment-code固定输入、指标和故障测试；不生成新benchmark或独立auditor。

## 输入与authority

原comparison：`results/moving_budget_scheduler_comparison_20261027_v1_main01/report.json`，MAIN批准且本实现固定SHA `1eb2c51d6d4459b4aa6723fafed40ca8034729aa4f27d4359c28964125840b64`。

新RGB：`results/moving_budget_scheduler_rgb_cuda_20261027_v1_cuda_main01`，MAIN提供actual report SHA `08b96eb2f861be9a84e96738e030aa423ac4a0e4b6541f95355bd01b2d2554e7`，prepare SHA `cfa1326c7d16f9617e999a66153c72545792cae82783c82a26b0de087f249672`。report SHA从CLI显式传入，prepare从报告和ownconsume严格认证，不自行扫描最新目录。

固定comparison/RGB各自已冻结source/tests/doc SHA（代码PINS六项），比较前逐项重hash。原comparison使用其saved `inputs(...)`完整重建并与批准报告canonical精确比较，不信任自报身份/指标。RGB仅调用其新own `consume(output,report_sha)`，要求完整16images/8branches、35文件目录census、112events、原模型及原task/goal/PNG/NPY完整闭包；不调用run/prepare/encoder。缺失RGB保持原comparison的missing，本新报告在RGB缺失/partial时拒绝发布，不补0、不补预测或用pose代理。

报告双方必须绑定同physical report path/SHA、physical prepare SHA、冻结physical source SHA，以及同scheduler report SHA。全部模型/物理/源/原task和RGB artifacts的saved闭包合并逐hash验证，禁止冲突SHA。新三文件的hash也绑定在输出；不复制PNG/NPY/journal，只存引用和指标。

## 固定配对合同

原known moving2根×固定U_uniform/D_geometry/S_swapped/N_no_signal_uniform四臂，完整8分支，每分支prefix5/terminal10，共16image。稳定身份按root_slot/source_slot、root_sha256、task_id、blind_sha256、arm、physical selection、compiler model selection和compiler report联合核验，不按数组位置/收益排序。输入可重排，但不接受重复、bool根、遗漏或未知策略。

仅接受full10共同H2、native_steps=10且不truncated；5/10 checkpoint必须逐字段与原comparison checkpoint引用一致。raw initial state+goal重算root hash，与RGB provenance native root/goal/task_contract一致，同根四臂必须完全同latent root/goal/task/Q。原comparison已通过的physical/public/hidden-state/guard合同不改变，不将partial endpoint混入fullH2。

输出`original_comparison`完整deepcopy保留原报告，每个新row的`comparison_row`也完整保留原行；原missing字段不被悄悄改写。真实RGB单列`prefix5`/`terminal10`（含ownconsume身份、PNG/NPY引用及metrics）。原文件不写入，不修订其结论。

actual_Q严格identity mean192：mean((actual_native192-original_goal192)^2)。root_Q使用批准原root latent与同goal；核actual_Q_change=actual_Q-root_Q、prefix对自身pixelMSE/Qchange为0、terminal的Qchange仅减同branch prefix。RGB ownconsume独立复算原PNG/NPY和真实pixelMSE，本adapter只join其真实回执，不再次编码。

原模型端点成本`original_planner_terminal_Q`从原comparison.model_Q单独保留；`actual_terminal_minus_original_model_Q`只是同Q端点数值差，不能等同pose/native改善或因果效果。未知native预测、prefix预测以及RGB predicted_Q/predicted_vs_real_MSE保持null，不能用已有scalar model_Q虚构预测latent或补未知prefix。

逐根D−U/D−S/D−N：actual prefix/terminal Q差、原model_Q差、双根terminalQ差的描述均值。model差与actual差严格异号时标排序反转；任一差为0时返回null（tie不是反转或确定无反转）。不用不同任务的绝对Q排名，不输出p值/CI或泛化结论。

## 资源与状态

新saved assembler神经/物理调用均0。继承RGB实际16encoder/16projector和原comparison全资源账，保留source acquisition/anchor/scheduler/physical成本；不是新增这些调用，更不是等FLOPs/总预算/耗时比较。不聚合不同量纲的modelQ、actualRGBQ、native reward、object pose。原2known roots非独立确认，不同策略预算不据此变等。

输出新目录仅report.json，1MiB cap、200MiB floor（启动至少201MiB）；exclusive写入、文件与目录fsync，拒绝覆盖。失败在发布完整报告前抛错，不产生假完整结果。status=`saved_comparison_RGB_joined_not_admitted`；P1/P2/P3、independent_confirmation、full_C3_demonstrated、compute_advantage、walltime_advantage全false。

goal API仍被已撤销但未完成的旧review blocked目标占用，不能新建goal；未虚假完成旧目标或科学P1。

## MAIN执行命令（worker未执行）

cwd `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_moving_budget_scheduler_comparison_rgb_20261027_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.moving_budget_scheduler_comparison_rgb_20261027_v1 \
  --comparison-report /home/gbk/code/LeWorldModel/le-wm-main/results/moving_budget_scheduler_comparison_20261027_v1_main01/report.json \
  --comparison-sha 1eb2c51d6d4459b4aa6723fafed40ca8034729aa4f27d4359c28964125840b64 \
  --rgb-output /home/gbk/code/LeWorldModel/le-wm-main/results/moving_budget_scheduler_rgb_cuda_20261027_v1_cuda_main01 \
  --rgb-report-sha 08b96eb2f861be9a84e96738e030aa423ac4a0e4b6541f95355bd01b2d2554e7 \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/moving_budget_scheduler_comparison_rgb_20261027_v1_saved01
```

这里只交可执行saved consumer，不需要prepare或GPU。worker测试不创建tmp/result，不实际读取/编码PNG或重建actual comparison；MAIN已提供的actual状态仅是外部输入，RAM通过不冒充本新consumer actual执行成功。
