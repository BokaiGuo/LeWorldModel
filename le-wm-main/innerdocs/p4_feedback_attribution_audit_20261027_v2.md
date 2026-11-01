# P4 terminal20 派生证据独立saved审计 v2

新增独立v2，不改原attribution auditor v1、其报告/“当时未保存真实step20 latent”结论，亦不改已封存terminal20 encoder。只读取保存文件、解码PNG、流式hash、做NumPy算术；不导入encoder/runtime/compiler/torch，不加载模型，不调用神经/AD/物理。

## 输入与核验

默认v1证据：`results/p4_feedback_attribution_audit_20261027_v1_verify02/report.json`。
默认真实派生receipt：`results/p4_terminal20_cuda_encoding_20261027_v1_cuda_main01/receipt.json`。

独立保留prepare.json SHA `54b8aaee4d5cb58844eac0b6741e86a4f8b6eb4dadc9acfb178e93e6905a3495`、encoder源码SHA `b0656a1e3feab4d277d5d3159e8a4fbea5267ee390670291df08be2a3fafdcd2`和v1源码SHA `f73e10ad32f33908d0199331834c45de9fe8e6f957dfa7eab582fc302e483216`，不从新receipt盲信这些身份。

完整复验v1输入字节并用不可变v1纯保存函数重建全部round/history/RGB/budget结果，与原report逐字段相等。再核对terminal20 prepare/execution/launch关联、封存source/archive/args、原report/protocol/journal、原冻结weights/config流式SHA与loader/transform历史绑定。goal/Q由原协议取数组并核对独立规范SHA，不复制旧大protocol或checkpoint。

严格重建encoder→existing_projector每臂各一次、全局各两次的完整attempt/return/endpoint journal；核对成功status、零blocked/errors/forbidden、HOST CUDA allocation、模型前后state hash相等、0predict/AD/physics。这里只验证保存收据及文件，不重新计算神经forward或模型tensor-state hash，不声称独立观测过旧GPU执行。

通过source_report SHA/root SHA/arm/step20/event index精确连接原始observation、prepare endpoint、derived row；复核PNG文件/像素SHA、state字段及native192 NPY的独占路径、SHA、float32/shape/finite，禁pickle。数组metric与receipt保存Q代价相符才进入算术。两个臂保持同一批、同一已用根；缺第二批不被新编码补造。

## 新增数值与科学边界

每臂15→20输出原预测Q、step15真实编码Q、step20真实编码Q及真实Q变化；报告真实latent与预测latent的raw L2/MSE、Q投影L2/MSE、目标交叉项及残差恒等式误差。原pose/XY/wrapped-angle/official指标并列保留。imagined臂预测从自身滚动历史起始，误差包含累积context drift，不能混称与actual臂完全相同条件的一步误差。

v1历史字段保持`unconfirmed_not_saved`；v2新增`derived_real_RGB_native192_verified`，只表示后来对原真实RGB的派生编码已核验。P1/P4 admission、efficacy均false，不提供唯一根因、独立泛化或“第二批”证据。

## 运行

在`/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p4_feedback_attribution_audit_20261027_v2.py
.venv/bin/python -B -m experiments.spectral_reachability.p4_feedback_attribution_audit_20261027_v2 --output results/p4_feedback_attribution_audit_20261027_v2_main01
```

新结果目录独占创建，先封存v2源码/tests/本文与v1安全算术依赖，完成后复核封存和所有读取输入未变。小文件每份32MiB、累计128MiB；模型只流式hash（单文件1GiB cap），输出上限2MiB。失败另写新failure.json并抛错，不能将缺失/坏收据记作科学通过。测试为纯算术/显式mock；实际saved审计单独执行，均不是新神经或物理实验。
