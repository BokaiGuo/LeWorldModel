# P2 saved composition contract audit v1

只读保存数据的实现合同审计，不依赖P1科学准入，不运行模型、环境、CUDA或physics。不修改旧seal，不选择新root，不依据结果改动作/门槛。源文件位于 `experiments/spectral_reachability/p2_composition_contract_audit_20261027_v1.py`，对应测试为 `tests/test_p2_composition_contract_audit_20261027_v1.py`。

## 实际输入与验证

固定frontier planner `results/p2_composition_predict_frontier_cuda_20261027_v1_main01/report.json`，外部SHA `4023a0e3896a90407188d260a45dd334e0802ba17ecd59ef62ea931603eba173`；物理 `results/p2_composition_predict_frontier_physical_20261027_v1_main01/report.json`，外部SHA `9144c3f25f2880af62d94dd88257918fa508b5ddbea51d98e9ccbc3d3b7532e2`。

复用冻结80/40全槽exporter的纯数组重建：源seal、bank、program_id、goal/Q/context、全部见证与赢家重放收据。额外逐槽检查480候选的float32 tanh raw映射、full20 logit ball、拼接、条件suffix自己的prefix context、history真实归一化算术、原Q非线性评分和stable-first argmin；160个suffix独立KKT（最大scaled residual约7.04e-14）。replay检查是保存的whole/segmented上下文精确相等，**不是本审计重新神经重放**。

用 `(root_sha256, method)` join新重建计划、物理前prepare计划、8物理branch与8真实endpoint。逐步journal绑定实际raw commands/trace/RGB，拒绝借用旧终点。复核physical durable/final rows；完整公共调用账为32collector reset、8warmup reset、120warmup step、80selected step、8encoder/8projector。internal physics substeps仍未知，不把200公共step称为全部物理计算。

另重新执行旧paired40/40保存合同审计，4root/160suffix KKT；只复核实现，不与frontier合并科学样本。报告保留全部输入绑定，不拷贝约73MiB上游。

## 实际结论

当前未证明动作拼接、残差、坐标、裁剪、rolling history、Q或选取的可修实现错误；这不等于穷尽所有bug。root1的预测/实际Q/pose均损害，root2预测胜但实际Q/pose反转；root0预测胜、实际Q负但pose改善；root3预测负、实际Q正但pose损害。单位不同只能比较排序方向，不能将Q与pose差相加。末态信息不用于推断整轨迹接触。

公平性错误的明确边界：每root两臂各168 attributed predict，仅是predict frontier相等。conditional另有40个192×10 Jacobian、7680数学VJP方向、40SVD、40solve；joint则80候选而conditional40候选。共享4predict只记一次，全批1328predict/1328actionencoder、164J、31488VJP、168SVD、164solve。backward API、GPU FLOP/time/memory、CPU验证时间与内部physics等未知，严禁宣称equal total compute。资源接口会拒绝把168/168标为完整资源相等，不空跑padding。

静态失败路径风险：CUDA v1 `actual_entries` 的terminal journal `finally` 或hook恢复可能覆盖原异常；独立CUDA v2已有异常留存强化。本次成功v1资料没有显示此路径发生，不能据此解释控制负结果，也不追认v2为已执行有效性证据。

冻结检查证明保存的source/protocol/actions与actual journals一致，不能凭文件hash证明墙钟先后、没有隐匿试验或独立前瞻确认。保存Jacobian的真实性/预测模型保真仍需独立证据。0/8实际success不是理论否定，P1/P2均false。

## 命令、文件与输出

cwd `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_p2_composition_contract_audit_20261027_v1.py
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.spectral_reachability.p2_composition_contract_audit_20261027_v1 --output results/p2_composition_contract_audit_20261027_v1_audit02.json
```

实际首次报告 `results/p2_composition_contract_audit_20261027_v1_audit01.json`，约172KiB，SHA `3c5e04604440b0b870b9a94d60b6b06a58a81652dd6ed3b250aa09968c86dc95`；源码SHA `30612b4bde14d2d4396bb66adfc1e3038c13a0f5cb067982a4f91f750d574dbc`。CLI拒绝覆盖，单报告≤2MiB且写前保留200MiB。测试含saved-only真实数据和纯接口篡改，不能称神经或物理复现实验。

最小下一协议：先固定每轴资源cap与有效候选分配，补测未知成本，不以padding凑等；预声明独立derivative/fidelity核验与root-disjoint组合消融准入。仅当前合同检查不需要P1准入，新的科学效果声称仍需相关前置证据。
