# H15 当前动作原生 AD：部分实现

新增 `p2_h15_gn_native_ad_20261029_v1.py`，继承现有 NativeTree 的 frozen 检查、模块入口观测及 ledger。在当前 logit 映射的 float32 raw 上执行三包原生前向，逐输出计算 192 个 VJP 和 raw-backward 回调；保存 raw Jacobian 和 float64 tanh 链式 logit Jacobian。该代码仅定义 backend 方法，尚未接入主机 loader 或 lifecycle。

`p2_h15_gn_direction_guard_20261029_v1.py` 固定方向为第一个坐标及单位全一向量，每次 Jacobian 验证调用四次原模型评分。`eps=0.001`，两端均投影进共同球；使用投影后实际差向量计算 `J Δz/(2 eps)`，与端点响应差比较，L2 容差 `0.003 + 0.05 * max(norm(actual),norm(expected))`。验证点不允许参与赢家选择。此数值门槛是预先指定的工程门槛、尚未在真实模型校准；可能 fail closed，不能根据运行结果事后放宽。因为动作映射先 cast float32，差分只能近似检验连续 raw 模型加 tanh 链式导数，不能证明对离散量化映射处处可微。

每根按前一方案的 17 次候选评分、4 次当前 Jacobian，另需 16 次差分探针评分及 2 次付费 whole/segmented 最终重放；若每次三包预测，理论为 117 predictor samples 和 768 VJP directions。此为计划数而非实测，真实 ledger 必须逐入口计数并检查预测模块内部调用、失败和 cap。不能宣称与 H15/CEM 总计算相等。

CPU 测试含实际 PyTorch autograd、192 VJP 及账本；合成模型测试不证明冻结真实模型 CUDA 兼容。缺口：固定 task 与 model/precision source 绑定、GN 循环和差分相接、完整账本预算、逐调用审计、两根生命周期、落盘 saved consumer、prepare 和运行授权均未实现；因此尚无可执行 HOST CUDA 命令，不能提前调度 GPU。既有文件未修改，未执行 GPU、物理或训练。
