# 共同后缀 raw 动作坐标接线

先前共同后缀接口在输入坐标直接做统一 box 投影，没有规定如何把不同维度的模型归一化动作坐标映射回物理 `[-1,1]`；因此不能直接把它的边界当成环境动作边界。新接口以 raw 物理动作作为优化变量，对每个五步 packet 按来源任务的逐维 mean/std 归一化为 10 维模型 token，并在 raw 空间投影。输入历史仍是已归一化历史 token，不能再次归一化。

该已知任务的 mean 约为 `[-0.0087, 0.0068]`，std 约为 `[0.2019, 0.2002]`。如果把归一化坐标 `[-1,1]` 直接当作完整物理范围，会将实际 raw 搜索范围限制到大约各维 `mean ± std`，而不是 `[-1,1]`。这是新接口必须避免的尺度错误，不是对旧实验已发生该错误的断言。

对近期回溯候选实现的具体代码核查显示，该路径先对 logits 应用 tanh 得到 raw，再按 mean/std 生成模型 token；其另有固定的 logits 范数约束。上述新接口问题因此不能直接归因到这条回溯路径，也不能单凭代码检查证明其所有部署动作绑定正确。

解析替身回归 2 项通过：覆盖历史对齐、逐维归一化、反向更新、物理边界和非法 std 的 fail-stop；其中一个断言直接核对手算的逐维 raw 梯度更新值。随后使用已知根的实际冻结 PushT checkpoint，在主机 CUDA 上与直接参考实现进行两步优化，raw 输出最大差为 0，模型哈希前后一致。报告 SHA-256 为 `5e3ea95c6ab2b76429c19eddf5a49be1c24ad6903f5d14744e5de7c39fb3e41c`。

这是新接口的接线验证，不说明历史失败必然由此造成。尚未证明历史正常化的训练血缘；参考路径没有计入账本，未执行物理动作，尚无四臂共同预算、强基线或 P1 结果。

代码：[raw 优化器](../le-wm-main/experiments/spectral_reachability/codex_four_arm_raw_suffix_20261030_main.py)、[CUDA 验证](../le-wm-main/experiments/spectral_reachability/codex_four_arm_raw_cuda_probe_20261030_main.py)；[报告](../le-wm-main/results/codex_four_arm_raw_cuda_probe_20261030_main_MAIN01/report.json)。
