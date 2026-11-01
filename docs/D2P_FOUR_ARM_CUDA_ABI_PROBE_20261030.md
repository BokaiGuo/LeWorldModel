# 四臂共同后缀接口：冻结 checkpoint 的 CUDA 接线探针

使用既有 joint Vreach 准备文件中的第一个已知根与冻结 PushT checkpoint，在主机 `cuda:0` 比较共同后缀接口与直接调用 `action_encoder`、`predict`、`autograd.grad` 的两步更新。两条路径的动作输出最大绝对差为 0；运行前后模型哈希一致。报告 SHA-256 为 `333c994735b00dc0c27e7cc25a6d29498d0ee3153b34c14875529bbee64ed5d4`。

接口账本记录 4 次 action encoder、4 次 predict、2 次 backward；参考路径额外重复计算，未纳入该账本。因此这是接线探针，不是公平预算比较。诊断使用归一化动作坐标中的零初始化和盒约束，不证明其对应合法物理动作范围或完整真实任务性能。未执行新物理动作、未训练模型、没有独立新根，P1 和公平性均未通过。

产物：[报告](../le-wm-main/results/codex_four_arm_cuda_abi_probe_20261030_MAIN01/report.json)；[执行代码](../le-wm-main/experiments/spectral_reachability/codex_four_arm_cuda_abi_probe_20261030.py)。下一步需要把实际动作归一化、候选生成及筛选、最终 replay、各方法的独立后缀优化和真实反馈执行接入同一计费合同。
