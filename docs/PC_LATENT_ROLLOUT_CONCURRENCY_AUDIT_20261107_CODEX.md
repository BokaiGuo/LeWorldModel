# TwoRoom latent rollout：并发安全候选未通过梯度等价检查

**后续更新：**本报告记录的是最初重写候选的失败；差异原因与改用
LeWM 原生缓存接口后的结果见
[后续修复报告](PC_LATENT_ROLLOUT_NATIVE_CACHE_FIX_20261107_CODEX.md)。
本报告保留当时的负向结果，不作为当前代码状态。

## 问题和修改范围

原真实／想象反馈诊断通过临时替换共享模型的 `encode` 方法复用原生
rollout；并发调用时，其他请求可能观察到临时替换。该风险不意味着此前
单线程结果错误。新增独立的 H=1 latent rollout 候选，不修改模型方法、
不修改既有诊断和历史结果；它目前只是实验代码，**不得作为正式替代**。

## 检查

CPU 上小型确定性 JEPA 测试覆盖时域 1、2、5、8 的输出和动作梯度、
并行调用及非法形状，共 6 项通过。这只证明该测试模型上的一致性。

使用实际冻结 TwoRoom RGB checkpoint、已曝光 root 38 和 39 的图像，
各取 8 条合成归一化动作包，比较原生注入路径与独立候选；允许阈值
为输出和动作梯度最大绝对差均不大于 `1e-5`。此处合成动作包用于数值
诊断，不代表合法环境动作或规划分布。

| 模型时域 | 输出最大差 | 动作梯度最大差 | 严格检查 |
| --- | ---: | ---: | --- |
| 1 | 0 | 0 | 通过 |
| 3 | 1.61e-6 | 0.05913 | 未通过 |
| 5 | 1.29e-6 | 0.08004 | 未通过 |

每个输入还重复运行了原生注入路径，输出和梯度最大差均为 0；
加载前后模型指纹一致。尝试对齐张量拆分及输出视图后，同样未通过。
梯度对应整个预测轨迹平方和，不是目标代价梯度；尚未判明差异根因，
也未确认它对优化决策的影响。不能因为输出近似相同就声称梯度等价，
也不能据此认定原生模型或理论有误。

该检查没有执行物理环境、验证并发 GPU 行为或测量控制收益。
PC0–PC5 不改判；第一篇仍为 RGB-only。本次 `code-debugging` 技能用于
检查数值、张量形状和回归，结果触发停止集成而非放宽阈值。

[候选实现](../le-wm-main/experiments/pc_latent_rollout_safe_20261107_codex.py) ·
[测试](../le-wm-main/tests/test_pc_latent_rollout_safe_20261107_codex.py) ·
[GPU 审计](../le-wm-main/experiments/pc_latent_rollout_safe_gpu_audit_20261107_codex.py) ·
[第一次完整结果](../le-wm-main/results/pc_latent_rollout_safe_gpu_audit_20261107_codex.json) ·
[视图对齐后结果](../le-wm-main/results/pc_latent_rollout_safe_gpu_audit_v2_20261107_codex.json)。
