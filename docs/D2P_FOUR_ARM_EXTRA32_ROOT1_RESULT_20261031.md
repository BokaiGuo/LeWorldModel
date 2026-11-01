# 第二已知根：赢家追加 32 次更新的真实反馈结果

按 `D2P_FOUR_ARM_EXTRA32_ROOT1_PROTOCOL_20261031.md` 在主机 CUDA 与
真实 PushT 物理环境完成四臂反馈。全部臂执行 20 步，原生成功均为
false，无记录错误。每节点数字为实际 Python API／module hook
计数，时间为四节点同步墙钟之和，包含审计开销。

| 臂 | 物体位置误差 | 物体角度误差 | 每节点预测调用 | 每节点 `autograd.grad` | 四节点时间（秒） |
| --- | ---: | ---: | ---: | ---: | ---: |
| structured | 56.3030 | 0.10359 | 75 | 208 | 6.893 |
| native-style | 51.7106 | 0.24541 | 171 | 48 | 5.042 |
| random | 37.1668 | 1.14709 | 171 | 48 | 5.798 |
| immediate-progress | 30.3113 | 0.98511 | 88 | 16 | 3.577 |

模型参数／buffer 前后哈希一致。16 个节点的 predictor 与 action
encoder hook、Python autograd API 计数均与节点账本相符；保存态
消费器复核节点账本、模型赢家追加动作、模型分数和实际派发的
前五步动作。独立计数只覆盖被拦截的 Python API，不证明所有可能
的自动微分入口均被覆盖；消费器没有独立复演整个物理事件链。

相对同根旧合同的 native-style 物体位置误差 77.2534，这次追加
优化后的误差为 51.7106；random 从 35.2834 变为 37.1668。
各臂的角度、联合状态和成功判据仍需分别报告，不能把单一位置
变化当成成功率改进或统计证据。新增优化使用模型分数而非真实
outcome 选择动作。

这只是事后已知根上的新开发合同；四臂成本构成不同，前缀及
完整流程成本没有认证，逐轴上限不等于相同 FLOPs 或公平预算。
native-style 不是官方强基线。该结果不证明独立泛化、P1 机制、
公平胜出或理论失败。

- 准备 SHA-256：`4f8468316ccb49d12459745dbea4c9cf22fd4b24f655cf06f5f094e74059d628`
- 运行报告：`le-wm-main/results/p1_four_arm_extra32_root1_20261031_RUN01/report.json`，
  SHA-256 `3e53dfd8a73298df7e08717836d6b30c0baf602239c5d6f661d7910a44b55df2`
- 保存检查：`le-wm-main/results/p1_four_arm_extra32_root1_saved_20261031_RUN01.json`，
  SHA-256 `ab79e79e09ed820a8a3f9b9cd0683add029980ecb1d5b36c465271c30c5c6295`
- 新合同的预算差异和 autograd 计数器恢复／不匹配拒绝测试：4 项通过。
