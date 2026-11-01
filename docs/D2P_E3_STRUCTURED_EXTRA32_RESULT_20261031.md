# Structured 获胜候选额外优化：已知根探索结果

按 `D2P_E3_STRUCTURED_EXTRA32_PROTOCOL_20261031.md`，在原有四臂
真实 RGB 反馈实验中，structured 也对模型选出的候选执行额外
32 次优化；native、random、immediate-progress 的规则保持不变。
两个根分别以主机 CUDA 和物理执行完成；冻结模型未训练。

| 根 | structured 原结果 | structured 额外优化后 | 额外优化后其他臂 |
|---|---|---|---|
| 0 | 20 步未成功；物体位置误差 13.129 | 20 步未成功；物体位置误差 23.918 | native 失败；random 第 13 步成功；immediate-progress 第 20 步成功 |
| 1 | 第 8 步成功 | 第 10 步成功；物体位置误差 13.950 | native、random 失败；immediate-progress 第 20 步成功 |

位置误差不是完整任务成功指标。早停后无后续状态，不补值。
这是观察历史开发结果后提出的两个已知根探索对照，不是独立确认；
structured 的结构计算与额外优化均计费，仍未证明等预算。它不能
证明 P1 的非即时进展或能力机制，也不能排除其他实现问题。

两个报告分别通过对应保存态检查器：根 0 检查 15 个到达节点、
根 1 检查 14 个；核对目标绑定、执行动作、候选选择、额外优化
接受条件及模型调用和 Python autograd API 账本。保存态检查未
独立复演完整物理事件链，账本不等于完整计算成本。相关既有根绑定
测试 4 项通过。根 1 第一次启动与根 0 的非阻塞 GPU 锁冲突，在
进入执行前退出；根 0 完成后重新启动并正常完成。

- 根 0 report SHA256：`2c5ea970af448a0fb5242b5f805797bc4cd4e08731f1ae5ddd502532c8af7e8d`
- 根 1 report SHA256：`79fd8c5969acfd8fb17e181f33befe2c849fd5c4d02c6d9c2d62aa81c8de5953`
- 原新重选前缀共同后缀和各自规划结果见
  `D2P_P1_RESELECTED_DELIVERABLE_STATUS_20261031.md`；本实验
  并非该前缀的新能力干预。
