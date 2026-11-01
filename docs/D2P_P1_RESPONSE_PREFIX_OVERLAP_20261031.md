# P1 结构化候选与新前缀面板：动作重合核对

对已封存的模型候选和新重选前缀配对表，按相同 root slot、条件和
arm 比较五步 float32 原始动作字节的 SHA-256。54 个臂内候选中，
14 条与对应的 selected、zero 或 greedy anchor 至少一个动作相同；
其余 40 条未匹配这个特定的 24-anchor 面板。候选可能跨臂重复，
不能把 40 条解释成 40 个独立动作或断言它们未在其他实验中执行。

本核对不读取成功、任务误差或其他物理 outcome 来选候选；动作相同
也不自动证明目标、观察、后缀规划及预算合同完全相同。结构化与
即时筛选此前在八臂均选择同一个模型赢家，不能将其余候选强行
称为该方法选出的赢家。

代码 `le-wm-main/experiments/spectral_reachability/p1_response_prefix_overlap_20261031.py`；
结果 `le-wm-main/results/p1_response_prefix_overlap_20261031_RUN01.json`，
SHA-256 `b3584f6150c214d3e311917559105c0ba25afcb0cd10bca61c37748f08633837`。
三项测试核查固定分母、无 outcome 字段、损坏动作及缺失 anchor
拒绝。该结果只是保存数据审计，不是新物理执行、独立确认或 P1 通过。

对新前缀已有的完整 H10 和真实 RGB 多起点规划，见
`D2P_P1_UPHILL_H10_COMPLETE_RESULT_20261031_MAIN.md`。其 24-anchor
汇总从固定输入重算得到相同 SHA-256
`98fdc09052d0afc83bf78815af19e98c869cb048ba9291e600167b2b0d452cde`；
保存链测试通过，但全链等预算和独立根仍未认证。
