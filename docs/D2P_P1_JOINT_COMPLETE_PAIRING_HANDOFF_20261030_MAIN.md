# P1 完整共同后缀与独立规划的配对分析

`le-wm-main/experiments/spectral_reachability/p1_joint_complete_pairing_20261030.py`
提供完整面板验收后的分析入口。它先直接调用 H10 全量消费者，
重新验收 24 shard 和 1008 分支，再按固定准备文件的根、条件、
臂、角色、任务、前缀动作、cp5 像素和共同后缀动作哈希配对
`p1_joint_anchor_table_20261030_RUN01.json`。

输出包含 24 条独立规划观测和 336 条共同后缀配对，每条后者
关联三个前缀的索引，不将相同独立规划结果重复计作 42 个样本。
它分别保存共同后缀绝对终点、相对同前缀零后缀的变化、
相对 zero/greedy 前缀的终点差，以及相对 zero 前缀的响应差；
有一侧终点缺失时差值为 null。根和前缀的物体、角度、推杆、
联合 XY 与原生成功指标来自 24 行前缀表。状态和返回步数各自保留。

测试 `test_p1_joint_complete_pairing_20261030.py` 和
`test_p1_joint_anchor_table_20261030.py` 共 4 项通过。真实固定
24 个根和前缀的两套指标实现交叉核对均为 24/24；这些只是接线
测试，不是新的物理结果或 P1 通过证据。

运行前确认物理作业已结束，`run_report.json` 完整且状态为
`recorded_not_consumed`，再独立计算其 SHA-256。命令从 `le-wm-main`
目录执行，替换报告 SHA 为实际值：

```bash
.venv/bin/python -B -m experiments.spectral_reachability.p1_joint_complete_pairing_20261030 \
  --directory /home/gbk/code/LeWorldModel/le-wm-main/results/p1_joint_h10_20261030_run_RUN01 \
  --report-sha256 ACTUAL_REPORT_SHA256 \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/p1_joint_complete_pairing_20261030_RUN01.json
```

该入口仅接受完整面板，不在当前运行期间消费部分 shard。结果仍
是已知根开发分析：不能将共同动作效果当作可规划能力，不能将
同一后缀优化预算称为包括前缀生成在内的全流程等预算，也不能
依此开放 P1 或宣称独立泛化。
