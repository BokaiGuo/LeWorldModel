# TwoRoom 固定池动作邻居稳定性：探索结果

按[协议](PC_TWOROOM_NEIGHBOR_STABILITY_SCREEN_PROTOCOL_20261107_CODEX.md)，
先使用已封存的模型评分选链，再读取同一固定池的真实执行结果。
三个规则均可使用每根全部 64 条 B 链评分。

| 选择规则 | 25 步内曾成功 | 相对原最低分独胜 / 独负 |
| --- | ---: | ---: |
| 模型原始最低分 | 8/10 | — |
| 同前缀两近邻均值 | 7/10 | 1 / 2 |
| 同前缀两随机邻居均值 | 7/10 | — |

近邻法在全部 10 根选中了不同链；全池有成功链为 9/10 根。近邻法选中的
7 条成功链中有 4 条在前 10 步已成功；它们不能计为利用前缀创造机会的证据。
因此该规则**未达预定探索出口，停止扩大此具体构造**。其唯一独胜
root 0 上随机邻居也成功，不能归因于动作邻近信息。

近邻平均距离为 1.392，随机邻居平均距离为 1.892（六维动作块欧氏距离）。
由于 B 是独立均匀采样、而非小扰动，这项结果不能证明真正局部扰动下
的鲁棒性无效，也没有测试描述保留或闭环收益。所有任务和物理结果
早已曝光；64 条 B 的查询成本必须全部计入，不等价于现有低预算规划。
本次没有新增 GPU 推理或物理执行，不能作为独立确认或改判 gate。

相关定向测试 2 项通过；另外独立按保存的邻居索引重算 640 个近邻均值，
最大差为 0。`experiment-design` 用于固定比较组、配对指标和停止规则；
该出口只适用于本次探索，不是 PC gate 阈值。

[封存选择](../le-wm-main/results/pc_tworoom_neighbor_stability_screen_20261107_codex_plan.json) ·
[结果](../le-wm-main/results/pc_tworoom_neighbor_stability_screen_20261107_codex_result.json) ·
[实现](../le-wm-main/experiments/pc_tworoom_neighbor_stability_screen_20261107_codex.py) ·
[测试](../le-wm-main/tests/test_pc_tworoom_neighbor_stability_screen_20261107_codex.py)。
