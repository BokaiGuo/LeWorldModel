# 六基线资源矩阵：保存结果投影

本工具只消费已批准六基线 comparison v3，固定 SHA
`96cf27b143bb1a839edf7bc1e4de02f4a97e433c6475c4c731c78af76300d32e`。
不重新调用神经、物理或选动作，不报告科学胜负；原两根仍为校准。

输出每方法两根的逐项资源与合计，API、子模块、AD forward、scalar-loss backward、
output VJP、J/SVD分别列出，重叠工作不相加为forward。未单独计量保null，不冒充0。
物理/RGB多方法cohort只计整次，不按方法平均分账；原8次共享编码保留共享与standalone两种视图。
预测调用相同不意味着等总算力；15对方法均保留时域、搜索域、CPU/AD/延迟及编码成本的匹配缺口。

输入上限1MiB，输出JSON和Markdown合计256KiB，空闲保200MiB。
新目录排他创建、输入发布前复验SHA、输出文件和目录fsync；失败保partial，不覆盖旧结果。
使用真实已封存资产，不训练、不拟合Q、不调整成功阈值。RAM测试只是工程验证。

从Python项目根执行：

```bash
.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_moving_baseline_budget_matrix_20261027_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.p1_moving_baseline_budget_matrix_20261027_v1 --comparison results/p1_moving_baseline_comparison_20261027_v3_main01/comparison.json --comparison-sha 96cf27b143bb1a839edf7bc1e4de02f4a97e433c6475c4c731c78af76300d32e --output results/p1_moving_baseline_budget_matrix_20261027_v1_main01
```

worker实现因外部模型服务403余额不足中断；MAIN完整审读source/tests后接手补齐本文及目录持久化。
是否实际成功以对应命令和新产物为准，本文不预填执行结果。
