# P4 saved 配对分析：独占持久化

此入口复用已审定的 `agent_p4_saved_pair_budget_20261029_v1`，固定验证其 source/tests/doc 三个 SHA，并将本持久化 source/tests/doc SHA 及原分析的五个实际输入 SHA 同时封存进结果。重算是原 saved 数据分析，不重跑 full strict，也不读未来数据、执行神经或物理。科学门与原分析均不变。

输出只允许 results 目录下本 stem 新文件，以 O_EXCL 创建，不覆盖；先验证全部输入和源码闭包再写，8MiB 输出上限、200MiB 剩余空间门槛，file fsync 后 directory fsync。写入失败可能保留 partial，不自动删除或重试；verify 要求指定实际文件 SHA、当前完整 source closure 和完整规范字节重算一致。SHA 绑定不是数字签名或新的历史认证。

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p4_saved_pair_budget_persist_20261029_v1.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p4_saved_pair_budget_persist_20261029_v1 write --output results/agent_p4_saved_pair_budget_persist_20261029_v1_MAIN01.json
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p4_saved_pair_budget_persist_20261029_v1 verify --result results/agent_p4_saved_pair_budget_persist_20261029_v1_MAIN01.json --sha256 ACTUAL_SHA
```

每臂完整计量/FLOPs/继承费用未知；同控制时刻的 I/R 是不同轨迹的描述性比较，不能称因果或 fresh。后续源文件变化将使已封存结果的 verify 拒绝，需保留本三件套原字节。
