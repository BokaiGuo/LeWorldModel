# P4 三反馈：保存数据的同时间配对与资源

本独占分析器只读固定 SHA 的 MAIN02 report、prepare、完整 saved strict receipt、MAIN native audit 和 rolling seam receipt；不重跑这些验收、不执行模型或物理。旧 rolling consumer 认证编码及上下文接线，不提供本分析的四对同控制时刻状态误差表。已绑定 receipt 是原验证结论，本程序不是重新执行 full strict。

固定两 known roots × GN/D2P prefix × I/R，8 臂、4 对、24 编译槽。先核每对起点 state7/goal7 和 prefix5 一致；固定 step 0/5/10/15/20 上分别给 state−goal 全七维（原单位，角未包裹）、agent/object/joint XY 欧氏误差、wrapped-angle 绝对误差（弧度）及 velocity XY 差范数，以及 R−I 差。I/R 后续真实执行动作可不同，同时间不等于同状态或因果配对。原 native final metrics/success 原样保留，早停末端不与另臂不同时间冒充同时间对照；缺 checkpoint 保留 null，不以前值/终态填充。

实际全局 attempted/completed 资源向量直接来自 report，并与完整 saved strict receipt 一致；分别给事前 caps。各臂仅列实际 encoding 记录数与 strict 验证的 compile 记录数，**不是**独立计量的每臂总费用；FLOPs、device 时间、继承费用及每臂全资源未知为 null。不求和嵌套 API 充作总算力，不称 equal FLOPs 或等成本。无统计推断或伪造置信区间：两已知根不是独立确认样本，四对不等于四根。本分析不是预测误差重算、因果收益、freshness 或 P1/P4 admission。

CLI 从项目根运行，analyze 只写 stdout；若需保存，MAIN 应使用独占新路径且避免覆盖。verify 对外部提供的实际文件 SHA 和完整重算内容逐值比较，不调用 full strict：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p4_saved_pair_budget_20261029_v1.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p4_saved_pair_budget_20261029_v1 analyze
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p4_saved_pair_budget_20261029_v1 verify --analysis ACTUAL_FILE --sha256 ACTUAL_SHA
```

分析器将固定输入的字节 SHA、schema、receipt/report 关联与资源账核对；不重新认证源图像、每次模型调用或全原生日志，依赖所绑定 receipt。输出结果是探索性保存数据描述，分析口径在已见数据后实现，不伪称预注册。文件改变或关键 identity/census 不符直接失败，不丢行。
