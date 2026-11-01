# P3 独立全轨迹与 schedule/candidate 支持分解

本文件新增伴随消费者，不修改冻结producer v2或独立numeric v1。它只复用我们
独立实现的numeric reader/核算，不import或调用任何producer consumer计算函数。
不执行NN/物理、不调阈值、不选根、不训练；所有6臂/18节点及6对都保留。

绑定 MAIN 实际 report `3e02de2f957253830a9122e62926563ab6ff80b8eb3c61207d4d1c367bbe044f`，
prepare `f9802bc72edd7088240a97ce0f330bbab486cb6c1deaaa6fe1f28a707d8edd08`，
strict `aa5c34faabae5205dbeb7dd216fa5ff0631d16239eae25de06bf2b9a2084e94a`。
worker只读过schema元数据，没有运行此实际分析；MAIN亲自执行下列CLI。

先调用 standalone numeric.analyze 独立重计实际journals与native数值，再读取原
report 的全部node request/result：核全部candidate whole/segmented相等、每个原Q、
fraction/raw支持、stable(cost,index) winner及其字面动作。完整输出所有候选，不只winner。
每boundary保留context/raw10/goal identity、chi/signed_gain/scheduling_gain、decision、
pending commitment、实际predict attempted/completed。活跃H3中间付费观察不等于重新调度。

每个原root的全部三对都比较 warmup与control的**全部**phase/step：state7精确bits、
完整snapshot、goal7、goalpose、goalRGB/pixels、PNG SHA和原文件bytes；缺帧保留不同，
不因不相同删对。PNG bytes读取有原receipt SHA，图像解码/renderer正确性依赖原fullstrict，
本工具不是又一个renderer审计。

检查真实raw控制序列是否等于原prefix5 + 所有已封存完整commitment的已执行前缀，
保留完整承诺时机(start/end/seal)。配对同时给出实际control长度及predict/VJP付费。
4与8候选fraction通常不是嵌套格点；通过所有候选**实际rawbits**交集报告支持关系，
不只凭候选index或fraction猜赢家相同。模型分支task_id因mode不同本来可能不同；
比较的是实际context/goal/raw支持，而不是要求不同臂的task_id相等。

允许的有界结论：某个已见根的两策略实际选中动作/原生轨迹完全相同，但支付的
候选replay/predict工作不同。这既不证明调度bug，也不是等FLOPs、全局策略等价、
科学P3成立或fresh结果。任何实际相同/不同均待MAIN运行，不先写入预期结果。

```sh
.venv/bin/python -m pytest -q tests/test_agent_p3_commitment_numeric_recompute_20261029_v1.py tests/test_agent_p3_commitment_trajectory_recompute_20261029_v1.py
.venv/bin/python -m experiments.spectral_reachability.agent_p3_commitment_trajectory_recompute_20261029_v1 --output results/agent_p3_commitment_controls_20261029_v2_MAIN01 --report-sha 3e02de2f957253830a9122e62926563ab6ff80b8eb3c61207d4d1c367bbe044f --prepared results/agent_p3_commitment_controls_20261029_v2_prepare_MAIN01.json --prepare-sha f9802bc72edd7088240a97ce0f330bbab486cb6c1deaaa6fe1f28a707d8edd08 --strict-receipt results/agent_p3_commitment_controls_20261029_v2_consume_MAIN01.json --strict-sha aa5c34faabae5205dbeb7dd216fa5ff0631d16239eae25de06bf2b9a2084e94a --receipt results/agent_p3_commitment_trajectory_recompute_20261029_v1_MAIN01.json
```

执行工作目录 `/home/gbk/code/LeWorldModel/le-wm-main`。结果为独占新receipt；no overwrite。
预算仍notmatched，原科学门全部false，MAIN审读后运行，worker不运行生产核算。

最终轻量联合测试：**31 passed in 1.72s**（numeric22 + trajectory9）。numeric三件套
已保持冻结，其文档“实际SHA待提供”是接收到新MAIN消息前的实现时状态；本伴随文档
记录上述实际完整权威，尚未生成独立真实核算receipt。producer v2三SHA复核未变。
