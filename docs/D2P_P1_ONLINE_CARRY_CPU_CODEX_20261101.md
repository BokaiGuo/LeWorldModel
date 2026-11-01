# P1 CPU 在线 carry 执行器

入口：`le-wm-main/experiments/spectral_reachability/p1_online_carry_cpu_codex_20261101.py`。
这是独立适配层，不修改原执行器，不代表物理实验结果。

在 `le-wm-main` 下运行 CPU 合成演示（输出文件必须不存在）：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python -m experiments.spectral_reachability.p1_online_carry_cpu_codex_20261101 --ledger /tmp/p1_demo.jsonl --steps 3
```

测试：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. pytest -q tests/test_p1_online_carry_cpu_codex_20261101.py tests/test_p1_carry_binding_20261101.py tests/test_p1_rgb_carry_replan_codex_20261101.py tests/test_e3_cp10_dls_feedback_mix_20261031.py
```

实际接入时实例化 `OnlineExecutor`：`observe(boundary)` 提供当前 RGB、
CPU 上的模型上下文及 `context_source`；`candidates(boundary, observation)`
提供 CPU float32 候选和来源；`issue(packet)` 发送动作并返回 JSON 可序列化回执。
`dls_feedback_candidates` 在边界 5、10、15 调用既有 DLS/随机混合器；
边界 0 保留原候选。它是可选初始化工具，不声称独立生成 DLS 候选。

每节点预留按剩余 episode 预算和节点上限同时约束，自动缩小新候选前缀；
carry 与新候选使用相同尾长评分。账本记录 RGB、输入上下文、候选、选择计划、
动作意图、回执、预算结算和失败。动作意图不等同于实际执行：回执缺失时执行器
停止，必须由调用方检查环境状态，不能直接重试。账本含原始观测，需调用方管理
存储与访问权限。

预算仅覆盖选择器重评分，不覆盖 RGB 编码、DLS/候选生成、环境交互或适配器内部
其他计算；不是全流程公平性或实际成本证明。上下文与候选来源由适配器声明，
账本保留其内容但不能证明 RGB 编码、历史动作或候选生成的真实性。账本为追加
JSONL，不提供崩溃恢复、原子持久化或并发写入保证。接入前应由调用方确保 CPU
模型及环境，以及单写者、唯一账本和异常后人工或外部系统核对。
