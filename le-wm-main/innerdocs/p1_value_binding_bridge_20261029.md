# P1 同候选价值桥接：离线身份合同

已有固定 40 槽证据将原 prefix/suffix、context、goal 和 terminal20 的物理返回严格接合；它不提供这 40 槽逐候选即时评分及条件 Vreach 的完整对应证据。联合 Vreach 的保存审计是另一条候选链，文档明确缺生产宿主、根来源闭合及物理释放；不得以槽号或相似动作将两链拼接。

本模块提供可执行的中间接合器，输入为四个由各自上游独立导出的 JSON receipt：`immediate`、`suffix`、`vreach`、`physical`。每个 receipt 都必须含相同 candidate_id、task_id、完整上下文和目标 SHA、prefix 后上下文 SHA、时间、原始 float32 prefix/suffix（各 5×2 little-endian hex）和 full-action SHA，以及本 receipt 的 source SHA、数值和数值时间。即时及 Vreach 绑定 prefix 终点；后缀评分及物理值绑定 terminal。原始位串比较保留 signed zero；Vreach 仅表示 prefix 条件下的可达价值，不声称它由执行的 suffix 决定。数值符号、单位、目标函数由上游原合同定义，此桥不合成分数、不改 gate。

运行（工作目录 `le-wm-main`）：

```bash
.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_value_binding_bridge_20261029.py
.venv/bin/python -B -m experiments.spectral_reachability.p1_value_binding_bridge_20261029 /absolute/path/to/receipts.json
```

接合器仅验证输入之间一致性，并不验证 receipt SHA 指向的文件、上游生成过程、模型或物理 authority；攻击者可以伪造全部字段。缺失的生产桥接仍需让各原 producer 基于已认证的实际事件独立导出上述 receipt，并以原 strict consumer 验证源文件和原始中间上下文、评分定义及物理返回；不得用一份共享的人工元数据填充四份 receipt。没有这层认证时，输出不能视为真实同候选证据，也不构成 P1 放行。此代码不启动模型、CUDA、训练或物理。
