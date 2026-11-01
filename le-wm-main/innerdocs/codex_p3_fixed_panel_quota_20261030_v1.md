# P3 固定面板配额探索接线

本文件和配套代码在任何本方案结果产生之前写入；不是外部预注册，也不是对旧结果的修订。已知根、已见历史结果，科学 gate 不变。

旧 H2/H3 allocator 将连续 gain 映射至整数，存在与 uniform 同为 5/5 的区间；已有后续 rank 对照的配额实际分开，且 cold 版本已有 `cold_saved_verified_not_admitted` 的保存验收，不能把它们说成尚未执行。最新 dual 主机运行六槽只有两个 signal 槽完成，其余失败、整体 incomplete；原产物和失败保持原样。dual 的 progress/R 是不同预算，不能跨分支宣称等工作。

本探索不同于上述方案：每臂固定独立评估同两组各 15 个 H2 候选，每候选两次 predict，再按 uniform 10/10、signal 15/5 或 5/15、reverse 反转的 eligibility 从 20 个候选选最低 cost；精确权重平局保持 10/10。所有 30 项均计算计量，只有 eligibility 不同，因此为测量代价而非最便宜部署基线；不保证赢家或动作不同。禁止按结果重新选择权重、候选、阈值或跳过候选。不同候选需绑定同一预定 bank、任务和模型；同候选不得跨臂缓存冒充独立调用。

当前实现只有候选回调接缝，30 次 callback 和自报的 60 predict 不是经核验的 CUDA kernel 计量。异常时保留已完成数，进行中的 predict/metric 标为 unknown；sink 本身若失败不能保证失败记录耐久。模型、输入、源身份、独占输出、实际调用事件、固定六槽/反馈及独立 saved consumer 均未接入，故不提供生产命令、不运行 CUDA 或物理。候选选择费用只与所述计数有关；FLOPs、墙钟、加载与全流水成本未知，不声称相等。

离线测试（不作神经推理）：

```
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_codex_p3_fixed_panel_quota_20261030_v1.py
```

后续真实运行前须实现并测试真实候选 replay/metric 及逐操作计量、固定来源与输出、完整失败 journal、独立 consumer，并在 CUDA 初始化前持有既有固定 inode 合作锁直至清理结束；合作锁不约束不使用它的进程。
