# P1 新前缀完整面板的配对表

新增只读 CSV 导出器，输入必须是完整面板消费器生成的配对 receipt
及其 SHA-256。每行对应一个共同后缀、一个前缀角色；表格同时
保留根和任务身份、前缀动作及 RGB 哈希、前缀物理变化、共同
后缀终点与相对各自零后缀变化、两种 RGB 刷新规划的完整 receipt
和资源记录。早停的固定时域终点保持空值，不以末帧补齐。

同一 anchor 的规划结果在 42 个后缀行中重复，仅用于便于逐行
对照；对规划结果做统计时必须按 `anchor_index` 去重。重复臂、
共享物理根和已知根也不得视为独立样本。资源仍是原记录，不
因此认证为等成本或实际总成本。该工具不运行模型或物理，不
修改原合同，不作 P1 准入判断。

导出摘要中的 `anchor_records` 是记录数，并明确标记
`anchor_records_are_independent_samples=false`；不能把它解释为
独立任务数。

完整 H10 面板完成并按既有 handoff 验证、生成配对 receipt 后：

```bash
.venv/bin/python -B -m experiments.spectral_reachability.p1_uphill_pairing_csv_20261030 \
  --receipt /home/gbk/code/LeWorldModel/le-wm-main/results/p1_uphill_complete_pairing_20261030_RUN01.json \
  --receipt-sha256 ACTUAL_PAIRING_SHA256 \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/p1_uphill_pairing_20261030_RUN01.csv
```

在 `le-wm-main` 目录执行，哈希必须从实际封存 receipt 计算。
当前面板仍在运行，故还没有真实完整 CSV 或新增科学结论。
模拟面板测试仅验证删失保留、重复标注及身份异常拒绝。
