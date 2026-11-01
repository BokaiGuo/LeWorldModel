# P1 历史 reset 账本接收检查

此工具补充既有 quarantine、历史排重索引和十次 dispatch 恢复映射；不重建
账本，不替代其 seed 包络、pose 比较或原始证据核验。模板中的空 attempts
表示未提供记录，绝不是零次 reset。A–H 共十次，其中 B 三次；标识是本工具
的槽位标签，实际绑定须回查 recovery map 的原始 dispatch 与返回证据。

从 `le-wm-main` 运行（只输出到终端，正常阻断返回 2）：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_reset_ledger_intake_20261101_exclusive < experiments/spectral_reachability/p1_reset_ledger_unknown_template_20261101_exclusive.json
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_reset_ledger_intake_20261101_exclusive.py
```

后续收到来源时复制模板至新文件，保留十槽；填原账本的 SHA256、声明的 coverage、
每个 attempt 的来源内 JSON pointer 和有明确序列化合同的 state 字节 SHA。
不得只输入成功或选中项；失败、未选、重启也须按真实账本保留。工具拒绝额外字段，
报告相同 state 摘要及全部引用；摘要比较仅在双方序列化合同一致且原字节经核对时
有意义。它不打开账本、不核验 SHA 或 pointer、不将空列表认作零尝试，不处理
角度环绕、容差、snapshot/history 或跨环境物理身份。完整声明即使填满也只是
声明；退出码始终为 2，所有授权和认证字段固定关闭。

既有记录见 `D2P_RESET_LEDGER_RECOVERY_MAP_20261030.md`、
`D2P_HISTORICAL_EXCLUSION_INDEX_CODEX_20261029.md`、
`D2P_INDEPENDENT_FRESHNESS_CHECKLIST_CODEX_20261029.md` 和
`D2P_FUTURE_HOLDOUT_TRAINING_SPLIT_AUDIT_20261030.md`。checkpoint 权重身份
不建立 checkpoint 到实际训练 run、数据版本、成员或 normalizer 拟合的血缘；
训练外、开发外、历史物理排重和 source 授权是不同命题，目前均未认证。
还需原始穷尽 ledger 或运行闭包与有限包络、历史范围/失败/重启证明、
候选实际身份与账本的可信映射、训练来源证据及独立审阅。
