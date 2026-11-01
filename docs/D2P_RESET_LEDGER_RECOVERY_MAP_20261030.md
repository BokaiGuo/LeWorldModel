# 历史 reset ledger：十次调用证据映射

## 结果和边界

本次固定合并原历史索引的十条 dispatch 身份、已有 delta 的十三条进程返回收据、
原 coverage 的八个来源报告身份；重新校验三份文件的完整 SHA、delta 解压后的完整
semantic SHA、原会话 508,228,945 字节固定前缀及 23 条相关 dispatch／返回行 SHA。
读取原日志前缀只作字节哈希，不解析行中的 outcome；delta 只解析
`dispatch_evidence.dispatch_receipts`，coverage 只投影 `sources`，历史索引只投影
`original_dispatch_log`。没有打开各 report、P4 保留槽、候选、模型或环境。

|来源|调用数|已保存进程证据|reset ledger 结论|
|---|---:|---|---|
|A|1|异步 session 37004；轮询 exit 0|未建立完整 ledger 或运行时闭包|
|B|3|前两次各有 traceback；第三次 stdout，未保存内部 exit code|三次均缺失败前及其他尝试的完整证据|
|C|1|stdout，未保存内部 exit code|同上|
|D|1|stdout，未保存内部 exit code|同上，另需原 P0 依赖|
|E|1|异步 session 29686；轮询 exit 0|同上，另需原 P0 依赖|
|F|1|stdout，未保存内部 exit code|同上，另需原 P0 依赖|
|G|1|stdout，未保存内部 exit code|同上，另需原 P0 依赖|
|H|1|异步 session 59270；轮询 exit 0|同上，另需原 P0 依赖|

每条记录在机器结果中含 dispatch 原行号／SHA、返回行号／SHA、原 operation index、
轮询 session、已有 delta 内的 semantic JSON pointer、错误类型及允许的 producer
traceback 帧、原 report 家族路径／SHA 和 coverage pointer。原 report 的家族关联
不证明该 report 在 B 前两次失败时已经存在。进程退出或 traceback 不证明此前
reset 次数为零，也不能证明失败／未选／重启已穷尽。

**会话号更正：**旧 `agent_freshroots_resume_20261027_next.md` 把 A/H 的会话号
记为 71462/42594；固定 delta 按原工具返回的 operation index 分别关联 37004/59270，
轮询收据也分别引用这两个号。原返回行包含多个操作和多个 session，71462/42594
确实也出现在同一行，但不能按行内第一个 session 推断属于该调用。E 为 29686。
本次重核相关行原始字节 SHA；session/operation 语义仍继承既有 delta 的解析结果，
并未独立重新解析原行输出，也没有改写旧说明。

对每次调用仍需两条可替代路线之一：原来源真实、穷尽的 reset-attempt ledger；
或绑定实际运行的 producer、factory、installed environment、wrapper 等加载闭包和
可证明穷尽的有限包络。B 每次还需自己的 shell 入口版本；D–H 还需原 P0 依赖。
两条路线都还需要失败、未选、重建、replay、外部重启及机器／工作树／时间范围
证据和独立来源审核。已有材料不足以判定原记录永久不可恢复。

## 产物与复现

- 最终机器映射：`le-wm-main/results/codex_reset_ledger_recovery_map_20261030_LOCAL04.json`，
  SHA256 `d101bc2cad391b004998324ba31f8cc8bca3a02ef10c74a6b5f518851dce0582`。
- 脚本：`le-wm-main/experiments/spectral_reachability/codex_reset_ledger_recovery_map_20261030.py`。
- 测试：`le-wm-main/tests/test_codex_reset_ledger_recovery_map_20261030.py`，11 passed；
  覆盖十调用分母、B 三次、会话错配、错误来源／缺失／孤儿收据、delta 篡改、
  原行 SHA 篡改和非目标字段不投影。
- LOCAL01–03 为本次迭代快照，保留不覆盖；LOCAL04 增加 coverage pointer，是最终版。

在 `le-wm-main` 中以新结果名复现：

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -m experiments.spectral_reachability.codex_reset_ledger_recovery_map_20261030 --output results/codex_reset_ledger_recovery_map_20261030_LOCAL05.json
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_codex_reset_ledger_recovery_map_20261030.py
```

映射汇总已有证据，不是原始 ledger 或新的历史权威；所有完整性、freshness、source
授权及 P1 准入字段固定为 false，不选择或补充任何根。
