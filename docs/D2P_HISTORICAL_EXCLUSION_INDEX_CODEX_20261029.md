# D2P 历史身份与条件隔离索引

日期 2026-10-29。只读原日志及已存在的原始结果，新增独立索引、脚本及测试；不运行历史命令、模型或物理，不根据结果挑根。本次没有找到足以认证原运行闭包或完整 reset ledger 的新证据。

## 实际复核

- 从已有 coverage 找到 Oct 23 原会话 JSONL 的固定前缀，重新计算其 508228945 字节 SHA `0a2c097548ef4bd9818affcf82ecb53102ddfa965498b966c13aa7067efdf70c`，并逐行核对 A–H 的 10 个调用 ID、时间戳和行号；记录各行 SHA。B 的三次调用均保留。这些是派发意图，不证明执行时加载字节或全部 reset。原有返回链对 B 前两次异常及 A/E/H exit 0 的恢复见 `agent_freshroots_resume_20261027_next.md`；退出或异常不等于完整尝试记录。
- 重核 80 份原 prefix JSON 的完整 SHA、`/prefix_state` 及 state7/pose hash，保留每一条来源路径和指针。80 条记录的 state SHA 和规范 pose key 各有 8 个不同值；这是精确键计数，不是按容差去重后的独立根计数。它们是历史 observed-prefix，不能直接标成原采样根或推断 reset seed。其他 snapshot、RGB3、raw10 及祖先关系的缺失照原记录保留。
- 重核 A–H 的 8 份原 report SHA 和 52 条 selected 字段 JSON 指针，保留原始声明；已有汇总的 44 种 root-goal hash 不是 44 个物理独立根。52 条不能覆盖失败、未选、重启或未保存尝试。
- 将已有 11 条闭区间条件 seed 包络单独列入索引，逐项保留假设和未知项。A/B/C 的 reserved 区间不是实际观测；对其他区间也不能称原执行已认证。seed 0 的显式 state 重建应另行核对 state，不是随机根身份；该信息见原 quarantine 资产，不能以本索引的区间替代。

上述整合是对已发现材料的直接字节和指针复核，不是新发现了 80 个样本，也不是完整历史排重认证。原结果存在 outcome 字段时仅提取上述预先列定的身份指针，不利用 outcome 选择记录或改变分母。

## 产物与复现

索引 `le-wm-main/results/codex_d2p_historical_exclusion_index_20261029_LOCAL01.json` SHA256 `ac2544512d033ec14fd07808c3abe2dfd96676bb6b82ec89569f498f58256f98`。生成器固定四份输入 SHA、原日志有界前缀、原文件 SHA，路径/指针不匹配即失败；输出采用 exclusive create，不覆盖旧文件。调用使用其他新输出名：

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.codex_d2p_historical_exclusion_index_20261029 --output results/codex_d2p_historical_exclusion_index_20261029_LOCAL02.json
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_codex_d2p_historical_exclusion_index_20261029.py
```

实际结果：88 份原 JSON 验证，80 prefix、52 selected、11 条条件区间、10 条原始 dispatch；3 tests passed。数据只是已知排重索引，不是授权消费的完整 manifest。脚本不自动合并近似 pose、不自动判定新候选，也不把缺失当不重复。

## 仍需原始证据

逐调用原 reset ledger，或原进程绑定的 producer、四层 factory、安装环境及 wrapper/transitive 闭包和可证明穷尽的尝试上界；B 三次均须覆盖。还需包含其他机器/工作树、未保存失败及重启的历史范围认证，以及 selected roots 与 state7/snapshot/history 的原始映射。缺失这些时不能清除 16 条 inherited reset blockers，也不能宣称 global freshness 或准入 P1。若原记录不可恢复，应明确记载不可恢复，不能依据未找到记录推断 fresh。
