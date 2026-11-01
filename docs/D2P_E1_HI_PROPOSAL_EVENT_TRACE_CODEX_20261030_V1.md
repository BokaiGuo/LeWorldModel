# E1 proposal／历史共享费用：Hi 行级日志追踪

结论：Hi report 中存在四个 root/condition 各自的原始 hook event 列表，可以从中
重算行级观察计数，并与原四槽 receipt 汇总逐模块对账；但不能由此宣称任何 E1 臂
的端到端费用完整或等预算。原 arbitrary/compiled proposal 的完整祖先与共享成本
本轮未闭合，继续 unknown。

本次只读固定 E1 six-target prepare、Hi report、Hi consume receipt：
SHA 分别为 `cd96f6e28b25d3b7e3b6649c760b2f111747aec5c0ce9c8a41741318209df8a0`、
`083d0d9aa190b561e6ccf865b09d7224b97d404b16ba38c792bd8af386f5622a`、
`40f28414682c9540e9e195b361a36dfc514f816358adbeacb379fedc460e4ef0`。
report/receipt 绑定同一 manifest；两条 off 行的 root、blind、FIRST data SHA 与
E1 learned target 准备记录一致。本次没有跟随文件中的路径，也未访问 P1 文件、
NPY、模型或硬件。

四行各含 85 个已返回事件。工具核对行序、事件索引、parent 指向、returned 和
输入 shape，按原 producer 的口径重算各模块 attempted、returned、input rows 与
input context tokens；四行逐模块求和等于 report 和 receipt 的原整体账。两条 off
行**分别**从各自的 85 事件重算得 encoder 2、high_cost 3、high_predictor 14
（98 input rows）、low_predictor 2，均已返回；不是将总数除四所得。

每行每模块 event-set 摘要为 `codex.e1.hi_row_events.v1` 对象的 canonical
JSON（排序键、紧凑格式、末尾换行）SHA256；对象包含 report SHA、row index、
module kind 和原 report 中该行按序的完整事件。摘要为本次派生，不是上游签名。
报告中同时保留四槽原始汇总，未自动分摊共享费用。

**边界：** 行级执行含有不同阶段及嵌套模块，不能自动断言每项是 FIRST 所需的
最小或完整因果成本，不能将嵌套 hook 相加或称为 FLOPs。report 内事件是当时保存
的观察记录，并非本工具验证的独立写前日志；本工具未核对 array bytes、完整事件
语义、外部时序或上游 fullstrict，也不能证明日志外的遗漏不存在。上游已保存 receipt
的可信性作为边界继承。历史共同 setup、数据／训练、此前搜索、arbitrary 和 compiled
proposal 等未分配开销仍未闭合；不把未知补零，也不将逐行投影写入旧 ledger。

独占结果文件：
`le-wm-main/results/codex_e1_hi_proposal_event_trace_20261030_v1_LOCAL01.json`，
SHA256 `b4382de707531a59e36bc287f8932656f2cb3186cf552e13e96d5f0f68d1d349`。
工具固定采用排他创建，已有文件时失败不覆盖。

```sh
cd /home/gbk/code/LeWorldModel/le-wm-main
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_codex_e1_hi_proposal_event_trace_20261030_v1.py tests/test_codex_e1_event_ledger_consumer_20261030_v1.py
```

测试覆盖真实来源、逐轴总账、目标身份、缺失／篡改事件、错误 parent/shape、
错误 SHA 及排他写入；不会启动生产或改写已有资产。
