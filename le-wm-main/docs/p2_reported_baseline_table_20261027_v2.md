# 四方法 reported-only 表：独占 v2 视图

本轮只新增四方法视图、测试、本文及独占结果；不编辑已封存v1、registry或外层文档。
没有新增原文核验或实验，不联网、下载、运行GPU/physics/神经CPU。
使用 experiment-code 的证据分层与测试驱动技能，以缺模块失败回归开始。

唯一数值来源为已封存v1 report，SHA256：
`f6a0d4f13155a252b85acf413fa7c79d9357b608e14d883426eae2fbe80284c8`。
复用已有 `d2p_reported_baselines.validate_record`，不建立第二个数值registry。
源JSON上限2MB、锁定schema与SHA，拒绝重复键、非有限数、重复条目、漏掉PlanWAM列、
将本地结果冒充paper_reported或给未核实方法偷偷追加数值。
导出时复核两份原核验文档的SHA、行号和原行，并对所有输入二次读取检查稳定性。

## 表的实际内容

四行summary依次为GRASP、Hi-LeWM、LeWM、PlanWAM。前三项没有已核验性能数值，
`value=unknown`、`status=missing_verified_result`，数值条目数0绝不代表性能0。
PlanWAM保留12条原记录，包括NAVSIM-v2 EPDMS 90.9、NAVSIM-v1 PDMS 93.8，
HUGSIM的RC/HD-Score各四难度及原文Avg.；不重算平均，不将0–100指标改称本地success。
未纳入HWM、AW-0、AVIC等其它方法，既有v1中的那些记录原封不动。

LeWM的论文身份v1确认与Table10数值/协议未核验是两个状态：
summary中的result version保持unknown，身份原文另存`identity_source_anchors`。
GRASP-style restart Adam不等于官方GRASP，Hi-LeWM身份与原表仍缺失。
不从附件或模型记忆补成功率、页码、误差条、checkpoint或预算。

`table.md`用于直接阅读；`method_summary.csv`保留四方法缺项与前置要求；
`reported_only.json/csv`仅含已核验PlanWAM原记录，完整保留版本/表号/任务/单位/
checkpoint/观察/动作/预算/统计量/source；`report.json`同时包含原始协议细节和封存信息。
PDF页码、checkpoint revision、seed、不确定性缺失仍为unknown。64候选不等于64次predict，
4秒trajectory不等于4个LeWM预测token。任何表都不读取或拼入本地校准得分，
不计算排名、相对增益、配对推断、等总算力或P2准入。

## 执行

cwd `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_p2_reported_baseline_table_20261027_v2.py \
  tests/test_p2_reported_baseline_table_20261027_v1.py tests/test_d2p_reported_baselines.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m experiments.spectral_reachability.p2_reported_baseline_table_20261027_v2 --write
cd results/p2_reported_baseline_table_20261027_v2
sha256sum -c SHA256SUMS
```

固定新目录已存在则拒绝覆盖；退出0只表示视图导出成功，不表示新证据或研究准入。
后续补三方法表值仍需取得并核验一手原文、版本、表页与完整协议；本任务不授权联网。

验证：新增12项通过；连同既有v1、registry、baseline protocol v2共133 passed（0.47s）。
这是离线投影与防错回归，不是新文献核验或控制性能证据。
