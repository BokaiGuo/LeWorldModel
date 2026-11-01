# D2P：四篇基线可引用原表值与不可横比条件

日期：2026-10-27。本归档只整理已有一手缓存与封存导出，无新下载、训练、神经或物理。
**“可直接引用”仅指能引用该论文特定版本/原表的reported claim，不表示可与本地结果
直接横比。** 不创建SOTA排名、配对推断、统一预算或P1–P5科学准入。

复用既有 `p2_reported_baseline_table_20261027_v3/export/`，不覆盖、不重新export。
其完整 **238 cells＝236个原文数值＋2个原文缺失**，不是238个独立实验或本地根。
新增离线consumer实际从pinned原始HTML/既有PDF文本重建全部238条，与sealed
`export/report.json`逐项相同；不只是从旧解释文档抄数字。下列内联表是明确范围的
引用便览，不声称穷尽全部ablation；全量sweep及控制行继续保留在原封存长表。

## 来源版本、页表与完整范围

PDF页为从1开始的物理页；PlanWAM无已认证PDF页，使用版本化HTML子表定位。

| 论文 | 版本/原表 | 页 | 完整选定cells与可复用表 |
| --- | --- | --- | --- |
| GRASP / Parallel Stochastic Gradient-Based Planning for World Models | [2602.00475v1 Tables1/3/4](https://arxiv.org/html/2602.00475v1#S4.T1) | 7/8/8 | 112；[完整四planner、环境与horizon表](/home/gbk/code/LeWorldModel/le-wm-main/results/p2_reported_baseline_table_20261027_v3/export/grasp_reported.md) |
| Hi-LeWM / Mind the Gap: Promises and Pitfalls of Hierarchical Planning in LeWorldModel | [2607.12547v2 Tables3/4](https://arxiv.org/html/2607.12547v2#A3.T4)，预算Table5 | 13/13；预算14 | 106＝Table4全部23配置×4列＋Table3作者复现flat LeWM的14cells；[完整sweep](/home/gbk/code/LeWorldModel/le-wm-main/results/p2_reported_baseline_table_20261027_v3/export/hi-lewm_reported.md) |
| LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels | [2603.19312v3 Table10](https://arxiv.org/html/2603.19312v3#A7.T10)，子表A7.T10.6 | 26 | 8＝4solvers×2models；[完整solver表](/home/gbk/code/LeWorldModel/le-wm-main/results/p2_reported_baseline_table_20261027_v3/export/lewm_reported.md) |
| PlanWAM: Planning-Shaped Future Representations for End-to-End Autonomous Driving | [2610.11382v1 Tables1/2/3](https://arxiv.org/html/2610.11382v1#S4.T3)，子表S4.T1.6/T2.1/T3.1/T3.3 | unknown | 12＝两NAVSIM主指标＋HUGSIM两指标×5列；[完整目标方法表](/home/gbk/code/LeWorldModel/le-wm-main/results/p2_reported_baseline_table_20261027_v3/export/planwam_reported.md) |

最终数据入口：[原sealed report](/home/gbk/code/LeWorldModel/le-wm-main/results/p2_reported_baseline_table_20261027_v3/export/report.json)。
每cell已有原始row/column、literal token、figure/subtable ID、HTML SHA、协议原句，
前三篇另有PDF SHA/页及该页文本SHA。不是引用早期 `*.parsed.json` 的合并嵌套表。

## GRASP：这是学习模型内成功，不是真实环境成功

[Table1 / S4.T1.3 / p7](https://arxiv.org/html/2602.00475v1#S4.T1.3)，Push-T，
open-loop，每setting500 trials。每格为 **success(%) / median successful time(s)**：

| 方法 | H40 | H50 | H60 | H70 | H80 |
| --- | --- | --- | --- | --- | --- |
| CEM | 61.4 / 35.3 | 30.2 / 96.2 | 7.2 / 83.1 | 7.8 / 156.1 | 2.8 / 132.2 |
| LatCo | 15.0 / 598.0 | 4.2 / 1114.7 | 2.0 / 231.5 | 0.0 / unknown | 0.0 / unknown |
| GD | 51.0 / 18.0 | 37.6 / 76.3 | 16.4 / 146.5 | 12.0 / 103.1 | 6.4 / 161.3 |
| GRASP | 59.0 / 8.5 | 43.4 / 15.2 | 26.2 / 49.1 | 16.0 / 79.9 | 10.4 / 58.9 |

两个unknown是原文时间破折号，不得填0或将失败排除后时间当全任务成本。
Tables3/4全部四方法×三任务×三H的72cells在完整表保留；不能只挑有利H。

原文关键限定（[S4.SS2.p2.1](https://arxiv.org/html/2602.00475v1#S4.SS2.p2.1)）：

> All reported metrics measure task success under the learned world model.

同文S4.SS1.p2.1明确按环境/horizon sweep超参后报告最好setting；不是固定同算力比较。
S4.SS2.p1.1说明DINO-wm使用5个动作预测5步；但表中H到primitive/model-step的准确
映射未独立确认，不擅自统一单位。Table4只写median completion times，分母unknown，
不能套Table1的successful-only定义。检查真实物理需要另封合同；内部restart-Adam
代理不等于官方GRASP的虚状态松弛、噪声与梯度处理。

## Hi-LeWM：保留配置，不把offset当horizon

[Table4 / A3.T4.3 / p13](https://arxiv.org/html/2607.12547v2#A3.T4.3)。下表固定
`(high_h, high_replan, low_h, low_r)=(2,5,2,1)`，展示两schedule与所有d，不取每格
最优配置。单位success %，±为Table4明确的**跨seed SD**，不是SEM/CI。

| schedule | d（环境步offset） | PushT Hi-LeWM | PushT Hi-LeWM-C | Cube Hi-LeWM | Cube Hi-LeWM-C |
| --- | --- | --- | --- | --- | --- |
| staged | 25 | 90.7±6.2 | 70.7±5.2 | 65.3±2.5 | 60.7±2.5 |
| staged | 50 | 42.0±2.8 | 64.0±3.3 | 46.7±2.5 | 48.0±0.0 |
| staged | 75 | 15.3±4.1 | 22.0±3.3 | 49.3±0.9 | 50.0±4.9 |
| online | 25 | 28.0±8.5 | 28.7±5.2 | 69.3±2.5 | 54.7±0.9 |
| online | 50 | 35.3±6.6 | 34.7±3.4 | 50.7±3.4 | 48.7±2.5 |
| online | 75 | 15.3±3.8 | 32.7±3.8 | 58.7±4.7 | 42.7±4.1 |

原环境PushT-v1、OGBCube-v0，当前/目标图像，expert同轨迹采样起终；A3.SS2.p4.1
为每配置50episodes，通常3seeds。Table3是**Hi-LeWM作者对flat LeWM的复现**，不是
LeWM原文的新独立重复。完整23配置的92cells全部保留在原长表，本文固定配置不称最佳。

[Table5 / A3.T5.3 / p14](https://arxiv.org/html/2607.12547v2#A3.T5.3)：

| d | 环境执行budget | 高层CEM samples×steps/top-k | 低层CEM samples×steps/top-k |
| --- | --- | --- | --- |
| 25 | 50 | 900×20 / 10 | 300×30 / 150 |
| 50 | 100 | 1500×40 / 10 | 900×30 / 150 |
| 75 | 150 | 1200×60 / 10 | 1200×30 / 150 |

S2.SS0.SSS0.Px2.p1.1明确低层LeWM冻结，但另外训练macro-action encoder和high-level
predictor；A3.SS6.p1.1的-C还需训练action stream建立empirical macro bank。
因此不是training-free，也不是仅与本地低层预测calls匹配即可；编码、bank构造、
高/低层优化、反馈和墙钟均须另核。d是目标offset，不是high_h或low_h。

## LeWM v3：Table10的±不擅自解释

[Table10 / A7.T10.6 / p26](https://arxiv.org/html/2603.19312v3#A7.T10.6)，Push-T SR(%)：

| solver（原文标签） | LeWM | PLDM |
| --- | --- | --- |
| CEM | 96.0±2.83 | 78.0±5.0 |
| SGD (β1) | 26±4.32 | 4.67±0.06 |
| RMSProp (β2) | 67.33±2.49 | 49.33±8.26 |
| Adam (β1,β2) | 84±7.12 | 80±3.27 |

**v1没有本次Table10，不能将这组数误引为v1。** Table10未确认±定义及seed数，保持
`reported_pm_unspecified`。不从其他表借SD/SEM或把±当统计显著性。

通用协议定位：A2.p4.1的CEM300samples×30steps、elite30；A4.p1.1的5-action block；
A4.SS0.SSS0.Px4.p1.1的H5blocks=25环境步、执行整段后replan；A6.SS1.p1.1的PushT
goal-offset25、执行budget50。**它们是通用协议，不证明每个Table10 cell的具体绑定。**
各梯度solver的迭代、学习率、停止和资源量unknown，不把CEM300×30套给Adam等。
checkpoint精确版本、episode/root身份、预处理和动作映射尚需绑定，不能拿本地固定
15-step suffix或P1校准分数直接比该整任务SR。

## PlanWAM：驾驶分数不是操纵成功率

原版本2610.11382v1，PDF页unknown，但下列HTML figure/subtable定位可引用：

| 环境 | 指标（0–100） | reported | 原表 |
| --- | --- | --- | --- |
| NAVSIM-v2 navtest | EPDMS | 90.9 | [Table1 / S4.T1.6](https://arxiv.org/html/2610.11382v1#S4.T1.6) |
| NAVSIM-v1 navtest | PDMS | 93.8 | [Table2 / S4.T2.1](https://arxiv.org/html/2610.11382v1#S4.T2.1) |

HUGSIM closed-loop，原表Table3（不是四项success %）：

| 指标/子表 | E | M | H | X | Avg. |
| --- | --- | --- | --- | --- | --- |
| RC / S4.T3.1 | 87.6 | 48.3 | 51.9 | 37.2 | 53.7 |
| HDScore / S4.T3.3 | 75.3 | 32.9 | 35.6 | 22.0 | 38.7 |

S4.SS1.SSS0.Px1.p1.1定位NAVSIM navtrain→navtest及不做HUGSIM-specific fine-tuning
的迁移。A3.SS4.p1.1/p3.2/p3.3明确436scenarios、各难度均值与按scenario数加权Avg；
**不能重新算(E+M+H+X)/4**，更不能映射为PushT SR。

S4.SS1.SSS0.Px2.p1.1说明4历史帧×4摄像头，4秒轨迹、64候选×8waypoints；冻结
DINOv2 ViT-S底座但训练rank32 LoRA及规划/评分模块。四相机信息访问、训练及scorer
不能免费；候选64不是完整推理预算。checkpoint、seeds与全pipeline成本unknown。

## 引用边界与main仍缺的前置

| 证据层 | 本轮状态 | 不能替代 |
| --- | --- | --- |
| Mock/单元测试 | 拒绝数值/来源错配、伪造physical/paired/全成本；仅软件合同 | 模型准确性或效果 |
| 实际离线核验 | 原缓存重建238cells逐项等于sealed export；20项新增tests通过 | 新神经或物理实验 |
| paper-reported | 原版本、页表、协议下的外部claim | 本地同root/checkpoint配对结果 |
| 科学准入 | 本轮无新准入 | P1同根非贪心/能力/任务收益合取或P2组合增量 |

所有方法仍需各自精确checkpoint/normalizer、合法资产、start/goal/query身份、观察/动作
权限、反馈/horizon/终止与完整成本。GRASP尤其区分模型内/物理，Hi-LeWM区分高层训练/
约束bank与staged/online，LeWM区分solver预算，PlanWAM区分驾驶指标/训练模块。
未知保持unknown，不把同名环境、相同数字预算或同一LeWM字样当全协议可比。

SAGE另见[新身份解析v2](/home/gbk/code/LeWorldModel/docs/p2_sage_primary_source_resolution_20261027_v2.md)：
题名/ID已确认，仓库与权重入口已定位，实际权重文件未认证；摘要数值不是本节已核实
表页值，因此不混入四论文238cells，不把身份解析完成当baseline跑通。

## 实现、实际小报告与命令

新增consumer：`le-wm-main/experiments/spectral_reachability/d2p_baseline_reported_values_20261027_v1.py`。
仅调用既有v3模块的离线 `build_report`，显式禁用其`urlopen`；不调用fetch/export。
真实重验完整来源和sealed report，还检查SAGE旧版本SHA、HTTP响应大小与reported边界。
新增tests：`le-wm-main/tests/test_d2p_baseline_reported_values_20261027_v1.py`。

cwd `/home/gbk/code/LeWorldModel/le-wm-main`：

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_d2p_baseline_reported_values_20261027_v1.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.d2p_baseline_reported_values_20261027_v1
```

首次合同测试15 passed in0.21s，补内联表/定位及请求账检查后最终20 passed in0.31s；
stdout小报告 `reported_source_archive_consistent`，
238selected /236numeric /2missing；network/neural/physics=0，local_reproduction/
scientific_admission/cross_paper_ranking=false。这是实际saved-data核验，mock测试只证明
拒绝路径。只读现存PDF/已提取文本，不生成/下载新PDF、图片、缓存或checkpoint。

封存身份：`export/report.json` SHA256
`52e687811e6d6f728ab3eda60836ea05eff11f36da2bb4af975d1dd19ef380f1`；
既有v3 parser源码SHA256
`b9b975be0f71ec0104a6fd88ca52bdc0b6a22791122e68948280e9b243ff4c40`。
各原始HTML/PDF/text SHA保存在原report的source字段；新增consumer先验源码/导出SHA
再重建，发现变化即拒绝。磁盘实测约374MiB可用；新增文件仅几十KiB，不触碰保留200MiB。
