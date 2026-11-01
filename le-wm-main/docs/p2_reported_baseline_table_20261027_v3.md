# 四方法一手表值核验与独占 v3 导出

本次新增一手文献核验，不再从 v1/v2 registry 投影缺失行。只写新的 v3 模块、测试、本文和
`results/p2_reported_baseline_table_20261027_v3/`。旧 sealed 文件和外层文档不编辑。
使用 academic-research-suite 的一手来源核验流程及 TDD；数值从官方版本化 HTML 的原始单元格解析，
仍调用既有 `d2p_reported_baselines.validate_record`，不另建权威 registry。

## 论文、版本及选择范围

页码均为从 1 起计的物理 PDF 页索引；HTML 细分表 ID 区分同一 figure 内的子表。
这里的条目数是指标单元格数，绝不是独立试验、种子或根数量。

| 方法/来源论文 | 一手原文版本 | 选定表与 PDF 页 | 导出条目 |
|---|---|---|---:|
| GRASP: Parallel Stochastic Gradient-Based Planning for World Models | [2602.00475v1](https://arxiv.org/html/2602.00475v1) | Table 1 / p7；Tables 3、4 / p8，保留 CEM、LatCo、GD、GRASP 全行 | 112 |
| Hi-LeWM: Mind the Gap: Promises and Pitfalls of Hierarchical Planning in LeWorldModel | [2607.12547v2](https://arxiv.org/html/2607.12547v2) | Tables 3、4 / p13；预算 Table 5 / p14 | 106 |
| LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels | [2603.19312v3](https://arxiv.org/html/2603.19312v3) | Table 10 / p26，保留四 solver × LeWM/PLDM 两模型 | 8 |
| PlanWAM: Planning-Shaped Future Representations for End-to-End Autonomous Driving | [2610.11382v1](https://arxiv.org/html/2610.11382v1) | Tables 1、2 的目标方法主指标；Table 3 两子表全部难度与 Avg；PDF 页 unknown | 12 |

共238条，其中236条已报告数值、2条原文破折号缺失。Hi-LeWM 的106条包括完整23配置×4列=92条，
加上该论文自己复现的 flat LeWM 14条；后者不是原 LeWM 论文新独立结果。
不取最佳配置替代 sweep，也不把相同试验的 SR 与时间当成两个独立样本。
这些是明确选定的主规划结果范围，不宣称穷尽全部论文 ablation 或全部其它方法。

## 已核实的新值与严格解释

GRASP Table 1 的 Push-T H40/50/60/70/80 成功率为59.0/43.4/26.2/16.0/10.4%，
对应8.5/15.2/49.1/79.9/58.9秒。正文 `S4.SS2.p2.1` 明确所有结果是在学习到的 world model 内评价，
不能写成真实环境物理成功率。该表时间只统计成功 trials；LatCo H70/H80 成功率0.0保留，时间破折号保持 unknown。
Table 4 只写 median completion times，没有单独明确分母；导出不借用 Table 1 的成功条件分母。
每 setting 500 trials；正文按环境/horizon sweep 后选最好超参，不是同总计算预算。
H 保留论文报告值，primitive/model-step 的精确单位映射未独立确认；5-action DINO-wm 模型协议另以原文 quote 保留。
restartAdam 代理不等于原 GRASP 的虚状态松弛/噪声/梯度合同。

Hi-LeWM Table 4 保留 staged 与 online、Hi-LeWM 与 Hi-LeWM-C、PushT-v1 与 OGBCube-v0 全部配置。
例如 d50 staged `(hh,hreplan,hl,rl)=(2,5,2,1)` 的 PushT 为42.0±2.8 / 64.0±3.3%；
d75 同配置 online 为15.3±3.8 / 32.7±3.8%。这些是配置示例，不是跨论文优势。
d是起点到未来目标的原始环境时间偏移，不是规划 horizon；每配置50 episodes，通常3 seeds，表明确±为跨 seed SD。
Table 5 三档高层 CEM 分别900×20、1500×40、1200×60，低层300×30、900×30、1200×30；
仍不能换算为全 pipeline cost。低层模型 frozen 但高层 macro encoder/predictor 需要训练；
constrained 还需要训练 action stream 构造的 empirical macro bank，不是 training-free。

LeWM **v1没有 Table 10，v3才是本次取值版本**。Table 10 LeWM 的 CEM/SGD/RMSProp/Adam
为96.0±2.83 / 26±4.32 / 67.33±2.49 / 84±7.12%；PLDM 同表完整保留。
该表±的统计定义未确认，保持 `reported_pm_unspecified`，不借用别表的 SD/variance/seed 数。
论文通用 CEM 300×30、elite30，PushT H5 action blocks、目标offset25、执行budget50、5 blocks后replan，
作为**通用协议**另列，不能证明每个 Table10 cell 的精确运行绑定。
梯度 solver 的 optimizer预算全部 unknown，绝不套用 CEM 数量。

PlanWAM 一手 HTML 重新核实 NAVSIM-v2 EPDMS90.9、NAVSIM-v1 PDMS93.8；HUGSIM
RC为87.6/48.3/51.9/37.2/53.7，HDScore为75.3/32.9/35.6/22.0/38.7（E/M/H/X/Avg）。
原 Avg 按场景数加权，不重算四难度均值。协议4历史帧×4摄像头、64候选×8 waypoints/4秒，
冻结 DINOv2 base 但训练 rank32 LoRA及规划模块。PDF超过8MiB cap，在读 body 前拒绝；
页码明确 unknown，仍有版本化 HTML figure/子表 ID。驾驶指标不能和 PushT 或本地校准分数混排。

## 来源、空间与输出

先经 arXiv 检索/摘要工具核验身份，再用只读 HTTPS 请求读取官方版本化 HTML/abstract；
单文本上限2MiB、单PDF上限8MiB、原始响应总上限24MiB，不获取模型权重、数据集、图片大包。
三份小PDF共9,737,498 bytes，另有九份HTML/摘要共约2.42MB；PlanWAM PDF请求失败也保留receipt。
没有GPU/physics/神经CPU/训练。PDF仅用本机 `pdftotext -layout` 提取文本，核对页/表定位。
失败请求不等于论文不存在，不通过绕过上限重试大下载。

`sources/*.html`/`*.pdf` 是原始权威响应。`sources/*.parsed.json` 是首次采集时的解析衍生物，
早期解析可能合并嵌套表，原样保留供溯源，**最终导出不读取它们**。
当前解析器去掉 MathML annotation，按最近 figure 和独立子表重新解析 pinned HTML。
PDF、文本衍生物及所用 HTML 都锁定 SHA；每值绑定原行/列、原始 token、版本/任务/指标/协议段落、
HTML SHA、PDF SHA/页及该页文本 SHA。未知 checkpoint revision、seed IDs、全流水线成本不补造。

最终独占 `export/`：

- `report.json`：完整238条，来源、协议 quotes、配置、missing 与比较边界。
- `reported_only.csv`：便于论文表格/后续人工核对的长表，不读本地结果。
- `grasp_reported.md`、`hi-lewm_reported.md`、`lewm_reported.md`、`planwam_reported.md`：分论文可读表。
- `comparison_readiness.json/md`：单独列出每方法的本地复现资产、训练、episode、预算与统计定义缺口。
- `source_manifest.json`、`SHA256SUMS`：全部新原文/receipt/衍生物/输出与新代码、测试、本文的内容封存。

不跨论文排名，不报告本地复现、配对推断、同算力或P2/P3准入。原文协议不完整会阻止直接比较，
但不阻止将明确出处的一手值作为 reported-only 文献表交付。

## 离线执行和复核

cwd `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_p2_reported_baseline_table_20261027_v3.py \
  tests/test_p2_reported_baseline_table_20261027_v2.py \
  tests/test_p2_reported_baseline_table_20261027_v1.py \
  tests/test_d2p_reported_baselines.py tests/test_d2p_baseline_protocol_v2.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m \
  experiments.spectral_reachability.p2_reported_baseline_table_20261027_v3 --export
sha256sum -c results/p2_reported_baseline_table_20261027_v3/export/SHA256SUMS
```

新增22 tests通过，联合155 tests通过。TDD先验证缺函数失败后实现；覆盖严格数值、缺失/零成功、
嵌套表不合并、版本、原文SHA污染、完整sweep、horizon/offset区分、梯度预算不借用、离线exclusive输出。
这些测试证明提取和来源合同，不是控制性能的新实验证据。固定 `export/` 已存在会拒绝覆盖；
封存后不要重跑采集或编辑源，后续补值须新增版本。
