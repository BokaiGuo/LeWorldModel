# D2P 作者报告数值交接清单

日期：2026-10-27。性质：文献 evidence map / reported-only 引用交接，不是本地基线复现、代码审计或科学准入报告。

## 1. 范围、核验状态与使用规则

问题：现有 PlanWAM、Hi-LeWM、LeWM、GRASP 一手资产中，哪些作者报告数值可以带版本、页表和协议直接引用？仅收录下列四个固定版本，不宣称最新版本、系统综述或穷尽全部消融。使用 literature-review 技能区分原文证据、可适用范围和未解决项；不作新颖性或理论失败判断。

本次实际读取本地原始 HTML 的目标表和协议段落，并将 GRASP、Hi-LeWM、LeWM 的目标页直接从既有 PDF 提取到 stdout 交叉核对。未新增下载、网络检索、模型、物理或实验运行。**以下数值均为已核原文的作者 reported，不是 moving 两根的本地配对结果。** 原文表中的破折号记 missing，不能填零；未知预算、误差条含义、checkpoint 不靠常识补齐。

可引用来源与版本如下。PDF 页码为从1开始的物理页，本次读取页脚与页表对应；PlanWAM 页码 missing，但 HTML 原文已核，不是整篇无法访问。

| 方法及论文题名 | 固定版本 | 已核原表 / 页 | 数值核原文状态 |
| --- | --- | --- | --- |
| GRASP — Parallel Stochastic Gradient-Based Planning for World Models | [2602.00475v1](https://arxiv.org/html/2602.00475v1) | Table 1 / p7；Tables 3、4 / p8 | HTML + PDF 原表已核 |
| Hi-LeWM — Mind the Gap: Promises and Pitfalls of Hierarchical Planning in LeWorldModel | [2607.12547v2](https://arxiv.org/html/2607.12547v2) | Tables 3、4 / p13；预算 Table 5 / p14 | HTML + PDF 原表已核 |
| LeWM — LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels | [2603.19312v3](https://arxiv.org/html/2603.19312v3) | Table 10 / p26 | HTML + PDF 原表已核；不可标为 v1 的 Table 10 |
| PlanWAM — PlanWAM: Planning-Shaped Future Representations for End-to-End Autonomous Driving | [2610.11382v1](https://arxiv.org/html/2610.11382v1) | Tables 1、2、3；PDF 页 missing | HTML 原表已核；不伪造 PDF 页码 |

## 2. 本地一手证据入口与 SHA

所有路径相对于 `/home/gbk/code/LeWorldModel/le-wm-main/results/p2_reported_baseline_table_20261027_v3/`。以下 HTML SHA 已在本次重新计算。保留现有封存资产，未重新抓取或覆盖 export。

| 原文资产（sources/ 下） | SHA256 |
| --- | --- |
| grasp_html.html | `2a17a21d05d1792b0fa7b7bd9442be973af6ab37236263994a59a4eb4e6fa899` |
| hilewm_html.html | `c4c7e725d862017334cbfd72c8e3e6dcdadb1ebff4aae45b3af847ba1e76bc7b` |
| lewm_v3_html.html | `49feca90079f39674b3fe7f23de7cb6e76562c4d4ff5efd0a96743ca668e70b3` |
| planwam_html.html | `67be22fee283aa4bad78578c91d83e6a3085bc8927c63597ee7c9721073bd873` |

前三篇 PDF 分别为 `sources/grasp_pdf.pdf`、`sources/hilewm_pdf.pdf`、`sources/lewm_v3_pdf.pdf`。早期 `lewm_html.html` 是另一版本，不用于下面 solver 表。`*.parsed.json` 是派生解析，不替代原始页表。

既有[完整封存导出](/home/gbk/code/LeWorldModel/le-wm-main/results/p2_reported_baseline_table_20261027_v3/export/report.json) SHA256：`52e687811e6d6f728ab3eda60836ea05eff11f36da2bb4af975d1dd19ef380f1`（本次重新计算）。原导出选定 238 cells＝236 数值＋2 原文缺失；这是表格单元数，**不是样本数或本地独立试验数**。本清单核下列原表与协议，不声称本次重新执行了全部导出程序。旧来源文档中的“当时尚不可访问”记录保留，不据此否认现在本地已有一手缓存。

## 3. GRASP：必须保留 learned-model success 限定

### 3.1 可直接引用的原表数值

[Table 1 / S4.T1 / p7](https://arxiv.org/html/2602.00475v1#S4.T1)：Push-T open-loop，每 setting 500 trials。每格为 **成功率 % / 成功试验的中位成功时间 s**，时间分母不是全部 trials。

| 方法 | H40 | H50 | H60 | H70 | H80 |
| --- | --- | --- | --- | --- | --- |
| CEM | 61.4 / 35.3 | 30.2 / 96.2 | 7.2 / 83.1 | 7.8 / 156.1 | 2.8 / 132.2 |
| LatCo | 15.0 / 598.0 | 4.2 / 1114.7 | 2.0 / 231.5 | 0.0 / missing | 0.0 / missing |
| GD | 51.0 / 18.0 | 37.6 / 76.3 | 16.4 / 146.5 | 12.0 / 103.1 | 6.4 / 161.3 |
| GRASP | 59.0 / 8.5 | 43.4 / 15.2 | 26.2 / 49.1 | 16.0 / 79.9 | 10.4 / 58.9 |

[Table 3 / S4.T3 / p8](https://arxiv.org/html/2602.00475v1#S4.T3)：短程成功率 %，每 setting 500 trials。每格按 **H10 / H20 / H30** 排列，不是误差条。

| 方法 | Push-T | PointMaze | WallSingle |
| --- | --- | --- | --- |
| CEM | 100.0 / 96.0 / 76.0 | 100.0 / 100.0 / 100.0 | 98.0 / 100.0 / 100.0 |
| LatCo | 67.4 / 55.2 / 35.4 | 100.0 / 99.8 / 99.0 | 91.4 / 81.6 / 62.8 |
| GD | 99.2 / 88.4 / 69.0 | 94.6 / 91.8 / 94.6 | 91.8 / 92.8 / 91.6 |
| GRASP | 100.0 / 89.2 / 75.2 | 100.0 / 100.0 / 99.8 | 95.2 / 91.8 / 95.0 |

[Table 4 / S4.T4 / p8](https://arxiv.org/html/2602.00475v1#S4.T4)：median completion time，单位 s；每格仍按 H10 / H20 / H30。表注未明确 successful-only 分母，**不能套用 Table 1 的时间定义**。

| 方法 | Push-T | PointMaze | WallSingle |
| --- | --- | --- | --- |
| CEM | 1.3 / 7.5 / 23.6 | 0.7 / 2.0 / 6.3 | 0.9 / 1.5 / 2.1 |
| LatCo | 39.9 / 437.2 / 1800.0 | 1.6 / 13.7 / 36.6 | 12.1 / 23.8 / 111.1 |
| GD | 0.5 / 2.4 / 25.3 | 0.3 / 0.6 / 1.2 | 0.4 / 0.5 / 0.7 |
| GRASP | 1.5 / 2.1 / 9.1 | 0.7 / 1.8 / 2.1 | 1.6 / 1.8 / 2.1 |

### 3.2 模型、数据、预算与可比性合同

- [§4.2 / S4.SS2.p1.1–p2.1](https://arxiv.org/html/2602.00475v1#S4.SS2)：使用 DINO-wm 框架训练的世界模型，模型输入5个原始动作预测5步，二维动作组成10维包。原文明确：**“All reported metrics measure task success under the learned world model.”** 成功是在规划 horizon 内进入 goal region，不是本地 native simulator success。
- [§4.1 / S4.SS1.p2.1](https://arxiv.org/html/2602.00475v1#S4.SS1.p2.1)：按环境/horizon sweep 超参并使用最好 setting；状态初始化为带噪起终点线性插值，动作初始化零。不能从该表推断固定同总算力比较。
- Missing：逐表 checkpoint SHA、绑定的数据 split / task IDs、完整 forward/backward/并行批量/硬件成本账，以及 H 与 primitive/model-step 的无歧义映射，本清单未建立；不自行换算为本地 H2/H3。
- 可支持的引用：GRASP 作者在其 learned-model benchmark 的长程 Push-T 上报告上述成功与时间。不能把这些百分数改写为真实物理成功率，不能把本地 nominal/restart-Adam 变体标为官方 GRASP 复现，也不能据作者表替代本地强基线。

## 4. Hi-LeWM：固定配置展示，训练边界不可省略

### 4.1 可直接引用的原表数值

[Table 4 / A3.T4 / p13](https://arxiv.org/html/2607.12547v2#A3.T4)：以下固定 `(high_h, high_replan, low_h, low_r)=(2,5,2,1)`，列出全部三个 offset、两种 schedule，不逐格挑最优。单位成功率 %，± 为原表明确的跨 seed 标准差（SD）。`-C` 是 constrained empirical-macro CEM，不是另一个相同支持域的同预算标签。

| schedule | d（环境步 offset） | PushT Hi-LeWM | PushT Hi-LeWM-C | Cube Hi-LeWM | Cube Hi-LeWM-C |
| --- | --- | --- | --- | --- | --- |
| staged | 25 | 90.7±6.2 | 70.7±5.2 | 65.3±2.5 | 60.7±2.5 |
| staged | 50 | 42.0±2.8 | 64.0±3.3 | 46.7±2.5 | 48.0±0.0 |
| staged | 75 | 15.3±4.1 | 22.0±3.3 | 49.3±0.9 | 50.0±4.9 |
| online | 25 | 28.0±8.5 | 28.7±5.2 | 69.3±2.5 | 54.7±0.9 |
| online | 50 | 35.3±6.6 | 34.7±3.4 | 50.7±3.4 | 48.7±2.5 |
| online | 75 | 15.3±3.8 | 32.7±3.8 | 58.7±4.7 | 42.7±4.1 |

[Table 3 / A3.T3 / p13](https://arxiv.org/html/2607.12547v2#A3.T3)：同论文作者复现的 flat LeWM，不是 LeWM 原论文 Table 10。固定 h5/r5，action block=5，CEM=300×30；均值±SD，3 seeds。

| d | 环境执行 budget | PushT 成功率 % | Cube 成功率 % |
| --- | --- | --- | --- |
| 25 | 50 | 94.0±2.0 | 65.3±4.2 |
| 50 | 100 | 52.7±5.0 | 52.0±3.5 |
| 75 | 150 | 18.0±2.0 | 53.3±7.0 |

[Table 5 / A3.T5 / p14](https://arxiv.org/html/2607.12547v2#A3.T5)：两环境共享的 Hi-LeWM CEM 预算，不是 flat LeWM 的预算。

| d | 环境执行 budget | 高层 samples×steps / top-k | 低层 samples×steps / top-k |
| --- | --- | --- | --- |
| 25 | 50 | 900×20 / 10 | 300×30 / 150 |
| 50 | 100 | 1500×40 / 10 | 900×30 / 150 |
| 75 | 150 | 1200×60 / 10 | 1200×30 / 150 |

### 4.2 模型、数据、训练与协议

- [§3 / S3.p1.1](https://arxiv.org/html/2607.12547v2#S3.p1.1)：PushT-v1、OGBCube-v0，图像 goal-conditioned control，stable-worldmodel library。d=25/50/75 是**环境步的目标 offset**，不是 high_h、low_h 或本地模型步 H。
- [A3.SS2.p2.1–p4.1](https://arxiv.org/html/2607.12547v2#A3.SS2)：从 expert demonstration 同轨迹取 start 与 future goal；每配置50 episodes，通常3随机 seeds。原文主文报告 sweep 最佳值；这里不将固定配置表称作全 sweep 最优。完整23配置继续保留在[既有长表](/home/gbk/code/LeWorldModel/le-wm-main/results/p2_reported_baseline_table_20261027_v3/export/hi-lewm_reported.md)。
- [S2.SS0.SSS0.Px2.p1.1](https://arxiv.org/html/2607.12547v2#S2.SS0.SSS0.Px2.p1.1)：冻结低层 LeWM，**另外训练 macro-action encoder 与 high-level predictor**。[A3.SS6.p1.1](https://arxiv.org/html/2607.12547v2#A3.SS6.p1.1)：-C bank 由训练动作流的连续片段经 learned macro encoder 编码。不能称整体 training-free 或只替换 planner。
- [A1.SS1.p1.1](https://arxiv.org/html/2607.12547v2#A1.SS1.p1.1)：作者报告最终 Hi-LeWM 训练 PushT 为单 A100 的7.56小时，Cube 为单 H100 的12.80小时；evaluation run 约5–58分钟。这是对应作者实现的成本陈述，**不是完整预训练成本或本地 latency**。
- Missing：逐表完整 checkpoint/revision、数据 split/task IDs、所有训练阶段 lineage、实际每次重规划与编码调用合计。本地 moving 两根没有与其 expert goal-offset 协议配对；不跨表相减声称本地提升。

## 5. LeWM v3：solver 表可引用，不能补造 solver 公平预算

[Table 10 / A7.T10 / p26](https://arxiv.org/html/2603.19312v3#A7.T10)，Push-T success rate（按原表数值保留）：

| solver（原表标签） | LeWM | PLDM |
| --- | --- | --- |
| CEM | 96.0±2.83 | 78.0±5.0 |
| SGD (β1) | 26±4.32 | 4.67±0.06 |
| RMSProp (β2) | 67.33±2.49 | 49.33±8.26 |
| Adam (β1, β2) | 84±7.12 | 80±3.27 |

核验边界：Table 10 的 caption 写 SR，没有在该表交代 ± 是 SD/SEM/CI；本清单保留符号，不赋予未经核实的统计含义或 solver-specific seed 数。表名与版本已核；旧 v1 缓存不作为此表来源。

模型/数据/规划上下文：

- [A4.SS0.SSS0.Px4.p1.1](https://arxiv.org/html/2603.19312v3#A4.SS0.SSS0.Px4.p1.1)：论文一般 CEM 配置为每轮300 candidates，PushT 最多30轮（其他环境10轮），elite30，初始 variance1；horizon5个 block、frame skip5，即25环境步；整段优化序列执行后再规划。
- [A4.p1.1](https://arxiv.org/html/2603.19312v3#A4.p1.1)：一般训练使用224×224、frame skip5，batch128，4 frames / 4 blocks of 5 actions 的子轨迹。[A6.SS1.p1.1](https://arxiv.org/html/2603.19312v3#A6.SS1.p1.1) 描述同 offline trajectory 的 start/goal；PushT 一般 evaluation 为目标 offset25、执行预算50环境动作。
- 上述是**论文一般设置的已核原句，不是所有 Table 10 solver 已独立匹配该预算的证明**。SGD/RMSProp/Adam 的迭代、学习率、停机、AD调用/计算成本以及该表逐 cell task/checkpoint 绑定仍 missing；不可给它们套上 CEM300×30，或宣称 Table 10 等总算力。
- 可引用句：“LeWM v3 Table 10 在其 Push-T 规划协议上报告 LeWM+CEM 96.0±2.83、LeWM+Adam 84±7.12 的 SR。”必须保留作者协议限定；不能写成本地 GN/D2P/nominalMPPI 的复现结果，也不能把 frozen local asset 直接认证为该表 checkpoint。

## 6. PlanWAM：驾驶指标可引用，不与操纵成功率横排

### 6.1 已核 HTML 原表数值

| 原表定位 | 环境 / split | 方法 | 指标 | 作者报告值 |
| --- | --- | --- | --- | --- |
| [Table 1 / S4.T1](https://arxiv.org/html/2610.11382v1#S4.T1) | NAVSIM-v2 / navtest | PlanWAM | EPDMS | 90.9 |
| [Table 2 / S4.T2](https://arxiv.org/html/2610.11382v1#S4.T2) | NAVSIM-v1 / navtest | PlanWAM | PDMS | 93.8 |

[Table 3 / S4.T3](https://arxiv.org/html/2610.11382v1#S4.T3)，HUGSIM zero-shot closed-loop，0–100评分：

| 指标 | Easy | Medium | Hard | Extreme | Avg. |
| --- | --- | --- | --- | --- | --- |
| Route Completion (RC) | 87.6 | 48.3 | 51.9 | 37.2 | 53.7 |
| HD-Score | 75.3 | 32.9 | 35.6 | 22.0 | 38.7 |

这些值不是 manipulation success %。Avg. 为所有436 scenarios 的按样本数加权均值，不是四个难度列的简单平均；[A3.SS4.p1.1、p3.3](https://arxiv.org/html/2610.11382v1#A3.SS4) 已核。没有 PDF 页码时直接引用版本化 HTML 表锚，不补猜页数。

### 6.2 模型、数据、训练、规划预算

- [§4.1 / S4.SS1.SSS0.Px1.p1.1](https://arxiv.org/html/2610.11382v1#S4.SS1.SSS0.Px1.p1.1)：NAVSIM navtrain 训练/navtest 评价；NAVSIM-v1 用 PDMS，v2 用 EPDMS；直接部署到 HUGSIM，无 HUGSIM-specific fine-tuning，不等于整体无需训练。
- [S4.SS1.SSS0.Px2.p1.1](https://arxiv.org/html/2610.11382v1#S4.SS1.SSS0.Px2.p1.1)：4历史帧×4相机；DINOv2 ViT-S 基础权重冻结但带 rank32 LoRA；每 camera-frame16 registers，128 history tokens，256 future-latent tokens；4秒内64 candidate trajectories、每条8 waypoints，另有 scorer 选优。
- 同段：两阶段 AdamW，16张 A800，每 GPU batch8，stage学习率1e-4/1e-5，latent alignment weight0.05。不能将“frozen base”写成全模型冻结；不能把64 candidates 写成64次完整神经 forward，更不能与本地 predictive-call 数认定等预算。
- Missing：PDF 页、逐表 checkpoint、trial/seed IDs、完整训练与推理耗时/forward/backward/module ledger。任务、观测、多相机、目标、指标与 moving PushT 本地合同不同，只能作为异协议相关工作/作者结果单列，不能替代本地强基线。

## 7. 给 MAIN 的引用与剩余前置

| 证据用途 | 当前可做 | 不可越过的边界 / missing |
| --- | --- | --- |
| 四篇 reported 数值表 | 引用上述固定版本、原表、任务、指标和必要限定；原文已核 | 不是官方代码/权重认证，不代表最新版本认证 |
| 本地实验章节的相关工作数值 | 单独标“作者报告，异协议”，保留模型/训练/预算差异 | 不拼入 moving 根同协议结果排名，不做 SOTA 或等总算力结论 |
| GRASP 长程机制先例 | 引用 learned-world-model 成功与 successful-only 时间 | 不能替代真实物理成功或视为本地 GRASP 官方复现 |
| Hi-LeWM 与 LeWM 对照 | 明确是 Hi-LeWM 作者 flat baseline 或 LeWM 原文 solver 表，两者分源 | 不能混淆两套 seeds/模型/训练/目标 offset/solver预算 |
| 进一步本地强基线 | 另需批准的实现/权重/输入输出/任务与预算合同及 actual 执行 | 本文不满足也不绕过 P1/P2 科学准入；固定两根仅 calibration |

本次交付没有新增本地 scientific outcome。Gap verdict 仅限引用层：四篇均已有可定位的一手数值，不能再一概称“没有原文可用”；但同协议强基线与公平预算的缺口**没有被作者 reported 数值填补**。本清单没有据成功或失败下理论结论。
