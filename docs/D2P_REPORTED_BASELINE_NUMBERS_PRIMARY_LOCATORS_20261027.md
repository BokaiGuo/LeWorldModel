# Reported baseline numbers：原文页表与协议补充核实

日期：2026-10-27。仅本地论文查证；没有网络请求、下载、模型、物理或训练。采用 literature-review 技能的 evidence-map 方法，问题限定为四篇固定版本的数值出处与协议，不做系统综述、SOTA排名或理论结论。

本文件补充[已冻结交接清单](/home/gbk/code/LeWorldModel/docs/D2P_REPORTED_BASELINE_NUMBERS_HANDOFF_20261027.md)，不替换、不修改它。旧文档 SHA256 仍为 `6dfd76a5d21034babc039c0f1f0bce94e4099baa8bb12c594b1d73e33e97ab2a`。所有 reported 数字均为作者结果，不是本地 matched reproduction，更不替代本地强基线。

## 1. 页表定位：本次实际读取哪些原文

PDF页均为1起始的物理页；下列三份PDF对应页的页脚一致。表号是原文表号，不是缓存解析器的数组索引。PlanWAM 在本仓库定位到 HTML 原文，没有定位到该文PDF，故所有PDF页码继续 missing；不能从HTML章节顺序推算页码。未核最新在线版本。

| 论文固定版本 | 结果原表 / PDF页 | 预算、模型与数据原文 / PDF页 | 可直接引用的边界 |
| --- | --- | --- | --- |
| GRASP，[2602.00475v1](https://arxiv.org/html/2602.00475v1) | [Table 1](https://arxiv.org/html/2602.00475v1#S4.T1) p7；[Table 2](https://arxiv.org/html/2602.00475v1#S4.T2) p7；[Tables 3/4](https://arxiv.org/html/2602.00475v1#S4.T3) p8 | §4.1 p6–7：每环境/H的超参sweep；§4.2 p7：DINO-wm、5动作包、learned-model成功定义 | 成功率是 learned-model success；Table 1时间只统计成功试验，Table 4分母仍未证实 |
| Hi-LeWM，[2607.12547v2](https://arxiv.org/html/2607.12547v2) | [Table 2](https://arxiv.org/html/2607.12547v2#A2.T2) p11；[Tables 3/4](https://arxiv.org/html/2607.12547v2#A3.T4) p13；[Table 5](https://arxiv.org/html/2607.12547v2#A3.T5) p14 | Appendix A/A.1 p9：代码权重声明/硬件时间；B.1/B.2 p9：结构/waypoint；B.3 p10：训练；C.2 p12：episode/offset/分母；C.6 p14：empirical bank | 训练新高层而非全模型training-free；d是环境步offset；Table 3是该文作者复现flat LeWM |
| LeWM，[2603.19312v3](https://arxiv.org/html/2603.19312v3) | [Table 5](https://arxiv.org/html/2603.19312v3#A7.T5) p24；[Table 10](https://arxiv.org/html/2603.19312v3#A7.T10) p26 | Appendix D p17–18：输入/encoder/predictor/CEM；E p18：数据；F.1：HTML锚[A6.SS1](https://arxiv.org/html/2603.19312v3#A6.SS1)，本次未新增确认其PDF页 | Table 10误差条含义及各梯度solver的完整预算仍missing；不从一般CEM配置推定公平性 |
| PlanWAM，[2610.11382v1](https://arxiv.org/html/2610.11382v1) | [Tables 1/2/3](https://arxiv.org/html/2610.11382v1#S4.T1)；PDF页missing | [Table 11](https://arxiv.org/html/2610.11382v1#A2.T11)训练；[Table 12](https://arxiv.org/html/2610.11382v1#A3.T12)数据/评价；[Table 13](https://arxiv.org/html/2610.11382v1#A3.T13)场景分母；PDF页均missing | 驾驶PDMS/EPDMS/RC/HD-Score，不是操纵成功率；64候选不等于64完整forward |

本次核查直接使用本地 PDF 经 `pdftotext -layout -f PAGE -l PAGE FILE -` 输出，无中间产物；同时读取原HTML节点。六组PDF目标页检查通过：GRASP p7/p8、Hi-LeWM p13/p14、LeWM p24/p26。它们检查目标页的表号及代表性数值存在，不是实验测试或独立试验。

## 2. PlanWAM：补齐训练时长口径与评价分母

### 2.1 原表结果锚点不变

以下均已核原HTML，PDF页missing：

| 原表 | 方法/任务/指标 | 作者报告值 |
| --- | --- | --- |
| Table 1，S4.T1 | PlanWAM，NAVSIM-v2 navtest，EPDMS | 90.9 |
| Table 2，S4.T2 | PlanWAM，NAVSIM-v1 navtest，PDMS | 93.8 |
| Table 3，S4.T3，Route Completion | HUGSIM E/M/H/X/Avg. | 87.6 / 48.3 / 51.9 / 37.2 / 53.7 |
| Table 3，S4.T3，HD-Score | HUGSIM E/M/H/X/Avg. | 75.3 / 32.9 / 35.6 / 22.0 / 38.7 |

### 2.2 本次新增确认的原文预算与数据字段

[Table 11 / A2.T11](https://arxiv.org/html/2610.11382v1#A2.T11) 已核：输入4历史帧、front/left/right/rear，1148×672；训练专用future observations在1/2/3/4秒，同四相机。训练为 **AdamW，每阶段50 epochs**；前10% steps线性warm-up，随后cosine decay；16×A800，8 samples/GPU，global batch128。规划4秒、8 waypoints（0.5秒间隔）、K64；DINOv2 ViT-S frozen base + rank32 LoRA。阶段学习率1e-4/1e-5。

这是作者设定的epochs，不是已核总GPU小时、全预训练成本、实际module调用数或本地latency；这些仍missing。

[Table 12 / A3.T12](https://arxiv.org/html/2610.11382v1#A3.T12) 已核：NAVSIM-v1基于OpenScene/nuPlan日志，non-reactive 4秒proposal simulation；navtrain **103,288 scenes（原文括注85k train / 18k validation）**，navtest **12,146**。括注为原文近似写法，不将85k+18k擅自解释为精确103,288的拆分。NAVSIM-v2采用本文相同split、扩展official proposal scorer。HUGSIM无simulator-specific adaptation。

[Table 13 / A3.T13](https://arxiv.org/html/2610.11382v1#A3.T13) 已核：

| source | scenarios | E | M | H | X |
| --- | --- | --- | --- | --- | --- |
| nuScenes | 88 | 18 | 34 | 18 | 18 |
| Waymo | 108 | 22 | 42 | 22 | 22 |
| KITTI-360 | 113 | 25 | 36 | 26 | 26 |
| PandaSet | 127 | 15 | 45 | 30 | 37 |
| Total | 436 | 80 | 157 | 96 | 103 |

分母算术80+157+96+103=436。Table 3 Avg. 按scenario数加权，不是四难度等权；原列值已舍入，不要求用一位小数的列值反算出未舍入原均值。不将RC/HD-Score改名为native success。

解析注意：此HTML的Tables 11/12/13 caption节点本身不包含数据行，行位于同一父容器中的相邻表。此次读取caption与该对应相邻表，不把“只读到caption”误报为“原表没数据”。无修改原缓存或解析器。

## 3. LeWM：补齐架构与数据；误差条不跨表继承

[Appendix D，p17–18](https://arxiv.org/html/2603.19312v3#A4) 已核：encoder为ViT-Tiny、patch14；predictor为ViT-S，learned positions、causal masking；PushT/OGBench-Cube history长度3，TwoRoom长度1；planning autoregressive。一般CEM为300 candidates、PushT最多30轮、其他环境10轮、elite30；horizon5 blocks=25环境步，执行整段后replan。

[Appendix E，p18](https://arxiv.org/html/2603.19312v3#A5) 本次直接核到的数据与训练epochs：

| 任务 | 原文数据规模/来源 | 原文训练epochs |
| --- | --- | --- |
| PushT | 20,000 expert episodes，平均196步；沿用DINO-WM设置/数据 | 10 |
| TwoRoom | 10,000 episodes，平均92步；noisy heuristic经门到目标 | 10 |
| OGBench-Cube | single-cube，10,000 episodes，各200步；benchmark heuristic | 10 |
| Reacher | 10,000 episodes，各200步；SAC policy收集 | 10 |

这些是论文一般训练数据陈述，不是本地HDF5/模型文件与Table 10单元格的SHA绑定。

[Table 5，p24](https://arxiv.org/html/2603.19312v3#A7.T5) 可额外引用：PushT SR，DINO-WM **92.0±1.63**、PLDM **78.0±5.0**、LeWM **96.0±2.83**。Caption明确“三训练种子均值及corresponding variance”、同一组50 trajectories、goal25步可达、执行budget50步。原文caption用词为variance；本清单不擅自改成SD或SEM。

[Table 10，p26](https://arxiv.org/html/2603.19312v3#A7.T10) 原值再次定位：LeWM CEM96.0±2.83 / SGD26±4.32 / RMSProp67.33±2.49 / Adam84±7.12；PLDM依次78.0±5.0 / 4.67±0.06 / 49.33±8.26 / 80±3.27。虽然两个CEM数值与Table 5相同，**数值相同不足以证明Table 10全部solver沿用Table 5的误差统计、种子和轨迹**。其梯度solver的迭代/学习率/停机/AD账仍missing，不补造。

## 4. Hi-LeWM：训练成本与表示变体必须分合同

[Appendix A/A.1，p9](https://arxiv.org/html/2607.12547v2#A1) 原文声明代码和checkpoints位于 `doi:10.5281/zenodo.21353240`。这是**论文中的资产指针声明**，本次没有联网访问、下载或核对该存储库内容/权重SHA，不能标成已认证官方本地checkpoint。

同页作者报告最终训练PushT单A100 **7.56小时**、Cube单H100 **12.80小时**；评价每run50 episodes、约5–58分钟。不得把最终高层训练时间写成包含低层预训练的总成本。

[B.1/B.2，p9；B.3，p10](https://arxiv.org/html/2607.12547v2#A2) 新核：主配置macro latent维32；action-chunk最大长度15，random_sorted选N5 waypoints、最大span15。AdamW学习率5e-5、weight decay1e-3、batch128、clip norm1、bf16。总参数30.5M＝trainable高层12.5M＋frozen低层18.0M。原文称模型最多训练15 epochs，VQ变体50 epochs。不能把VQ与连续宏动作当相同训练预算。

[Table 2，p11](https://arxiv.org/html/2607.12547v2#A2.T2) 为各表示设计的 **best PushT success %**，不是与本地根配对结果；原表未给这些单元格误差条：

| sampling / representation | size | d50 | d75 |
| --- | --- | --- | --- |
| random / continuous | latent32 | 46 | 20 |
| fixed stride / continuous | latent32 | 42 | 20 |
| fixed stride / continuous | latent8 | 42 | 20 |
| random / VQ | 16 codes | 38 | 20 |
| random / VQ | 128 codes | 44 | 34 |

不要把此表示sweep的46/20与Table 4的另一planning配置数值混为同一行，或把VQ128的34当主连续模型结果。Tables 3/4 p13、Table 5 p14的原表定位已再次核实；旧交接清单保留固定配置，不将它宣称为每格最佳配置。C.2 p12明确d是环境步offset、每配置50episodes、通常三seeds；原表模型、训练与高低层CEM成本不能只折成一个本地forward计数。

## 5. GRASP：补充原文消融，保留成功条件分母

[Table 2，p7](https://arxiv.org/html/2602.00475v1#S4.T2) 原表：Push-T H40；时间为成功trial的中位数秒。Caption称每100 stochastic steps做一次GD sync，“Sync Steps”行是该组消融的原表标签；不将其误读为总planning调用数。

| 组别 | setting | accuracy % | median successful time s |
| --- | --- | --- | --- |
| Sync Steps | No Sync | 1.6 | 4.5 |
| Sync Steps | 10 | 48.0 | 11.2 |
| Sync Steps | 25 (ours) | 59.0 | 8.5 |
| Sync Steps | 50 | 58.0 | 9.8 |
| Noise Level | sigma0.0 | 54.8 | 10.4 |
| Noise Level | sigma0.5 (ours) | 59.0 | 8.5 |
| Noise Level | sigma1.0 | 50.6 | 8.4 |
| Detach state gradient | Stop (ours) | 59.0 | 8.5 |
| Detach state gradient | Flow | 46.6 | 10.3 |

这是原论文特定机制消融，不是本地Adam restart实现的认证，也不是“更短时间就更好”的无条件比较。Table 1 p7、Tables 3/4 p8及§4.2“all reported metrics”限定再次确认；Table 1两个LatCo时间原文为破折号，仍missing。未将Figure 5的Wald95% CI定义传播到其他表，未从曲线读取未经标注数值。

## 6. 可复核的原始文件与未解项

路径根为 `/home/gbk/code/LeWorldModel/le-wm-main/results/p2_reported_baseline_table_20261027_v3/sources/`，本次重新计算：

| 原文文件 | SHA256 |
| --- | --- |
| grasp_pdf.pdf | `633bff79e5913401f578d4f13cbee65526221fef859d865cdaab6cfc0fa802a8` |
| hilewm_pdf.pdf | `03f0cf2e39166157c21baebb7808992f3481f9edf2b6cb4d771103e87b586533` |
| lewm_v3_pdf.pdf | `553a01501ac74ec56721cfdb936d3825ac941734133a1fd761f614c01b6887e9` |
| planwam_html.html | `67be22fee283aa4bad78578c91d83e6a3085bc8927c63597ee7c9721073bd873` |

明确保留missing：PlanWAM PDF页；各表checkpoint/data split/task IDs与本地资产一一绑定；LeWM Table 10梯度solver预算与误差定义；GRASP完整调用/训练/硬件账和Table 4时间分母；所有作者结果到本地moving两根的同协议可比性。论文指针存在不等于权重已核，epochs存在不等于GPU-hours已核，shared编码或forward数相近不等于等总算力。

交付结论：补齐的是读者可定位的作者报告和协议字段；没有新增本地科学结果，没有将reported作为local reproduced，也没有以两根calibration或任何失败推断P1理论成立/失败。
