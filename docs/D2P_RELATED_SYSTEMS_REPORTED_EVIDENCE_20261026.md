# D2P相关系统：原文/官方项目证据与定位（2026-10-26）

状态：partial。范围是HarnessPAI、WorldForge、AW-0以及AVIC / When and How Much to Imagine；LeWM/PlanWAM另见 `docs/D2P_REPORTED_BASELINES_LEWM_PLANWAM_20261026.md`。本页不拼跨任务性能榜，不将官方项目页表值伪装为已核验论文PDF，不将内部提案视为一手来源。

## 身份与证据层级

| 候选 | 本轮成功取得的来源 | 已核验身份/当前证据边界 |
| --- | --- | --- |
| HarnessPAI | 原始提案仅作检索线索；官方arXiv检索没有取得可核验结果 | `unverified`：全标题、作者、ID、版本、表页、协议、代码/模型URL均未确认。不确认提案“2026-09”或“跨rollout程序改进”是原文事实，也不断言论文不存在 |
| WorldForge | 同上 | `unverified`：名字可能重名；不得猜仓库或把同名非world-model项目当作目标。provider/proposal/predict/score/CEM组合描述仍待原文核验 |
| AW-0 | 成功读取[作者项目页](https://agentic-wam.github.io/)正文、方法/消融/局限及BibTeX；网页输出部分截断，未读到的内容不补造 | *Agentic World Action Models*，Zhuoyang Liu、Dantong Niu等；项目页BibTeX指向[arXiv:2610.30136](https://arxiv.org/abs/2610.30136)。AW-0为模型名。论文版本号、PDF表页及提交日期尚未核验，不能确认提案“10月21日” |
| AVIC / AVIC-R | 成功完整读取[作者项目页](https://adaptive-visual-tts.github.io/)及BibTeX | *When and How Much to Imagine: Adaptive Test-Time Scaling with World Models for Visual Spatial Reasoning*，Shoubin Yu、Yue Zhang、Zun Wang、Jaehong Yoon、Huaxiu Yao、Mingyu Ding、Mohit Bansal；网页指向[arXiv:2602.08236](https://arxiv.org/abs/2602.08236)。AVIC为Adaptive Visual Imagination Control，AVIC-R为另一个训练策略版本；网页标NeurIPS 2026，但本轮未独立核验会议论文集或PDF版本 |

“AVIC”名称由作者页面确认；不得把它扩写为别的缩写、误当纯物理MPC系统，或将AVIC-R的额外训练抹掉。以上两篇已定位作品仍缺原论文版本锁定；本页数值来源层级仅为`official_project_reported`，不是`paper_table_pdf_verified`。

## 逐系统科学职责，不强行排名

### AW-0：程序来源与训练范式对照

作者页描述code、future video、robot actions三流的mixture-of-transformers联合生成；code expert使用LLaDA并接冻结Qwen3-VL ViT，video expert使用Cosmos-Predict2.5，action expert生成连续动作。条件含当前多视角图像、语言、robot state及已有程序切片memory，不是单目标RGB输入。训练程序来自自动标注/技能库整理，而不是D2P冻结动力学局部谱即时生成。

项目页还描述hybrid execution：重规划时选择执行生成的程序或action expert的动作chunk；其效果不能简单归为某个CEM/GN优化器。适合作为“有训练的程序接口/技能复用”定位参照，不能将多模态训练系统的成功率和本地Reacher/PushT局部P1直接混比。

可核验的**官方项目页**示例（只保留一组内部消融，不构成跨系统榜）：

| 来源位置 | 系统/对照 | 指标 | 网页报告值 |
| --- | --- | --- | ---: |
| Table 3，code generation paradigm | Joint denoise | 8项真实任务平均rubric score，% | 71.1 |
| 同表 | Autoregressive code generation | 同上 | 54.5 |
| Figure 14 | Joint denoise | RTX 5090 inference latency | 0.23 s |
| 同图 | Autoregressive | 同上 | 0.55 s |

这些不是二元整任务success rate，也不是当前D2P wall time。网页还列YAM平行夹爪与Tianji–Sharpa灵巧手任务；每项试验数、seed、checkpoint revision、action horizon/chunk、stop criterion及误差条定义未在本轮取得的原论文中核验，均`unverified`。不能把71.1与PlanWAM的PDMS或AVIC的QA accuracy互比。

项目页明确列出物体拖动、对齐失误、抓取/接触丢失等失败，并承认程序语法有效不保证物理执行正确。这支持D2P讨论“动作见证不等于真实可达保证”，不证明D2P已解决这些问题，也不构成AW-0理论或方法的否定。

### AVIC：视觉空间推理中的选择性想象

作者页分开Direct QA、always-on MindJourney、AVIC及AVIC-R。AVIC由policy model判断现有证据是否足够再调用想象；AVIC-R使用GRPO、QA正确性与想象成本奖励训练gating/planning policy，同时冻结world model与QA model。因此不能把AVIC-R当作D2P“不训练额外router”的同约束对手。

以下均为作者网页报告、同一网页表中的指定设置，不是本地复现：

| 网页位置/任务 | 指定方法与模型 | 原指标/报告值 | 适用边界 |
| --- | --- | --- | --- |
| Table 1，SAT-Real | GPT-4.1 QA，Direct | Avg accuracy 74.0%，Avg WM 0，Avg Token 0.7K | token与WM单位依网页，不能映射成本地predict samples |
| 同表 | GPT-4.1 QA，MindJourney | 77.3%，Avg WM 12.34，67.1K tokens | 固定想象对照 |
| 同表 | GPT-4.1 QA，AVIC，GPT-4.1 policy | 79.3%，Avg WM 0.73，7.6K tokens | 自适应视觉想象，不是PushT控制 |
| 同表 | GPT-4.1 QA，AVIC-R，Qwen2.5VL-7B policy | 80.0%，Avg WM 3.03，5.2K tokens | 有策略训练，不能与AVIC合并归因 |
| Table 3，R2R | MapGPT GPT-4o | SR 36.0，SPL 30.8，NE 6.04 | 语言导航任务，非本地PointMaze |
| 同表 | MapGPT + AVIC，GPT-4o | SR 37.5，SPL 31.9，NE 5.97 | SR/SPL越高越好、NE越低越好；网页未明确的单位不补填 |

Table 1还包含其他QA backbones；这里只作模型明确的示例，不声称所有模型或任务均提高，也不从这些汇总值补配对显著性。SAT/MMSI/R2R的确切split、测试query ID、seeds、world-model权重与动作视角生成horizon、成本统计边界、参数许可和PDF页码仍待原文核验。网页“更多想象并非单调更好”可作C3设计背景，不能直接证明D2P几何调度有效。

## 资产可用性：链接公布≠已可运行

| 系统 | 作者页面已提供的资产链接 | 本轮确认程度 |
| --- | --- | --- |
| AW-0 | [代码](https://github.com/ZhuoyangLiu2005/agentic-wam)、[模型](https://huggingface.co/miniFranka/agentic-wam-pretrain-yamego3k-soup-e1) | 确认作者页href；未请求权重、未验证仓库revision、许可证、文件大小、gating或checkpoint完整性；状态`advertised_link_only` |
| AVIC | [代码](https://github.com/Yui010206/Adaptive-Visual-Imagination-Control/)、[策略模型](https://huggingface.co/Shoubin/AVIC-Qwen2.5-VL-7B-policy) | 同上；策略模型不能假定为完整world-model+QA bundle |
| HarnessPAI / WorldForge | 无已确认官方唯一URL | `unverified`，不虚构repo、许可或模型 |

## P0/P1可比性与后续缺口

当前Reacher P0是固定LeWM、RGB/history动作预测与物理关节距离，sample-matched calibration仍不等于总算力匹配；PushT新suffix是known4/all16固定前缀后的局部干预；P1必须验证同根非贪心、真实能力改变与真实收益。这里的代码程序、多模态机器人任务、QA accuracy或语言导航均不能直接填这些门。

- AW-0/AVIC：现阶段`context_only`。可用于程序来源/训练约束、信息访问及计算调度的相关工作定位，不进入同协议D2P性能主榜。
- HarnessPAI/WorldForge：身份与协议`unverified`，连定位事实也只作为待核验问题，不能先写“已超过/已覆盖D2P”。
- 只有取得论文锁定版本和完整任务/模型/观察/动作/预算/指标/seed设置后，才能录入`paper_reported`。是否进行本地移植由主线程另行冻结协议，不能直接凭网页数字判断公平胜出。

## 访问与执行记录

成功：作者AW-0与AVIC项目页只读HTTP访问；本地缓存/文献索引定向检索。失败/未完成：HarnessPAI/WorldForge官方arXiv检索未得到目标正文，组合检索一次返回无有效列表，后续正常审批超时；不将超时写成不存在。没有代理/连接器绕过审批，没有下载大权重、运行GPU、训练或本地基线实验。本任务仅新增本文件。
