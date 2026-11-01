# PRISM / IMWM 原文审查与 SAGE 附件核验补充（2026-10-26）

## 状态与证据边界

按 `academic-research-suite` 的 `deep-research/fact-check` 与 `source_verification_agent` 纪律，本次先核对新附件对 SAGE 的协议区分，再扩展 PRISM/IMWM。**三项均未取得可读原文，未新增任何已核实外部性能值。** 本文为 `partial` 来源审查，不是原论文核验成功、不构成论文不存在的判定，也不支持 efficacy、完整 compute matching 或 P1 准入。

本轮唯一新增本文件，不改动既有 SAGE/SuperMimic 文档、其它文档、runner 或准入代码。SAGE 的新增核验约束因此集中保存于下节，供后续原文核验使用；旧文档的 unknown 不因收到附件而自动解除。

## 新附件与 SAGE：明确区分，但不把转录升级为原文证据

已完整读取附件：`/home/gbk/.codex/attachments/380fd905-cc8f-4924-a507-d9292f925635/已粘贴的文本.txt`。

SHA-256：`893dbbd2b1ffa793233b88947911f179beb298df7a8ee1510eb63bec3dadc4de`。该哈希只绑定附件内容，不认证其中的论文转录真实。附件为研究说明文本，不是官方论文 PDF，也没有唯一论文 URL/arXiv ID、作者或版本。

附件提供的 SAGE 标题线索为 **SAGE: Subgoal-Conditioned Action Generation for Latent World Model Planning**；尚未独立确认该标题对应的论文身份。下面均是“用户要求保留且附件声称的区分”，不是本轮已验证的论文事实：

| 待核验对象 | 附件说明/用户要求 | 后续核验时必须执行的边界 |
| --- | --- | --- |
| Table 1 | state-assisted；生成器与 prior 的低维输入来自环境状态，不只 proprioception | 单独标记观察权限，不能直接当作 RGB-only 基线；原文还须确认状态变量、训练/推理访问范围 |
| Table 18 | visual-only；但仍有额外训练模块 | 与主表分行，不能因 visual-only 就写成零后训练；需核对组件、训练数据与实际测试输入 |
| H | goal offset，不是内部 world-model rollout horizon | 独立保存 `goal_offset` 与 `planning_horizon`；不按符号 H 相同建立预算等价 |
| Table 1 的 ± | 附件声称为三个固定 evaluation manifests 的样本标准差 | 待原文确认分母、聚合与自由度；不改写为 SEM/CI，也不将 manifests 当训练 seeds |
| Table 18 的汇总 | 附件声称为 150 queries 的合并成功率，而非多训练 seed 均值 | 不给该成功率附加主表的误差条，也不从舍入百分比反推重复实验误差 |
| 表号/版本 | 附件称 Table 1、Table 18 | PDF 页码、版本号与表题仍为 unknown；不同版本可能重新编号，不能仅凭附件表号定位引用 |

**不转录附件中的 SAGE 成功率到已核验结果表。** 其观察权限、H 和统计定义虽然是重要防错约束，但具体原文证据链仍缺失。本次没有新证据确认 GRASP/Hi-LeWM/SuperMimic 的原论文或数值，也不把附件对它们的叙述顺带升级为核验完成。

## PRISM / IMWM 本地与官方来源轨迹

本地检索范围为 `/home/gbk/code/LeWorldModel` 与附件文件名，以及本仓库 docs、文献索引与 references.bib 的定向全文关键词：`PRISM`、`IMWM`、`PRior-guided Imagination`、`Intuition Models Complement`。除本地 evidence policy 的候选列表外，未找到原论文或其可核验结果表。本检索不等于逐篇全文穷举所有未知文件名 PDF。

| 对象 | 附件提供的检索线索 | 证据状态 |
| --- | --- | --- |
| PRISM | 标题线索：PRISM: PRior-guided Imagination Sampling in World Models；附件称 Table 1 为 MPPI 与 prior-guided MPPI 的 PushT/Cube 比较 | `attachment_claim_only`；论文身份、表页与数值均未原文确认 |
| IMWM | 标题线索：IMWM: Intuition Models Complement World Models for Latent Planning；附件称 Table 2 为四任务控制成功率比较 | `attachment_claim_only`；论文身份、表页与数值均未原文确认 |
| [官方 arXiv 联合检索入口](https://export.arxiv.org/api/query?search_query=ti:%22Subgoal-Conditioned%22%20OR%20ti:%22PRior-guided%22%20OR%20ti:%22Intuition%20Models%22&max_results=15) | 查询 `ti:"Subgoal-Conditioned" OR ti:"PRior-guided" OR ti:"Intuition Models"`，最多 15 项 | 普通权限 curl exit 6：DNS 无法解析 `export.arxiv.org`；随后请求审批重试，获执行后返回 HTTP 429（curl exit 22），没有论文响应内容 |

官方请求时间为本轮 2026-10-26。HTTP 429 是服务限流/拒绝请求状态，不能解读为论文不存在或查询无结果；本轮不换连接器、代理、镜像或其它端点绕过限流。上述 URL 是检索入口，不是已确认的论文 URL。未猜测 DOI、arXiv ID、项目域名或作者。

## 原论文提取表：外部数值保持 unknown

| 必需字段 | PRISM | IMWM |
| --- | --- | --- |
| 完整标题/作者/唯一论文身份 | 标题仅附件线索；原文身份 unknown | 标题仅附件线索；原文身份 unknown |
| 官方论文 URL、项目 URL、DOI/arXiv ID | unknown | unknown |
| 版本、发布日期/修订日期、会议 | unknown | unknown |
| 原文表号与 PDF 页码 | 附件声称 Table 1；原文表号/页码 unknown | 附件声称 Table 2；原文表号/页码 unknown |
| 真实数值、单位、方向、基线值 | unknown，不转录附件数值 | unknown，不转录附件数值 |
| 是否报告控制性能 | 附件声称有；原文未核实 | 附件声称有；原文未核实 |
| 任务/环境版本、goal offset、初始状态 | unknown；附件仅给 PushT/Cube 线索 | unknown；附件给 TwoRoom/Reacher/PushT/Cube 线索 |
| 数据、训练/验证/测试划分与目标采样 | unknown | unknown |
| frozen world model、encoder、checkpoint | unknown | unknown |
| prior/intuition 组件、参数量、训练成本 | unknown；不把附件的规模描述当确认 | unknown；训练与检索角色待确认 |
| 动作维度、单位、归一化与 action repeat | unknown | unknown |
| RGB/状态/历史/goal 输入与额外信息访问 | unknown | unknown |
| 检索库内容、规模、测试泄漏/重用约束 | unknown | unknown；演示检索权限尤其需核实 |
| rollout horizon、真实执行步数、反馈频率 | unknown | unknown |
| 搜索候选/迭代、K 含义、warm start | unknown；附件的 K/MPPI 轮次仅线索 | unknown |
| predict calls/samples、编码、反向、共享开销 | unknown | unknown |
| wall time、硬件、峰值内存 | unknown | unknown |
| 成功判据、episode 分母、终止条件 | unknown | unknown |
| seeds、误差条、独立统计单位 | unknown；附件称 3 seeds×50 episodes，待核实 | unknown；附件称 12 evaluation cells/任务的均值和 SD，待核实 cells 构成与独立性 |
| 可直接纳入的 `paper_reported` 数值行 | 无 | 无 |
| 与当前 P0/P1 可比性标签 | `insufficient_metadata` | `insufficient_metadata` |

附件对 IMWM/Reacher 的统计优势判断也不是本轮核验结论：未取得逐 seed/cell 数据与配对结构，不能仅凭两个均值和 ± 推出显著或不显著。不同论文中的 SD/SEM/合并成功率、不同随机单位与训练 seeds 必须分别保存；也不能把 SAGE 的 H 定义推广到 PRISM/IMWM。

来源质量：两者均为 `UNVERIFIABLE_THIS_RUN`，不是 `FABRICATED`。设计证据等级、同行评审、作者身份、利益冲突、出版信誉、撤稿状态均为 unknown；外部独立交叉验证覆盖为 0。内部附件是检索线索，不能因写有“直接转录原文”而获得一手来源身份。

## 与本地 D2P 控制性能的比较条件

本地已完成的两根 Reacher P0 calibration 收据来自 seeds `20261040/20261041`，是 calibration-only，不是独立 outcome。既有独立分析显示默认 D2P/CEM predict samples 为 536/256，sample-matched 为 536/536，但 calls 仍是 122/8。不得把 PRISM 的候选 K、迭代数或 IMWM 的 cells 直接映射为这些 predict 资源；更不能据此声称全算力匹配。

PRISM 若确实额外训练 prior，需在原文确认后分列训练与推理预算，不得把它当作 D2P 零后训练同资源方法；也不得因其额外训练而删除必要性能参照。MPPI 名称相同不代表 cost、动作分布、反馈频率和 stopping rule 相同。

IMWM 若确实使用演示检索和 intuition model，必须对齐可访问库、动作片段/状态信息、数据划分与缓存成本。否则它与未获同等记忆访问的 D2P 只能是 contextual reference，不能计算公平的跨协议提升率。

当前 Reacher P0 的固定两 token 动作执行、P1 的首段后真实 RGB 重编译，以及 PushT 的 known-root fixed-prefix suffix intervention，分别回答不同问题；外部整任务成功率不能替代本地非贪心机制与未来能力改变的配对证据。外部设置未知时，本文件只判“兼容性未证”，不擅自认定外部用了不同设置。

后续仅在任务、split、模型/checkpoint、观察、动作、goal offset、规划 horizon、真实执行长度、反馈、预算、成功标准与统计单位逐项证实一致后才标 `direct_comparable`；原文明确不同则标 `paper_reported/context_only`。缺字段维持 unknown，不为凑比较表填 0。

## 交付判定

已完成：读取附件；将 SAGE 两表、观察权限、goal offset 与误差统计差异拆成待核验约束；进行 PRISM/IMWM 本地检索与一次按权限升级的官方检索；记录真实访问失败。

未完成：原论文身份/版本、原文表页与数值核验。最窄支持结论是来源缺口已明确，不能声称不存在可比性能；最强替代解释仍是访问失败导致原文未取得。下一项必需证据是官方论文 PDF/稳定 URL/版本及实验附录；本文件只适合 provenance 附录或 artifact，不适合主文性能排名或提交就绪宣告。

AI 辅助披露：Codex 使用 ARS 来源核验流程做本地读取与记录，没有以附件替代原文，没有上传私有材料到外部模型，没有运行训练、神经推理或 GPU，也未下载权重。
