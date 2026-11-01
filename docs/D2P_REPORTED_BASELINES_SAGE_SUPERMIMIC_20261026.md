# SAGE / SuperMimic 原论文报告值核验（2026-10-26）

## 当前判定

本次扩展遵循 `academic-research-suite` 的 `deep-research/fact-check`、`source_verification_agent` 来源纪律，以及本地 `docs/D2P_BASELINE_EVIDENCE_POLICY_20261026.md`：原方案候选不限四篇，本文件仅负责 SAGE 与 SuperMimic，不替代其它候选的核验。

**目前未取得这两篇目标原论文，外部可录入的真实性能数值为 0 条；没有确认可直接比较的 Reacher/PushT P0/P1 控制性能。** 后一句不是“论文没有控制性能”，而是本轮无法核验。不得以模型记忆、内部提案、同名论文、搜索摘要或第三方转述填表。缺失数值为 `unknown`，不是数值 0。

GRASP/Hi-LeWM 的当前来源获取缺口仍保留在 `docs/D2P_REPORTED_BASELINES_GRASP_HILEWM_20261026.md`；该缺失清单完成不等于原论文核验成功。本次扩展不取消或覆盖其待核验事项。

## 来源与访问记录

检索日期：2026-10-26。先对 `/home/gbk/code/LeWorldModel` 的 docs、本地文献索引和 BibTeX 搜索 `SAGE`、`SuperMimic`，并对本地仓库与 `/home/gbk/.codex/attachments` 的文件名进行不区分大小写检索。未找到以这两项命名的本地论文；未对所有未知标题 PDF 全文穷举，因此不声明本机绝对不存在副本。

| 检索来源 | 实际发现 | 证据权限 |
| --- | --- | --- |
| 原始提案 `/home/gbk/.codex/attachments/861f6764-1b25-4227-bf15-f1a237ba7d8f/pasted-text-1.txt` | 第 424 行有 SAGE 的多时域子目标/条件动作 proposal 描述；第 53、661 行附近有 SuperMimic 的时间标签/共享动力学描述 | 仅作检索线索；没有可核验论文 ID、表页或性能值。不能把描述作为外部已证事实 |
| `docs/D2P_CVPR_CRITICAL_PATH_20261026.md` | 把 Hi-LeWM/SAGE 列为待核实资产与强比较 | 项目计划，不是一手性能证据 |
| `docs/D2P_BASELINE_FAITHFULNESS_AUDIT_20261026.md` | 区分外部 `paper_reported` 与必须本地验证的同协议证据 | 使用边界，不确认 SAGE 原文身份/结果 |
| `docs/D2P_BASELINE_EVIDENCE_POLICY_20261026.md` | 将 SAGE 列为子目标接口候选、SuperMimic 列为共享可微动力学候选 | 覆盖范围，不确认实际原论文设置或可比性 |
| [官方 arXiv 联合检索入口](https://export.arxiv.org/api/query?search_query=all:SuperMimic%20OR%20%28ti:SAGE%20AND%20all:planning%29&max_results=15) | 普通权限 `curl` exit 6：`Could not resolve host: export.arxiv.org`；按要求请求审批后重试 | 尚无成功返回的论文元数据或正文。该 URL 是检索入口，不是论文 URL |

联合查询为 `all:SuperMimic OR (ti:SAGE AND all:planning)`，最多 15 条。SAGE 是可能重名的简称，查询本身不保证唯一身份，也不构成全量覆盖；即使将来返回同名论文，仍须核对作者、任务和方法。未猜测项目网站、DOI、arXiv ID、完整标题或作者；未通过其它连接器、代理或传输方式绕过权限。

## 原论文逐项提取：不得用 unknown 构造性能排名

| 字段 | SAGE | SuperMimic |
| --- | --- | --- |
| 完整标题、作者、唯一论文身份 | unknown | unknown |
| 官方论文 URL、项目 URL、DOI/arXiv ID | unknown | unknown |
| 版本号、发布日期/修订日期、会议 | unknown | unknown；不确认提案的 2026-10 时间标签 |
| 结果表号、主文/附录、PDF 页码 | unknown | unknown |
| 控制性能数值、指标方向与单位 | unknown；不录入定性“显著改善” | unknown；不将“多任务复用”改写为成功率 |
| baseline 数值、比较模型/方法 | unknown | unknown |
| 任务族、Reacher/PushT 环境版本 | unknown | unknown |
| 数据规模、训练/验证/测试划分、目标分布 | unknown | unknown |
| 世界模型/encoder/checkpoint 与冻结状态 | unknown | unknown |
| learned subgoal/proposal/policy/adapter 与训练成本 | unknown；提案提及不能替代原文 | unknown |
| 动作维度、物理单位、边界、归一化、action repeat | unknown | unknown |
| RGB/状态/历史/目标观察及额外信息访问 | unknown | unknown |
| 想象 horizon、真实执行步数与终止条件 | unknown | unknown |
| MPC/反馈频率、chunk 长度、开闭环 | unknown | unknown |
| 候选数、梯度步、重启、编译/共享缓存成本 | unknown | unknown |
| predict calls、batch samples、编码与反向预算 | unknown | unknown |
| wall time、硬件和峰值显存 | unknown | unknown |
| 成功阈值、episode 分母与失败计法 | unknown | unknown |
| seeds/重复数、误差条 SD/SEM/CI 及统计单位 | unknown | unknown |
| 原文是否提供控制性能 | unknown：未取得原文，不能断言有或没有 | unknown：未取得原文，不能断言有或没有 |
| 当前是否存在已核验可比控制性能 | 未建立，`insufficient_metadata` | 未建立，`insufficient_metadata` |

两篇候选均为 `UNVERIFIABLE_THIS_RUN`，不等于 `FABRICATED`。设计证据等级、同行评审、作者身份、撤稿、利益冲突与出版渠道可信度均为 unknown；未进行 S2/DOI 成功匹配或外部独立交叉验证。不得把检索失败说成论文不存在，也不得把内部提案升级为主论文数据。

## 与当前 D2P P0/P1 的可比性

本节是**纳入条件**，不是对外部论文实验设置的猜测。当前本地接口边界见已完成的计数分析、本地基线审计，以及上一个 GRASP/Hi-LeWM 核验文档。外部字段均未取得，故不能声称已知哪一具体外部设置与本地不同；只能记录兼容性未证。

### SAGE：子目标性能与可执行接口机制分开

若原文确认它使用额外学习的 subgoal/proposal，其训练数据、权重和训练成本必须与冻结 D2P 分开披露；不能为了补对比静默训练新模块，也不能仅因使用额外训练就删除候选。当前关于这些组件的描述仍只是提案线索，待原文确认。

需核验其是否在同一 Reacher/PushT 任务、同一 checkpoint、相同观察/动作接口、目标分布和执行预算下报告控制成功率。即使原文有整任务成功率，也不能自动替代本地 P1 的“暂时非贪心首段→未来能力改变→物理后果”配对机制证据。外部 learned-subgoal 性能不等于本地 arbitrary-latent-waypoint 控制已经实现。

### SuperMimic：共享动力学描述不等于同任务基线

必须由原文确认其任务类型、是否视觉世界模型、动作/状态访问、模型训练、梯度控制与评价指标，不能把提案中的“共享可微动力学”“短时想象”当作已确认方法身份。若实际任务是其它控制或动作模仿，只能在核实后作 `paper_reported/context_only` 参照，不能把模仿误差、reward、tracking 或视频质量当作 PushT 成功率。

### 当前本地资源坐标不能被公开论文替代

已完成的 Reacher 两根 calibration 分别来自：

- `le-wm-main/results/reacher_d2p_p0_predict_accounting_calibration2_v1_20261026_cuda/report.json`
- `le-wm-main/results/reacher_d2p_p0_predict_sample_matched_calibration2_v1_20261026_cuda/report.json`

本地独立分析记录：默认 D2P/CEM 的 planning predict samples 为 536/256；第二份为 536/536，但 calls 仍是 122/8。它们不是 outcome，不构成稳定 efficacy 或完整 compute matching。公开论文未核验的“预算”“步数”不能映射到这些数字，预测样本相等也不能填补编码/反向/反馈差异。

当前 PushT suffix bank 是 known-root 固定 prefixes 后的局部 observed-RGB suffix 干预；不是任意长时域整任务成功率 benchmark。Reacher P1 主 runner 的两阶段 observed-RGB 重编译与 full-horizon CEM 的反馈路径也必须单独核对。外部报告值只有在任务、split、metric、模型、动作、观察、反馈、执行长度、预算和统计单位逐项确认一致后，才可标 `direct_comparable`。当前两项都不满足该条件。

## 产物用途与缺口

本次最窄结论：候选覆盖已扩展到 SAGE/SuperMimic，来源缺口明确保留，但没有新增任何已核实外部控制性能值。当前最强替代解释是来源访问限制而非论文没有性能。缺的仍是可读原文、官方唯一身份/版本、表页及实验附录。

本文件属于 `partial` provenance 审计/附录材料，不进入主文性能排名，不触发本地准入，不支持 D2P 超过这些方法的宣告；不对整体研究成熟度作评级。后续不得把该文档的存在计成外部基线核验完成。

AI 辅助披露：Codex 按 ARS 流程做本地检索与缺项记录，未上传私有材料给外部模型，未运行训练、神经推理、GPU 或下载权重。本轮仅新增本文件，不修改现有 runner/准入代码或其它文档。

## 网络审批状态

普通权限官方 arXiv 请求 DNS 失败后，已请求审批重试；自动审批评审超时，工具返回 `The automatic permission approval review did not finish before its deadline`，未成功执行外网检索。不能把请求计作访问成功，也不据此判定目标不存在或操作不安全。GRASP/Hi-LeWM 前轮的两次审批均超时，原文未取得的状态没有被本次扩展覆盖。补齐数值需要明确批准官方网络访问，或提供原论文 PDF/官方链接；不绕过权限另取来源。
