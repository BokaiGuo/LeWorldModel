# GRASP / Hi-LeWM 公开报告值来源核验（2026-10-26）

## 结论：原文未取得，不录入性能数值

本轮按 `academic-research-suite` → `deep-research/fact-check` → `source_verification_agent` 执行定向来源核验。**GRASP 与 Hi-LeWM 的原论文、版本和结果表均尚未取得，已核实的外部 `paper_reported` 性能条目为 0。** 这不是两篇论文不存在、没有结果或没有公开资产的结论，而是本轮来源获取未完成。

不得从项目内部研究计划、聊天摘要或模型记忆转抄成功率、改进幅度、年份和会议身份。既有文字中的“GRASP，ICML 2026”“Hi-LeWM，2026-07”以及 Hi-LeWM 的 PushT 改善描述，本轮均未获一手材料确认，不能作为已验证引文或表格数据。本文也不据此断言这些描述错误。

本文仅是缺失项明确的来源审计产物，不是复现结果、完整文献综述或投稿就绪认证；不产生 efficacy、完整 compute matching 或 P1 准入声明。

## 检索与访问轨迹

核验日期：2026-10-26；检索关键词：`GRASP`、`Hi-LeWM`、`HiLeWM`、`hi-lewm`，以及论文/文献索引文件名。优先检查 `/home/gbk/code/LeWorldModel` 中命名相关的论文、Markdown 文献索引、BibTeX 和文献 JSON，并检查本地附件文件名。未对所有未知标题 PDF 逐篇全文搜索，因此“未找到”仅指本次定向检索范围。

| 来源/检索入口 | 本轮观察 | 能支持什么 |
| --- | --- | --- |
| `LITERATURE_EVIDENCE_MAP_20261013.md`、`docs/PUSHT_LITERATURE_TO_PROTOCOL_20261019.md`、`docs/VERIFIED_LOCAL_TRANSFER_LITERATURE_20261019.md` 等本地索引 | 未定位到两篇目标论文的可核验原文/表格 | 不能填外部数值；不等于穷尽全部本地文件 |
| `docs/D2P_BASELINE_FAITHFULNESS_AUDIT_20261026.md` | 有内部实现与外部基线缺口审计，但明确未核实官方 GRASP 原文/源码 | 仅作本地比较边界线索，不是外部论文证据 |
| 原始提案 `/home/gbk/.codex/attachments/861f6764-1b25-4227-bf15-f1a237ba7d8f/pasted-text-1.txt` | 只有简称、年份/会议描述和定性说明；相关位置没有可据此确认身份的论文 ID/结果表 | 仅作检索线索，不升级为引用 |
| [arXiv API：Hi-LeWM](https://export.arxiv.org/api/query?search_query=all:Hi-LeWM&max_results=5) | 普通权限 `curl` 返回 exit 6：`Could not resolve host: export.arxiv.org`；之后按权限流程请求审批重试 | 未返回元数据；不能确认标题、作者、arXiv ID、版本或结果 |
| [arXiv 官方检索：GRASP planning](https://arxiv.org/search/?query=GRASP+planning&searchtype=all&abstracts=show&order=-announced_date_first&size=50) | 普通权限 `curl` 返回 exit 6：`Could not resolve host: arxiv.org`；之后按权限流程请求审批重试 | 未返回候选列表；不凭 GRASP 缩写猜论文身份 |

以上两个 URL 是**检索入口，不是已确认的论文 URL**。没有猜测项目域名、DOI 或 arXiv ID；没有通过代理、其它连接器或替代传输绕过网络权限。审批结果见文末访问状态。

## 原论文报告值与协议提取表

“缺失”表示本轮未取得可核验原文，不表示作者没有报告。性能缺失绝不按 0 填写。

| 必需字段 | GRASP | Hi-LeWM |
| --- | --- | --- |
| 完整标题、作者 | 缺失：简称不足以唯一定位 | 缺失：未核实完整标题/作者 |
| 论文/项目官方 URL、DOI/arXiv ID | 缺失 | 缺失 |
| 论文版本号、提交/修订日期、会议身份 | 缺失；不确认内部笔记的 ICML 2026 标签 | 缺失；不确认内部笔记的 2026-07 标签 |
| 主表/附表编号、PDF 页码 | 缺失 | 缺失 |
| 真实报告数值、单位、基线对照值 | 缺失，未录入 | 缺失，未录入 |
| 任务、环境版本、难度/目标分布 | 缺失 | 缺失；不能仅因内部笔记提到 PushT 就认定同一 benchmark |
| 训练数据与测试/held-out 划分 | 缺失 | 缺失 |
| 模型、checkpoint、额外学习组件 | 缺失 | 缺失；额外学习组件/成本需原文确认 |
| 动作维度、单位、归一化、action repeat/chunk | 缺失 | 缺失 |
| 观察：RGB/历史/目标/额外状态访问 | 缺失 | 缺失 |
| rollout horizon、执行步数、重规划频率 | 缺失 | 缺失 |
| 候选数、优化迭代、重启、模型调用预算 | 缺失 | 缺失 |
| 编码、反向、共享编译、墙钟等开销 | 缺失 | 缺失 |
| 成功标准、终止条件、评估分母 | 缺失 | 缺失 |
| seed 列表/数量、测试 episode 数 | 缺失 | 缺失 |
| 误差条类型、独立统计单位、重复次数 | 缺失；不能补作 SD/SEM/CI | 缺失；不能补作 SD/SEM/CI |
| 可纳入的 `paper_reported` 行 | 无 | 无 |

来源质量判定：两者均为 `UNVERIFIABLE_THIS_RUN`，而非 `FABRICATED`。原文不可用，故设计证据等级、方法严谨性、同行评审、作者身份、利益冲突、撤稿状态均未评定；不能凭不可访问认定掠夺性出版或来源不可信。内部计划仅属检索线索，不满足结果声明的证据要求。外部独立交叉验证覆盖率为 0，不虚称双源核验通过。

## 与当前 Reacher / PushT P0/P1 的比较边界

这里区分**已核实的本地接口事实**与**待核验的外部协议**。外部字段缺失时只能判定“不可直接比较/兼容性未证”，不能具体声称作者用了某种不同设置。

### Reacher P0：当前两份 calibration 报告

直接读取：

- `/home/gbk/code/LeWorldModel/le-wm-main/results/reacher_d2p_p0_predict_accounting_calibration2_v1_20261026_cuda/report.json`
- `/home/gbk/code/LeWorldModel/le-wm-main/results/reacher_d2p_p0_predict_sample_matched_calibration2_v1_20261026_cuda/report.json`

两份均为 seeds `20261040/20261041` 的 `calibration_only`；`future_horizon=2`、`history_len=3`、`action_block=5`、`context_support=uniform`，`qpos_readout_binding.enabled=false`。这些是当前本地报告条件，**不是外部论文设置**。源码 `_terminal_batch` 从实际 root RGB/history 建立预测；P0 整条动作开环执行，物理 qpos distance 是 evaluator 后果指标，不自动等于外部论文成功率。

实际 planning predict 坐标：默认 D2P/CEM 为 calls `122/8`、samples `536/256`；sample-matched 为 calls `122/8`、samples `536/536`。数据见各报告的 `rows[*].methods[*].receipt.actual_planning_predict`。即使后者 samples 相等，也不代表编码、反向、墙钟或物理反馈预算相等。不能拿外部论文的时间/成功率作为这些坐标下的同预算结果。

本地 `run_reacher_d2p_p0_goal_progress.py` 的 `_restart_gd` 是动作参数多起点 Adam 和终点选优；报告名称为 `grasp_style_restart_gd`。**名字含 GRASP-style 不构成官方 GRASP 复现，不能把外部 GRASP 的数值贴到该臂。** D2P 与 local GN 在本地有意使用同求解器，也不构成两个独立外部算法结果。

### Reacher P1：反馈与评价协议需要独立对齐

静态查看 `run_reacher_d2p_p1_reconfiguration.py`：reconfiguration/zero/greedy 经 `_two_stage` 在执行首段后使用真实 successor RGB 重新编译后缀；主 runner 的 `cem_full_horizon` 一次规划后执行全序列。实际反馈路径必须与外部基线分别对齐，不能靠同 horizon 名称确认可比。外部 GRASP/Hi-LeWM 的反馈频率、任务目标、成功标准和完整预算均缺失，因此现阶段不能替代正式同根 P1 机制对照。

### PushT：当前 suffix intervention 不是完整任务成功 benchmark

静态查看 `run_pusht_d2p_nuisance_suffix_bank.py`：声明 known 4 calibration roots、all 16 prefixes、`identity_gn4/nuisance_gn4` 两模式、固定 prefix 后 observed-RGB suffix 重编译；协议预期 `3968` predict samples、`128` physical branches，且 `P1_admission=False`。这些是源码协议声明，本轮没有重跑或重新核验该作业完成结果。

该局部 suffix 干预不能直接对照未确认版本 Hi-LeWM 的长程成功率。即使原文确实报告 PushT，也必须先对齐初始状态/目标分布、评估 episode、观察、动作与 repeat、闭环时长、成功阈值、learned prior 训练成本及测试时资源；不能从同名环境推断兼容。

### 允许与禁止的使用方式

当前只允许将两者列为“待取得一手材料的相关基线”，不提供数值名次、不计算相对增益、不替代本地同协议实验。取得原文后，必须逐条补齐上表并绑定版本/表页；协议不一致的公开值只能标注 `paper_reported/context_only`，不能标注同预算、同模型、同 split 或本项目复现。

最窄支持结论：外部结果来源链尚未闭合；本地计数样本对齐不能填补官方基线身份/协议缺口。仍存的解释是检索/权限限制导致原文未取得，而非论文或结果不存在。缺失的关键证据是原论文可读全文、稳定官方 URL/版本及实验表/附录。本文适合作为 provenance/缺失项附录或 artifact，不适合作为主文性能表；本次核验为 `partial`，不作整个研究成熟度或 submission-ready 宣告。

AI 辅助披露：使用 Codex 与 ARS 来源核验流程进行检索、静态读取与文档整理；未调用外部模型审读，未上传私有实验报告，未运行训练、神经推理或 GPU。

## 网络审批访问状态

Hi-LeWM 官方 API 与 GRASP 官方检索的提权重试均未执行成功：两次自动审批评审均超时（`The automatic permission approval review did not finish before its deadline`）。这不是网络目标不存在或操作不安全的判定。没有将任何请求计作成功访问，也没有绕过审批；继续补齐原文需要用户明确批准官方网络访问，或提供官方论文链接/本地 PDF。
