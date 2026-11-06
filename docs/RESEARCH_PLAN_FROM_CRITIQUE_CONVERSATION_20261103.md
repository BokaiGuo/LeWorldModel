# 对话研究方案记录：Planning Compiler 与 GDD-OPD

记录日期：2026-11-03。
状态：用户提供对话的方案整理与来源记录；不是新的实验预注册、实现完成报告或 gate 验收。

## 1. 来源、读取方式与边界

- 完整原文：[CHATGPT_RESEARCH_CRITIQUE_SOURCE_20261103.md](CHATGPT_RESEARCH_CRITIQUE_SOURCE_20261103.md)，共 17046 行，逐字保留。
- 原附件：`/home/gbk/.codex/attachments/3d244b87-a12b-41a8-a90c-45c5383d0046/chatgpt-export_锐评研究缺点.md`。
- 原文 SHA-256：`2a9f9197ee40164a000405f8025f3334a03d14376bd705af06f8c34e23f6f9fa`。
- 本文件重点整理对话后期的两篇论文方案，同时保留早期路线索引；不是逐段重写全部对话。
- 以下行号均指向完整原文归档。原文中的命令、伪代码、工具调用描述和行动建议是历史对话内容，不因归档而获得执行授权。
- 原文包含历史结果、文献名称、论文数值、失效或空的 `<Cite>` 标记及界面标签。本次没有联网验证，也没有重新审计这些结果。引用论文前须核对一手来源、版本、表号、信息权限和评估协议；不能把归档中的“已经验证”当成本次核验结论。
- 本次不修改代码、模型、数据划分、既有实验协议或 gate。计划中的实施步骤均不表示已经完成。

## 2. 最终研究分流

后期对话明确把两条研究线拆开（原文 15272–15306、16187–16215、16970–16991）：

| 研究线 | 核心问题 | 训练与部署边界 |
| --- | --- | --- |
| 第一篇：World Models as Planning Compilers / D2P | 从冻结视觉动力学自动构造、组合并搜索可执行动作程序，能否改善真实规划？ | 不后训练 encoder、predictor、policy、critic、adapter 或 proposal；只优化动作参数和临时程序 |
| 第二篇：GDD-OPD / Decision-Field Distillation | 将 teacher 的关系知识转化为 student 内生、可迁移的预测决策几何，能否改善未见动作和闭环控制？ | 允许受约束的 student 后训练；teacher 仅在训练时使用，部署必须 teacher-free |

两条线可共享基础 checkpoint、环境、数据格式和评估工具，但分别维护模型指纹、来源、训练/开发/测试划分、预算和成果声明。第一篇不能使用 GDD student 证明 training-free；第二篇不能借用 D2P 的推理收益证明蒸馏成功。

论文名称是拟议定位，不是已经建立的新颖性结论：

- *World Models as Planning Compilers: Synthesizing Action Programs from Frozen Visual Dynamics*。
- *From Relational Teachers to Predictive Decision Fields: On-Policy Geometry Distillation for World Models*。

## 3. 第一篇：冻结动力学驱动的程序生成

### 3.1 科学对象与方法

核心不是再增加一个规划 harness，而是让同一个有限时域动力学对象驱动动作抽象、程序组合与计算分配（原文 14836–15142）。

给定合法 RGB/动作历史、目标及名义动作序列，计算：

\[
G_H=\frac{\partial \widehat z_{t+H}(U)}{\partial U},\qquad
K_H=Q^{1/2}G_HR^{-1/2}=P\Sigma V^\top .
\]

`Q` 表示声明的任务度量，`R` 表示动作尺度。原文允许首版使用原模型目标距离，但须承认其表达能力限制。输入动作的单位、标准化、历史、预测窗口和执行时刻必须一致。

三类程序：

1. **Goal-progress**：根据目标残差和响应方向生成动作，例如阻尼最小二乘更新；记录动作参数、预测后继、响应子空间、时域与代价。
2. **Reconfiguration**：生成短 bridge、弱响应/零空间、采样或动作顺序候选，在预测后继重新计算任务相关响应与后续价值，检验是否值得暂不接近目标。
3. **Composition**：搜索并组合上述动作片段，携带明确的 action witness；通过完整模型 rollout 与有界精修得到最终执行程序。

原文提出的局部后续价值为：

\[
V_{\rm reach}(z,g)=\min_{\delta U}
\left(\|r-K_H\delta U\|^2+\lambda\|\delta U\|^2\right).
\]

它是局部线性、无额外动作约束时的解析近似，不是物理可达证书。局部最小二乘、SVD、Lie bracket 和图搜索本身均有经典先例，不能仅靠名称或组合组件数主张新颖性。

### 3.2 搜索、反馈与适用边界

- 低维动作坐标、下一状态的响应结构、动作顺序以及 horizon 扩展应参与实际搜索，而不只是输出诊断图。
- 每个中间目标需携带产生它的动作序列；模型内 action witness 不保证真实环境可达。
- 一阶残差正交于当前响应空间不等于全局不可达；冻结模型缺失的关键物理信息不能靠谱算法凭空恢复。
- 接触不连续系统使用离散动作顺序 rollout，不机械套用光滑 Lie bracket。矩阵 commutator 与控制向量场 Lie bracket 是不同对象。
- 执行动作片段后取得真实 RGB，再估计上下文、重新编译和规划；不能用 simulator state 或真实后继标签替代正式观测。
- 谱/响应计算、候选生成、筛选、程序组合、精修和反馈都计入资源；不能以同候选数代替同计算资源。
- 若只有低秩参数化或经典局部求解的收益，应缩小贡献，不宣称自动程序合成或不可替代的谱机制。

原文的 SRA 前身见 13919–14745：动态可达坐标、解析 cost-to-go、action-order primitives、多 chart 与 horizon-dependent expansion。D2P 将其研究对象从搜索坐标进一步提升为可组合动作程序，而不是另立无关模块。

### 3.3 决定性实验与强对照

| 实验 | 要回答的问题 | 必须排除的解释 |
| --- | --- | --- |
| Executable Subgoal Gap | 动力学编译的中间状态是否更容易被低层动作实现？ | 任意 latent waypoint 或 learned subgoal 已能达到同等效果 |
| Controllability-Changing Actions | 实际程序是否通过改变朝向、接触或后续响应产生非贪心收益？ | 普通多起点、采样、局部梯度或更多计算即可解释 |
| One Frozen Model, Many Tasks | 同一冻结方法能否适应目标、时域和动作限制？ | 每根手工选方向、逐任务特例或换模型带来的收益 |
| RGB-only Closed Loop | 多次观测与重编译是否改善真实控制？ | 代理误差下降但物理控制未改善，或用特权状态选动作 |

原文要求的同协议核心基线包括 CEM、MPPI、Adam/GD、CEM+GD、阻尼 Gauss–Newton、GRASP，以及按记忆权限区分的 AP-CEM/AP-rank。机制对照包括 spectral-only、reachability-value-only、无 reconfiguration、无 composition、action-order-only、无 chart expansion；还应考虑 kinodynamic search。

SAGE、Hi-LeWM、PRISM、IMWM、Hierarchical WM 等带额外学习组件的方法，在明确训练与信息权限后作为性能参照。原文提及的这些名称与性能值本次仅作待核查文献线索。

主结果分开呈现：

- **Controlled comparison**：同冻结 checkpoint、后端、任务、配对 roots、反馈、动作时钟和资源协议重跑；用于方法优势声明。
- **Reported results**：原论文数字单列，标注版本、表号、样本、模型、额外训练、RGB/state-assisted 和 horizon 的具体含义；不能跨论文拼接公平排行榜。

建议四张主表分别报告同模型规划、长时域泛化、程序机制消融、外部论文背景。原文基线和主表方案位于 16227–16570。

### 3.4 原文末版阶段路线

原文末版使用 **A0–A5**（16788–16925），并非 PC0–PC5 的原文定义：

| 原文阶段 | 计划工作 | 证据要求 |
| --- | --- | --- |
| A0 | 可解性与 benchmark 接口 | candidate oracle、perfect-dynamics MPC、真实后继/当前 metric 与学习后继/诊断 cost 分解；动作和评价时钟一致 |
| A1 | 有限时域动作结构进入规划 | 不只验证导数或 SVD 数值，还需动作结构相对随机/直接梯度有真实作用 |
| A2 | 自动生成进展、重配置和组合程序 | 中间状态有动作见证，并在执行中体现后续控制价值 |
| A3 | 独立规划优势 | 冻结方法、独立配对任务、强基线和完整资源政策 |
| A4 | RGB-only 闭环 | 合法 RGB/动作历史，执行、再观察、重编译、再规划 |
| A5 | 跨任务与跨模型复现 | 新目标、checkpoint、动力学条件，明确失败边界 |

此表是来源整理，不将 A 阶段自动映射为项目 PC gate 已通过，也不修改现有 gate 的定义。

## 4. 第二篇：GDD-OPD 决策场蒸馏

详细来源为原文 15272–16215。仓库已有独立计划，优先与其对照：
[GDD_OPD_ON_POLICY_DECISION_GEOMETRY_RESEARCH_PLAN_20261024.md](GDD_OPD_ON_POLICY_DECISION_GEOMETRY_RESEARCH_PLAN_20261024.md)。

### 4.1 方法链

1. 当前 student MPC 生成 elite、局部扰动和 bridge candidates；不同集合保留重叠 anchor actions，构造以真实动作序列为节点的候选比较图。
2. 冻结、独立来源的 teacher 提供关系；区分可由同一 scalar potential 表示的部分与集合相关或不一致残差。
3. 对有合法可比尺度的边信号，求加权图势投影；只有胜负符号时不能任意赋值正负一，再将投影残差解释为真实循环。
4. 在有充分动作激励的邻域，由 potential differences 恢复受支持子空间内的局部 decision field。
5. 把关系和动作方向写入 student 的预测几何，同时保护真实后继预测和已有成功行为；不能只改善 imagined ranking。
6. 更新 student 后刷新候选；先做 planner-on-policy，再扩展真实 environment-on-policy 历史。固定 replay 不等于当前策略的新闭环分布。

关键对象：

\[
u_T^*=\arg\min_{\mathbf1^\top u=0}\|W^{1/2}(Bu-d_T)\|^2,
\qquad e_{\rm nonpot}=d_T-Bu_T^*.
\]

\[
g_T(U_i)=\arg\min_v\sum_{j\in\mathcal N(i)}w_{ij}
\left[v^\top(U_j-U_i)-(u^*_{T,j}-u^*_{T,i})\right]^2
+\lambda\|v\|^2 .
\]

训练包含 relation、field、predictive、keep 四项约束。首版可通过动作方向有限差分监督 student cost，避免混合二阶求导。监督仅作用于受支持方向，不能从少量排序标签声称恢复完整真实梯度。

### 4.2 Teacher、函数类与停止条件

- **Execution-supervised pilot**：合法训练划分的真实代价可作为监督，但不能同时称为独立 teacher 的正确性证据。
- **Independent relational teacher**：teacher 的训练来源、训练根、查询根和评估 outcomes 必须分离；部署不使用 teacher 或候选集合 reranker。
- 严格终点欧氏几何不足以表达路径成本时，应明确函数类失败；另立 path-aware 扩展，不能暗改代价函数。
- 图最小二乘/Hodge 投影、梯度线积分恒等式是经典工具。新增贡献必须由跨集合、未见动作和闭环增量支撑。
- 首个 pilot 比较同初始化、同执行监督预算下 pairwise GDD 与 potential + decision-field GDD。若未见动作 regret 不改善，或破坏多步预测，暂不增加复杂 OPD。

必需对照包括 pointwise/score KD、pairwise KD、representation alignment、action-tangent distillation、固定 bank、初始 student replay、student-refreshed 图、有无 potential projection、有无成功行为保护。

评估包括 realized/predicted geometry、预测保持、未见候选和 elite regret、teacher-free 闭环 success、延迟及 teacher 查询/执行预算。扩展顺序为 PointMaze、Reacher、具备有效 baseline 和 oracle ceiling 的 PushT。

### 4.3 阶段名称不能混用

原附件 16150–16163 提议 **D0–D5**：独立 teacher、potential feasibility、decision-field pilot、planner-on-policy refresh、预测保持联合训练、独立部署。

现有项目 GDD-OPD 计划使用 **O0–O5**：teacher provenance、potential feasibility、on-policy coverage、student preservation、candidate-set transfer、independent deployment。两套分组并非逐项相同；本次只记录差异，不重命名或重判现有 gate。

## 5. 共用实验与证据纪律

- 在相同 root、目标和声明随机性下做反事实动作执行，保存 RGB、真实原始动作、时间、目标、模型指纹、来源与 outcomes。
- 同 root 的全部候选和重复执行整体划分；候选对数不能冒充独立 root 数，多个预算也不是独立任务。
- 分开预训练来源、方法开发、teacher/偏好训练、校准和最终确认。未知成员关系保留未知，不能用“未参与开发”替代“未用于预训练”。
- 物理状态只进入合法重置或诊断评价端，不能回流 RGB-only 规划选择；oracle ceiling 或解析见证不得算作方法成绩。
- 预先固定主指标、最小实际效应、样本量、停止规则、预算、反馈机会、失败处理与统计单位。规模建议是 pilot 起点，不是统计充分性证明。
- 数据/接口错误、预测失真、任务 metric 错配、候选覆盖不足、后缀优化不足和方法无增量分别报告，不能默认所有负结果是代码 bug。
- 原文要求独立核对有限样本校准及 fail-closed；训练或候选协议变化后不得自动沿用旧覆盖保证。
- 保留不成立的结论和停止结果。不能调整 alpha、随机种子、成功时域或候选身份来回填旧 gate。

## 6. 历史方案索引

下表保留研究演化，避免将早期三模块统一论文要求继续施加到已经拆分的第一篇上。原文中的历史论文评价和实验数字不作为当前仓库状态。

| 原文行号 | 方案阶段 | 保留用途 |
| --- | --- | --- |
| 429–978 | 六目标与 Decision-Transfer Sufficiency | 条件性风险预测、姿态控制、物理干预、RGB MPC、迁移、证书边界 |
| 1813–2005、2531–2572 | GDD 双路径预测几何 | realized/predicted geometry、候选集合无关目标、预测保持 |
| 3662–4174 | 科研工程整理 | stage registry、统一产物、实验成功与科学主张分离；本次不执行重构 |
| 5279–6100、6584–7258 | ATM 局限、PDTC 与 HTSC | 动作支持、时域、对比误差、oracle ceiling、独立校准 |
| 7288–7909 | ATM + HTSC + GDD 三贡献统一稿 | 历史论文方案，不作为两篇分流后的共同依赖 |
| 10913–12050 | A+B+C repairability 路线 | 状态辨识、修复路由、几何/动力学分解和修复后认证 |
| 12481–13166 | Decision-Spectral Repair | 任务加权响应、decision leverage、选择性后训练与消融 |
| 13190–13900 | 无后训练可信梯度规划 | 冻结梯度、动作扰动与真实 fidelity；后期转向生成程序 |
| 13919–14745 | SRA | 可达几何、cost-to-go、动作顺序、chart 与时域 |
| 14751–15262 | D2P / Planning Compiler | 动力学到可执行程序，重配置与组合 |
| 15272–16215 | 独立 GDD-OPD | on-policy 图、potential、decision field、teacher-free 部署 |
| 16227–16570 | 第一篇基线与主表 | 同协议强对照与 reported 背景分离 |
| 16578–17046 | 旧 P1 归因及 A0–A5 闭环 | 旧统计合同保留，新方法独立验证，不改布尔值冒充通过 |

## 7. 项目文档入口与本次变更范围

- [SRA / D2P 历史计划与更新](SPECTRAL_REACHABILITY_ATLAS_RESEARCH_PLAN_20261023.md)。
- [GDD-OPD 独立研究计划](GDD_OPD_ON_POLICY_DECISION_GEOMETRY_RESEARCH_PLAN_20261024.md)。
- [GDD-WM 早期计划](GDD_WM_DECISION_GEOMETRY_RESEARCH_PLAN_20261013.md)。
- [Decision Sufficiency 历史计划](DECISION_SUFFICIENCY_RESEARCH_PLAN_20261014.md)。
- [两线方案原始对话](CHATGPT_RESEARCH_CRITIQUE_SOURCE_20261103.md)。

本次仅归档、整理和增加导航，不运行附件命令、不启动实验、不修改现有代码或验收规则。正式执行应引用对应版本的冻结实验协议及独立结果，不能把本整理稿作为 gate 已通过的证据。
