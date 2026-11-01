# P1 理论与 gate：可证明的条件、反例及最小可检验判据

日期：2026-10-28。角色：独立理论 worker。范围：只读现有提案、合同和保存产物；本轮唯一新增文件为本文。未修改源码、结果、准入、阈值、Q、动作或根；未训练、拟合 Q、运行神经模型、CUDA、物理或网络实验。下述数学构造是证明/反例，不是环境实验或 P1 证据。本文不签发阶段准入。

## 0. 结论先行

1. 原提案不是已经成立的非贪心规划定理。能无条件证明的是给定线性化的 ridge 最优解及其谱表达；从这里到 P1 还需要任务相关能力缺口、可行动作见证、局部近似有效、选择 margin、预测到真实编码的排序迁移，以及编码目标到物理目标的关系。
2. P1 在数学上并非不可能：§7 给出精确动力学下“暂时退步—能力改变—终态受益”的构造。同时，即使模型完美、AD/SVD 正确，也不能保证所有任务存在这种机会、stage+reach 选择一定选中、或有限预算 D2P 一定胜强优化器。§6–7 给出反例。
3. 当前不通过不能归结为一个理论负结果。已有真实接线/协议缺陷，剩余排序反转、度量失配、任务前提和独立性缺口也分属不同层。原 12/76 反转和最新历史 selector 的有害选择必须保留；没有被数学推导消除。
4. 真正的 P1 gate 比“平均预测准确/局部导数正确”强得多，特别是所有 eligible 前缀的机制保持和各必要对照 CI 严格有利。现有假设不能推出这些性质；这说明缺少定理前提，不是放宽 gate、改阈值或更换原研究目标的理由。

## 1. 已读依据与合同身份

### 1.1 指令与原提案

检查了仓库根、`le-wm-main` 及父目录的 `AGENTS.md`，未发现适用于本目录的该文件；根 `.agents/`、`.codex/` 当前为空。遵守本 chat 明示的独占写入与只读限制。

原提案完整 715 行：`/home/gbk/.codex/attachments/861f6764-1b25-4227-bf15-f1a237ba7d8f/pasted-text-1.txt`。关键是 §4.1–4.3、§7 E2 和 §8 P1：模型已经有相应动作动力学；直接局部梯度容易停滞而两阶段动作可达；短动作改变几何，再生成 progress；真实闭环优于单层搜索及普通 latent subgoal。原文也明确动作见证不保证真实可达，经典 DLS 不是新定理，PushT 接触不能机械套连续 Lie bracket。

项目解释与现状：

- `docs/SPECTRAL_REACHABILITY_ATLAS_RESEARCH_PLAN_20261023.md` 的 Planning Compiler / 核心机制与 D2P 分支；不要用旧 PointMaze-SRA 同名 P1 替代 D2P。
- `docs/D2P_RESEARCH_CLOSURE_AUDIT_20261025.md`，包括 2026-10-28 增补；历史“待运行”不得覆盖后续终态。
- `docs/D2P_NATIVE_GEOMETRY_AND_DOWNSTREAM_MAIN_20261028_v15.md`；包含本轮新增 P5 分析。
- `docs/D2P_RESEARCH_REQUIREMENT_EVIDENCE_MATRIX_20261026.md`、`docs/D2P_MECHANISM_CONTRACT_AND_MODEL_COHORT_20261026.md`、`docs/D2P_STATE_NATIVE_SUFFIX_CONTRACT_20261026.md`。
- 当前准入定义 `le-wm-main/experiments/spectral_reachability/d2p_stage_admission.py` 及 `run_reacher_d2p_p1_reconfiguration.py` 的机制分支、calibration binding 与 outcome gate。读取这些定义是确认理论对象，不是代替 MAIN 做代码审计。

### 1.2 “P1 gate”具体指什么：先区分旧 Reacher 与当前 PushT

需区分三层，不能以一个布尔值代替全部科学内容。

| 层次 | 当前要求 | 本文如何处理 |
|---|---|---|
| 原 P1/C2 科学命题 | 合格任务上选中非贪心动作、改变任务相关未来能力、产生真实闭环收益 | §2–8 正式化，给出条件与反例 |
| 旧 Reacher 路线标量阶段门 | 该模块中 `p1_admission` 是上游 P0→P1；`p2_admission` 才消费 Reacher P1 outcome 开 P2 | 不把函数名误读为已证明 P1，也不代表所有现行路线 |
| 当前实验/科学闭包 | 冻结来源/根/输入/动作/度量、独立校准与 outcome、能力评价、强对照资源及原实验族 | 数学命题不替代实际回执或完整研究闭环 |

读取到的 **旧 Reacher** `p2_admission` 要求非 calibration、eligible 数达到预声明最少数、全部 P1 integrity gates 真、选定机制 gate 真，以及必要对照 `zero_prefix_two_stage_d2p`、`greedy_two_stage_d2p`、`cem_full_horizon`（有 source bank 时另有 source-matched greedy）的每项配对 CI 上界严格 `<0`；触零不通过。该 runner 默认最少 eligible calibration/outcome 为 4/12，但应以具体封存合同为准，不将默认值覆盖历史声明，更不能把它套到当前 PushT。

旧 Reacher 机制分支是预声明的三选一：qpos 对同长度 zero、真实 RGB 编码对同长度 zero、或真实 RGB 编码对同库 greedy；代码使用候选首段代价 `>= control - 1e-8` 检验“没有更贪心”。这是其既有容差，不是本文提出的 margin。模型 eligible margin 默认 `1e-6`，也只能继承具体 calibration 封存值。不能结果后换分支。MAIN 指定的原 qpos 分支即 all-model-eligible physical non-greedy、独立最小 cohort 和上述三项统计对照。

**当前 PushT model-library 路线不是这一旧 runner 的一次新 outcome。** 它有 model-only 选择先封存、固定 40 槽/8 context 的模型—历史真实动作同位绑定、strict saved 消费、缺失终态显式保留、任务/metric 及历史独立性合同；最新 selector 报告明确 `fresh_roots=0,fresh_confirmation=false,P1_admission=false`。其工程 consumer 通过不是旧 Reacher P1 标量 gate 通过，也不是当前 PushT 的科学准入。不得从旧 Reacher 默认 eligible 数、qpos 指标或 15 步时域推断 PushT 已满足或应满足同一字段。本文 §2–8 给通用条件，实际开门仍由各路线自己的封存合同决定。

“过不了”至少有四种独立含义：

| 门类 | 失败/阻塞的对象 | 理论能否代替 |
|---|---|---|
| 数据/实现完整性门 | raw/历史/renderer/时刻/数组/事件/schema/loaded model 绑定不一致或缺失 | 不能；数学公式正确不生成真实回执 |
| freshness/校准门 | 独立根身份、历史排重、校准—outcome 分离、预声明最小 cohort 不足；当前部分 P4 缺 loaded bytes/reset ledger | 不能；理论绝不自动认证 fresh，P4 阻塞也不是 P1 理论反例 |
| 真实非贪心机制门 | 事前 selected 前缀的真实非贪心、任务相关后继能力及同根行动链不成立/未知 | 条件定理可说明需测什么，不能代替真实测量 |
| 强基线统计门 | 真实收益、必需对照、资源/反馈公平性、独立配对不确定性未满足 | 存在性证明不保证有限样本 CI，更不保证优于全部实际基线 |

同一轮可以有多个门同时阻塞；一个工程门修复也不会自动解除其余三类门。

特别重要：该标量门检查的是 **模型 declared future-score 改善**，没有单独验证真实后继能力的通用 oracle。即使这些布尔门全真，也不能省略原 C2 的真实能力评价。反之，外加 common-residual 几何诊断不是原 own-state 目标的同义替代。二者都保留，不借其差异改门。

## 2. 真实对象、预测对象与 P1 的精确定义

### 2.1 状态、观察与目标

令真实扩展状态为 \(x\in\mathcal X\)，含物理位置/速度、任务时钟以及重放所需的环境内部状态；若接触缓存或 RNG 会影响后续，就不能假定 public snapshot 已是完整状态。真实转移为

\[
x_{t+1}=f(x_t,u_t),\quad u_t\in\mathcal U_t(x_t),\quad
o_t=\rho(x_t;\nu_t),\quad z_t=E(o_t). \tag{1}
\]

\(\rho\) 是 renderer，\(\nu\) 是外观/绘制协议；\(E\) 是冻结 encoder。随机系统可将噪声实现纳入状态/共同随机数，或另外定义期望命题；本文确定性不等式不能自动覆盖未观测随机扰动。

planner 输入 \(h_t\in\mathcal H\) 为封存 RGB-derived latent 与动作历史，不假设单帧 \(z_t\) Markov。冻结模型 \(\hat f:\mathcal H\times\mathcal U\to\mathcal H\) 和其终态提取 \(\hat Z_H(h,U)\) 预测 \(H\) 步后的编码。真实对应为

\[
Z_H(x,U;\nu)=E\rho(f^{(H)}(x,U);\nu),
\quad L_H^E(x,U)=\|A[Z_H(x,U)-z_g]\|^2. \tag{2}
\]

目标 \(g\) 的真实任务代价 \(\ell(x,g)\) 与成功谓词 \(\operatorname{Succ}(x,g)\) 由各封存环境合同定义；\(z_g=E(o_g)\) 只是其视觉目标。\(A\) 是固定加权算子，\(Q=A^\top A\succeq0\)。代码若把投影矩阵叫 Q，须确认它是 \(A\) 还是 co-metric \(Q\)，不能按变量名多乘或漏乘。MSE 的 `/d` 可吸收到 \(A\)；正则项必须同单位缩放。

### 2.2 前缀、非贪心与可行动作见证

固定同根 \(x_0\)、前缀长度 \(k\)、后缀长度 \(H\)、总真实时间/停止规则、目标、动作约束和观察资源。前缀 \(B\) 到达 \(x_B=f^{(k)}(x_0,B)\)，对照 \(C\) 是合同指定同长度 zero 或同库 greedy。定义

\[
N_\ell(B,C)=\ell(x_B,g)-\ell(x_C,g),\quad
N_E(B,C)=L_0^E(x_B)-L_0^E(x_C). \tag{3}
\]

严格任务非贪心为 \(N_\ell\ge m_N>0\)；当前实际 gate 的“no-more-greedy”及其容差按 §1.2，不被这个分析定义替换。相对初始退步 \(\ell(x_B)-\ell(x_0)\ge0\) 是另一条件，速度/被动运动下不等价于对 zero 退步。RGB 非贪心不等价于物理非贪心。

动作见证为一条完整合法 raw 序列 \(W_B=(B,U_B)\)，带输入历史/模型/renderer/goal/normalizer 身份、各实际动作和状态返回、时间戳/终止、预测 postcondition。模型见证 \(\hat Z(h,W_B)\) 存在，只证明模型内有生成动作；不证明真实 \(Z(x,W_B)\) 接近它。不同 raw、不同历史或不同终态时刻不能继承旧标签。

### 2.3 三种“能力”不要混称

固定真实物理任务，定义 \(H\) 步可达集及 own-state 最优余留代价：

\[
\mathcal R_H(x)=\{f^{(H)}(x,U):U\text{ 全程合法}\},\qquad
J_H^*(x,g)=\inf_{y\in\mathcal R_H(x)}\ell(y,g). \tag{4}
\]

\(J_H^*(x_B)<J_H^*(x_C)\) 是目标相对的有限时域机会改善，不等于完整可控性/rank 增加。它也可能含被动漂移与已有位置优势，故须把几何/控制贡献另列。

给定同一物理任务坐标映射 \(\phi\)、固定残差方向 \(r_*\)、相同合法扰动域 \(D\)、名义后缀 \(\bar U_B\)，令真实任务响应矩阵为

\[
K_B^{\rm task}=D_U\phi(f^{(H)}(x_B,U))\big|_{\bar U_B}\,R^{-1/2},
\quad \Gamma_D(K,r_*)=\|r_*\|^2-\min_{v\in D}\{\|r_*-Kv\|^2+\lambda\|v\|^2\}. \tag{5}
\]

\(R\succ0\) 是动作尺度，\(\lambda>0\) 是固定正则；common-residual 能力改善为 \(\Gamma_D(K_B^{\rm task},r_*)>\Gamma_D(K_C^{\rm task},r_*)\)。这隔离目标方向上的响应，不等价于 own-state \(J^*\)，更不等价于模型 latent 响应。状态依赖的动作域须显式比较，不能暗称相同 \(D\)。接触/像素不光滑时 (5) 改为预声明有限动作响应/非线性可达集，不声称真导数。

最后，给定封存实际后缀算法 \(\pi\)，\(J_H^\pi(x_B,g)\) 为其真实闭环代价；实际收益

\[
J_H^{\pi_B}(x_B,g)<J_H^{\pi_C}(x_C,g) \tag{6}
\]

只比较所执行策略，不证明 \(J^*\) 的次序，更不证明几何改善。公平的强度/反馈/预算控制是解释 (6) 的必要条件。

P1 需要上述要件在 **同一被事前模型选择的根/前缀** 上相连，不能在 A 根展示非贪心、B 根展示能力、C 根展示成功。原提案要求的“模型已有动作动力学”，应操作化为与此动作域/模式/历史/任务方向有关的预测与排序条件，而非全局 MSE 看起来不大。

### 2.4 存在性的必要充分条件，不是任意任务保证

固定一个根、合同参考 C、物理目标和严格 margin。假设合法前缀/后缀集合紧、真实有限时域转移和代价连续，故 (4) 的 inf 可取到；离散有限模式也满足这一有限候选版本。令 \(\mathcal N\) 为符合所需真实非贪心条件的合法前缀集合，\(\mathcal K\) 为符合预声明真实能力改善条件的前缀集合。若参考后缀策略的实际终态代价为 \(j_C\)，则存在带合法动作见证、满足这些机制条件、且终态代价比该参考至少严格改善 \(m_T\) 的程序，当且仅当

\[
\boxed{\exists B\in\mathcal N\cap\mathcal K:\ J_H^*(x_B,g)<j_C-m_T.} \tag{E}
\]

证明：正向把实际见证后缀代价作为 \(J_H^*\) 的上界；反向为该 B 取达到最小值的合法后缀，即得到见证。严格不等式下即使 inf 不取到，也可用足够接近 inf 的后缀证明存在；若改成非严格成功边界，则需要可取到或边界处理。对多个固定实际参考，右侧用 \(\min_C j_C\)。若要严格胜每个前缀的**最优**后缀，必须把这些 \(j_C\) 换成真实 \(J_H^*(x_C,g)\)，不能拿一次弱求解充当它。

(E) 精确描述**真实系统内存在**的条件，不是可直接计算的 evaluator，也不证明模型找得到 B、有限 solver 找得到后缀、selector 选中 B，或 IID outcome 上统计显著。若 \(\mathcal N\cap\mathcal K\) 为空或所有参考已达物理最优下界，机会不存在；即使非空仍需终态 separation。局部谱信息只能在另加 §3–5 的误差/约束条件后提供局部证书，不能替代 (E) 的全局真实量。

## 3. 局部谱到底证明什么

### 命题 1：ridge 公式成立，但投影必须用正确的列空间

固定 \(h,\bar U,H\)，令 \(G=D_U\hat Z_H(h,\bar U)\)、\(K=AGR^{-1/2}\)、\(r=A(z_g-\hat Z_H(h,\bar U))\)，\(v=R^{1/2}(U-\bar U)\)。无动作约束的局部目标是

\[
V_\infty(r,K)=\min_{v\in\mathbb R^m}\{\|r-Kv\|^2+\lambda\|v\|^2\}. \tag{7}
\]

令 \(K=P_q\Sigma_q V_q^\top\) 是非零奇异值的紧秩 SVD，秩为 \(q\)。其唯一解和值为

\[
v^*=V_q\operatorname{diag}\!\left(\frac{\sigma_i}{\sigma_i^2+\lambda}\right)P_q^\top r,
\quad V_\infty=\|(I-P_qP_q^\top)r\|^2+
\sum_{i=1}^q\frac{\lambda}{\sigma_i^2+\lambda}(p_i^\top r)^2. \tag{8}
\]

证明：梯度为 \(2(K^\top K+\lambda I)v-2K^\top r\)，Hessian 正定，解正规方程；把 \(r\) 分为 \(\operatorname{range}(K)\) 和正交补，逐奇异方向完成平方即可。\(q=0\) 时 \(v^*=0,V_\infty=\|r\|^2\)。这是经典 ridge，不含真实排序/全局可达结论。

原文 \(r_\perp=(I-PP^\top)r\) 的四个容易混淆的情形：

- full SVD 的 \(P\in\mathbb R^{d\times d}\) 为整个正交基，则 \(PP^\top=I\)，任意 \(r_\perp=0\)，即使 \(\operatorname{rank}(K)\ll d\)。这不能当“无不可达方向”。
- reduced SVD 也未必是紧秩 SVD：零奇异值对应列仍可能保留；直接投影会把左零空间误算为可达。若在 (8) 求和值中正确保留零值权重 1，总 ridge 值仍可正确，但单独 \(r_\perp\)/\(\chi\) 会错。
- 截断到前 \(s<q\) 列 \(P_s\) 的投影只表示**保留程序库**之外的残差，不表示原动力学不可达。被丢弃方向可能小但非零，也可能是目标关键方向。
- 正则化响应 \(H_\lambda=K(K^\top K+\lambda I)^{-1}K^\top\) 的特征值为 \(\sigma_i^2/(\sigma_i^2+\lambda)\)，一般不幂等；\((I-H_\lambda)r\) 是 shrinkage 后残差，不是正交不可达分量。ridge 最小值是 \(r^\top(I-H_\lambda)r\)，不是简单的 \(\|(I-H_\lambda)r\|^2\)。

若要讨论截断误差，同一 \(K,r\) 下截断最优值比完整值增加
\(\sum_{i>s}\sigma_i^2(p_i^\top r)^2/(\sigma_i^2+\lambda)\ge0\)。故必须保存 full/reduced/retained basis、rank 规则、阈值与 \(\lambda\)，不能只报“rank 非零”。

具体到 MAIN 本轮只读核查：Reacher preflight `_reachability_value` 约 line311 使用 `full_matrices=False`，thin basis 包含 null 奇异方向。这里 **无界 DLS 的 V 公式已计入零 sigma，不能因单独 r_perp 看起来小/为零就判 V 错误**；应把投影诊断的 null-basis artifact 与 objective value 分开。本文没有独立复现该函数 bug，也不据一般 full-SVD 反例推定当前实现出错。例如 \(K=\operatorname{diag}(1,0),r=(0,1)\)，thin/full 的完整二维 P 均给 r_perp=0，但零 sigma 的 ridge 项恰贡献 1，正确值仍为 1。只有将该 r_perp 单独当“全部不可达残差”解释时，语义才不成立。

### 命题 2：动作界与非线性余项的证书

真实可行动作扰动域为紧集 \(D\)，\(\|v\|\le a\)。有界目标 \(V_D\) 是 (7) 把域改为 \(D\) 的最小值。总有 \(V_D\ge V_\infty\)；仅在无界唯一解属于 \(D\) 时取等。坐标旋转后的盒/球不能任意当成另一独立盒，tanh logits 的界也不等于 raw 动作界。

假设加权终态映射 \(Y(v)=A\hat Z_H(h,\bar U+R^{-1/2}v)\) 在整个合法 trust domain 内导数 Lipschitz 常数为 \(M\)。Taylor 定理给

\[
Y(v)=Y(0)+Kv+w(v),\qquad \|w(v)\|\le \eta=\tfrac12Ma^2.
\]

令 \(b=\|r\|+\|K\|a\)。对每个 \(v\in D\)，展开平方得

\[
\left|\|r-Kv-w(v)\|^2-\|r-Kv\|^2\right|
\le 2b\eta+\eta^2=:\omega. \tag{9}
\]

正则不变，取 inf 得 \(|V_D^{\rm nonlinear}-V_D|\le\omega\)。若数值求解 gap 至多 \(\xi\)，其返回动作的真实**模型非线性正则目标**至多 \(V_D+\xi+\omega\)。证明只用了逐点统一界与 \(|\inf a-\inf b|\le\sup|a-b|\)。它不自动界定物理误差；还需 §4。真实接触切换、raster 跳变或越出 trust domain 时 \(M\) 未知/不存在，此证书为 unknown，不是把 \(\omega\) 设 0。

只检查模型 AD 与同模型 FD，验证的是同一 \(\hat f\) 的导数实现；不能提供真实 \(f\) 的 \(M\)、响应或误差界。

## 4. 冻结预测到真实 RGB 排序的误差界

### 定理 3：固定动作对的精确 margin 分解

对同一根/目标/时刻/权重下两条固定见证 \(i\in\{a,b\}\)，令预测加权残差 \(\hat r_i=A(\hat z_i-\hat z_g)\)、真实编码残差 \(r_i=A(z_i-z_g)\)，误差 \(e_i=r_i-\hat r_i\)。若目标编码完全共用则目标误差为零；否则 \(e_i\) 必须包含目标编码差。

定义较小成本较优的 margin \(\hat d=\|\hat r_b\|^2-\|\hat r_a\|^2\)、\(d=\|r_b\|^2-\|r_a\|^2\)。精确恒等式为

\[
\boxed{d-\hat d=2\langle\hat r_b,e_b\rangle-2\langle\hat r_a,e_a\rangle+
\|e_b\|^2-\|e_a\|^2.} \tag{10}
\]

若事前可保证 \(\|e_i\|\le\varepsilon_i\)，令
\(b_i=2\|\hat r_i\|\varepsilon_i+\varepsilon_i^2\)。由 Cauchy–Schwarz，

\[
|d-\hat d|\le b_a+b_b,\qquad
\boxed{\hat d>b_a+b_b+m\Rightarrow d>m.} \tag{11}
\]

证明：对 \(r_i=\hat r_i+e_i\) 展开平方并相减得到 (10)，再分别上界。反向非贪心比较把候选和对照互换即可；每个 eligible 首段都须满足对应条件，不能用终态 margin 替首段 margin。

共同偏置不会一般相消：写 \(\hat r_a=v,\hat r_b=v+s,e_a=c,e_b=c+\delta\)，则

\[
d-\hat d=2\langle s,c\rangle+2\langle v+s+c,\delta\rangle+\|\delta\|^2. \tag{12}
\]

即使动作响应误差 \(\delta=0\)，共同偏置沿响应的投影 \(2\langle s,c\rangle\) 仍可翻转 margin。反过来，去共同偏置也不能消除 \(\delta\)。因此“余弦很高”“平均 MSE 降了”“teacher forcing 更近”“neutral anchoring”均非排序保证。

该界对给定有界误差在最坏情况下不可随意缩小：一维中令预测残差 \(\hat r_a=1,\hat r_b=1+\epsilon\)，\(e_a=\epsilon,e_b=-\epsilon\)，任意 \(\epsilon>0\) 均使真实顺序反转，尽管误差可任意小。没有正 margin 的排序不存在统一抗误差保证。

保存实际终态后用 (10) 可做**精确事后归因**，不能倒过来把实测 \(\varepsilon_i\) 当选择之前已有的全域证书。若 \(\hat d\le b_a+b_b\)，结论是“此界不能保证”，不是必然反转；有反转则说明“满足这些误差上界且 margin 超界”的合取不成立。

### 定理 4：有界能力值的扰动稳定性

比较同一 \(D,\lambda\) 下真实/预测线性目标，若
\(\|r-\hat r\|\le\epsilon_r\)、\(\|K-\hat K\|\le\epsilon_K\)、\(\|v\|\le a\)，设
\(d_0=\epsilon_r+a\epsilon_K\)、\(b_0=\|\hat r\|+\|\hat K\|a\)，则

\[
\boxed{|V_D(r,K)-V_D(\hat r,\hat K)|\le2b_0d_0+d_0^2.} \tag{13}
\]

证明：逐 \(v\) 比较残差，相差至多 \(d_0\)，预测残差范数至多 \(b_0\)，展开平方后取 inf。两个后继的值差须大于两侧 (13)、各自非线性余项 (9)、求解误差及需要的物理/编码度量误差之和，才有对应真实能力排序保证。不同域/不同 action chart 则此命题不适用。common gain 还需计入其基准 \(\|r_*\|^2\) 的误差；共同精确 \(r_*\) 时该项相消。

### 长时域与重观察

在完整历史空间上若真实可观测转移与 \(\hat f\) 单步差至多 \(\epsilon\)，且真实转移为 \(L\)-Lipschitz，初值误差 \(e_0\)，则相同动作下归纳得

\[
e_T\le L^Te_0+\epsilon\sum_{j=0}^{T-1}L^j. \tag{14}
\]

证明：\(e_{t+1}\le Le_t+\epsilon\)，逐次代入。\(L=1\) 为 \(e_0+T\epsilon\)，\(L>1\) 可指数增长。真实重观察降低初值误差，但不消除下一跳的 \(\epsilon\)；历史窗口有误也不能把 \(e_0=0\)。这是解释“真实前缀条件 H1 仍反转、开放 H2 进一步反转”所需的区分。接触切换/表示非 Markov 时统一 \(L\) 未给定，不能宣称已获长期界。

闭环策略随预测误差改变动作时，若反馈律为 \(L_\pi\)-Lipschitz、真实转移对状态/动作为 \(L_x,L_u\)，比较同一反馈策略可得 \(e_{t+1}\le(L_x+L_uL_\pi)e_t+\epsilon\)。离散 argmin 无正 margin 时可能不连续，不能直接套这个式子。

## 5. latent 目标到物理任务：还需要一座桥

### 命题 5：度量校准的充分条件与必要障碍

若在所有待比较合法终态及外观协议上，有事前固定 \(a_\ell>0,b_\ell\in\mathbb R\) 和 \(\tau\ge0\)，使

\[
|\ell(x,g)-[a_\ell L^E(x,g)+b_\ell]|\le\tau, \tag{15}
\]

则由两次三角不等式，真实编码代价优势 \(L^E(x_C)-L^E(x_B)>2\tau/a_\ell\) 保证物理代价优势。结合 (11)，预测优势大于预测误差界加 \(2\tau/a_\ell\)，即可迁移到物理排序。这里只是一个明确的充分条件，不要求实际用新拟合 readout 建桥；本轮禁止拟合 Q。

更一般地，严格递增校准函数 \(\psi\) 与误差界也可建桥。仅有 bi-Lipschitz 范数上下界通常不足以保持任意两点的次序；必须有足够分离的上下界区间。

必要障碍：若两个任务相关状态满足 \(E\rho(x)=E\rho(x')\) 而所需动作/成功谓词不同，单帧 latent 不能给出所有该类任务的确定性正确判断。证明：相同输入的确定性算法输出相同；构造两个要求互斥动作的状态，其中至少一项失败。历史能否消歧必须另外验证。同样，若相同物理状态仅外观变化就改变 latent 任务排序，则不存在覆盖这些条件的**精确、外观不变、保持所有排序**的当前 latent metric 解释；这不等于 compiler 理论普遍失败。

任务反例：物理距离 \(\ell=q^2\)、encoder \(z=(q,10c)\)、目标 \((q_g,c_g)=(0,0)\)。终态 A 为 \((q,c)=(0,1)\)，B 为 \((1,0)\)。物理 A 好于 B（0<1），精确 latent 距离却 B 好于 A（1<100）。即使 \(\hat f=f\) 且完整可微，也没有真实排序保证。另如成本把速度罚项和位置混在一起，速度改善可掩盖任务位置损害。

success 是不连续谓词，平均代价下降不保证 success。当前 moving PushT 合同还要求 agent+object 四维 XY `<20` 且角差 `<pi/9`，不能换成 native7、只物体 XY 或 coverage。要证明成功须分别对各约束有 robust slack；20.24546 不能四舍五入成成功。每环境沿用自己的封存成功定义，不将此阈值套给 Reacher。

## 6. 从局部能力到全局非贪心优势：不蕴含与选择条件

### 6.1 own residual 与 common response 不是同一个命题

一维、\(\lambda=1\)、无界 ridge：\(V=r^2/(1+K^2)\)。后继 B 有 \(K_B=2,r_B=10\)，C 有 \(K_C=1,r_C=0.1\)。B 的响应更强，但 own 值 20 大于 C 的 0.005；共同 \(r_*=1\) 下 B 的值 0.2 小于 C 的 0.5。故 common 响应优势不推出 own 更接近可达目标；own 更优也可能只因为 residual 小，并非响应改变。两因素交换 \(V(r_i,K_j)\) 可分解证据，但真实未来标签不得进入 selector。

对固定相同残差、无约束 ridge，若 \(K_BK_B^\top\succeq K_CK_C^\top\)，由
\(V=\lambda r^\top(KK^\top+\lambda I)^{-1}r\) 和正定矩阵逆的反序性质，可证 \(V_B\le V_C\)。这是**足够的矩阵序条件**，不是“最小奇异值增大”的同义词；own 残差、有限动作域或奇异方向改变时不可直接用。

### 6.2 三个明确反例

- **秩/奇异值不指向任务。** 目标残差为 \(e_1=(1,0)^\top\)。\(K_C=\operatorname{diag}(2,0)\)、\(K_B=\operatorname{diag}(0.1,100)\)。B 秩与最小奇异值增加，却在目标方向更差，\(V_B=\lambda/(0.01+\lambda)>\lambda/(4+\lambda)=V_C\)。
- **导数大但动作饱和。** \(u\in[-1,1]\)，B 的末端位移 \(y_B(u)=0.01\tanh(100u)\)，C 为 \(y_C(u)=0.1u\)，目标 0.5。B 在零点的导数 1 大于 C 的 0.1，但最大前进不足 0.01，C 可前进 0.1，真实最优终距 C 更好。局部 AD 完全正确并无帮助。
- **没有需要非贪心的任务缺口。** \(x^+=x+u, |u|\le1,x_0=1,g=0\)，一段 prefix 后有一段 suffix。zero prefix 后 suffix=-1 已到目标、成本 0。任何相对 zero 不更贪心的 prefix 都不可能严格胜它。模型完美不保证每个来源根是 controllability-gap 任务。

### 命题 6：原 stage+reach selector 的精确获选条件

固定有限候选集合 \(\mathcal B\)、\(\beta>0\)、分数 \(S(B)=c_{stage}(B)+\beta V_B\)。比较 B 与 C，

\[
\boxed{S(B)<S(C)\iff \beta(V_C-V_B)>c_{stage}(B)-c_{stage}(C).} \tag{16}
\]

这是移项恒等式；B 严格胜所有其他候选才唯一获选，平局须原 stable tie 规则。存在非贪心能力候选不等于选中它。若某候选在 stage 与 reach 上都被另一候选弱支配且至少一项严格，则任意正 \(\beta\) 都不选它。这只能排除**当前固定库内**的选择，不能证明连续空间无解。不能看真实结果后改 \(\beta\)、候选或 stage 权重。

### 命题 7：不能普遍优于全域最优平坦规划器

若平坦规划器可在相同总时域、同动作域、同反馈信息的全部可执行策略上全局最优，而 D2P 产生的程序属于该策略集合，则
\(\inf_{\pi\in\Pi_{flat}}J(\pi)\le\inf_{\pi\in\Pi_{D2P}}J(\pi)\)。证明：集合包含关系。

所以 P1 对 CEM/GD/GRASP 等实际有限预算算法的优势只能来自搜索/计算组织、动作支持或求解效率，不是一个超越全局最优控制的定理。实际 CEM 不是全局 oracle；此命题既不能据此否定 D2P，也不能将弱/不同反馈基线的劣势当程序级增量。API 数相同不是总计算相同。

## 7. 正面构造：机制可以成立，但原分数未必选中

构造两阶段真实系统，状态含阶段指示、任务位置 \(q\) 和模式 \(m\)。初始 \((q,m)=(1,0)\)，目标 \(q_g=0\)，物理/编码目标均为 \(q^2\)，冻结表示保留完整 \((q,m,phase)\)，精确模型与 renderer。第一阶段可行动作 zero 保持 \((1,0)\)，reconfigure B 到 \((2,1)\)。第二阶段 \(u\in[0,1]\)，动力学

\[
q^+=q-2mu,\qquad m^+=m. \tag{17}
\]

prefix B 成本 4、zero 成本 1，确实暂时退步；zero 的任务响应为 0，B 的响应为 2。zero 无法改善，B 取合法 u=1 达到成本 0。共同残差下响应改善，own 非线性最优代价也改善，动作见证明确且预测无误。这证明原核心机制在一个有限动作模式系统上可以成立；不是 PushT/Reacher 已被证实，也不证明优于能枚举两个模式的强规划器。B 的模式切换需有限 rollout，不伪称连续梯度可自动发现它。

若用原无界 ridge surrogate，\(\lambda=0.1\)，B 的 \(V_B=4\lambda/(4+\lambda)=4/41\)，zero 的 \(V_0=1\)。在 \(\beta=1,c_{stage}=q^2\) 时，\(S(B)=4+4/41>2=S(0)\)，选择器反而不选 B；严格选中需要 \(\beta>123/37\)。这只说明选择条件的必要性，**不建议把真实实验 beta 调到该值**。ridge 最优 \(u=40/41\) 本身在界内，也有严格终态收益，因此不是靠越界获得正例。

接触物理中动作顺序差异可通过实际短 rollout 定义；除非先证明向量场光滑、步长趋零、约束与模式稳定，不能把 (17) 或 PushT 的模式变化叫 Lie bracket 证明。

## 8. 一个真正覆盖 P1 要件的条件定理

### 定理 8（单根、固定程序、可检验的充分条件）

固定合同 \(\mathcal P\)、已在真实未来之前封存的模型选择器与候选、根 \(x\)，以及原必需控制策略集合 \(\mathcal C\)。假设：

A. 输入/renderer/历史/normalizer/动作 chart/模型 buffers/目标/时间/停止规则均符合 \(\mathcal P\)，见证全程合法且真实执行完整。选择 B 满足 (16) 的所有候选比较与原模型 eligible 规则。

B. 按该合同的首段机制坐标，模型非贪心 margin 扣除 (11) 两侧误差后仍超过其所需真实下界；若科学解释需要物理非贪心，则还有 (15) 或独立真实物理验证。既有 gate 容差按原值检查。

C. 预声明的真实能力指标固定为 own-state 或 common 诊断中的明确一种，并另报另一种；其预测优势超过 (13)、非线性余项、求解误差以及任务度量误差的两侧合计。若声称最优可达代价而非局部响应，必须有相应真实全域上/下界，不能拿一次局部求解冒充。

D. 对每个实际控制策略 \(C\in\mathcal C\)，固定动作/反馈策略的终态预测目标优势超过 (11) 的误差和 (15) 的任务校准预算；动态策略分歧另有 (14) 所需稳定性条件，或已封存的逐段实测证据。若结论是 success，还逐项满足成功谓词的严格 slack。

则 B 在该根上满足合同的非贪心、所声明真实能力改善及真实任务代价优于每个该控制策略；若 D 的成功条件也成立则真实成功。

证明：A 固定比较对象且保证见证可执行；B 应用定理 3/命题 5；C 应用定理 4 及命题 2，或其假设给出的全域证书；D 应用定理 3、(15) 与分段误差界。三个结论在同一根/见证上合取。

这个定理是有条件的，不偷放“模型完美”或“物理肯定更好”当前提；各 margin/误差均可指向待测对象。现在并没有这些条件的完整实证证书，不能称已证明当前实验通过。A–D 是充分而非全部必要条件；例如统一范数误差很大但误差与响应正交，排序仍可能正确。

### 独立根与强 gate

单根定理不推出根分布泛化。若 eligible 定义完全依赖事前模型信息，根独立来自固定条件分布，则可在该条件分布上评价；不能按真实结果筛根。对于每根配对任务损失差 \(D_i\in[-M_0,M_0]\)，Hoeffding 给
\(\Pr(|\bar D-\mathbb ED|\ge t)\le2\exp[-nt^2/(2M_0^2)]\)。这是说明所需独立性/有界性的分析式，不替换当前 CI 方法、样本数或多重比较合同。

“每个 eligible 都保持机制”需要逐根 margin 及误差控制，平均损失不够。若每根独立有机制失败概率 \(p\)，n 根全过概率为 \((1-p)^n\)；再小的非零 p 也不能导出确定性的全过。当前 40 槽、8 context、多个 word 不等于 40/8 个独立根；模型选择后反复在两个已知根分析，不生成新独立确认。

## 9. 当前证据：直接关联与不支持的推论

### 9.1 P1 固定库与 selector

读取 `le-wm-main/results/agent_p1_model_library_selector_evaluation_20261028_v1_MAIN01.json`，独立核 SHA256 为
`d8b3bedd38e9f9a56d0156f2452d5a20f363acbdad42ea5e3ee4d91f9626ff67`。

报告 `status=exact_seen_execution_reuse_exploratory_not_admitted`；40 固定槽、2 known roots、0 fresh roots、39 actual latent，index12 unknown 保留；`fresh_confirmation=false,P1_admission=false,C2_claim=false`。8 context 中 anchor 0/2/5/7 换词，其 allXY 改善量依次为
`+2.5203652350760706,-8.628791007168957,-22.9461823516071,-3.3688940071400637`，另四项为零。不是新策略真实派发或完整因果机制确认。

closure/v15 定位的原库内 76 可比 pair 中 12 反转，4 pair unknown；同词 Z/S 19 可比 pair 无反转、1 unknown。本文没有重新运行其 producer/consumer，只读合同/已验收报告，不把 12/76 消掉或补成 12/80。排序反转关联 (10)；换词后 allXY 损害须再拆“预测→实际编码”和“实际编码→物理任务”，不能仅凭符号归模型误差。

原 40 槽存在真实接触/物体响应，与“所有动作没有物理作用”不同；但不能由此推出任务相关可控性扩张。已知 allmode/mixedmode 选择仍 stage-greedy，即使局部导数验证正确，也没有选中 P1 所需非贪心行为；它先卡在任务/候选/选择链，不能用终态改善替代缺失机制。有限库非贪心候选被支配关联 (16)，不说明连续空间无解。

### 9.2 P5 最新保存分析，只作为度量边界线索

读取 `le-wm-main/results/agent_p5_live_raster_analysis_20261028_v3_saved_MAIN01.json`，独立核 SHA256 为
`6b167d6a24c2756f938a5840a02ee29211612558f8e780d1a6c905581731a928`。

16 交叉评分；同 state terminal visible/white featureMSE：xy `0.4516983608344624`、yx `0.5698911346941358`、x_minus_x `0.4516983608344624`、zero_zero `0.6196937512046183`；goal 为 0，没有成功施加 goal 颜色的表征 first stage，不能据此证明模型对真实 goal 颜色变化不敏感。

yx 对 zero_zero 的成本差在 terminal visible 为 `+0.0013478756126013014`、white 为 `-0.03459676792770838`，随 terminal 颜色反转。36 无序 word-pair×condition-pair 中 4 strict reversals、6 tie-involved，不是 36 独立样本。四词平均 terminal 主效应 `-0.02530242211827649` 不能遮盖逐词变化。

这些反映当前编码目标对外观干预的敏感性，与 (15) 所需的外观鲁棒任务关系有关；不证明真实物理任务变化、不直接量化 P1 的预测 \(\hat f\) 误差、也不构成跨模型/任务的普适反例。新 raster frame15、旧 coverage frame20，`native_vs_raster_rank_agreement=null`；不能合并成同时间排序证据。原 renderer/replay/RNG 时间接线故障仍按协议问题分栏，不回填旧门。

### 9.3 P3 与当前 P1 的逻辑关系

v15 paired quota 两侧各 634 predict API/sample、36 compiled/10 paid refusal/2 not_reached；完整资源向量不相同。它不提供等 FLOPs、真实收益或 P1 前提；重复物理复用也不是新根。P2/P3/P5 工程与 MAIN 实验可以继续，但不借其完成度打开 P1。

## 10. 最小事前固定判据：区分错误、失配、条件未满足

本节是交给 MAIN 的执行判据设计，**本轮未执行新增实验**，不是修改 gate 或新增替代准入。先固定完整 cohort、任务、候选、Q/A/R、rank/retained basis、动作域、时域、参考控制、margin/tie、缺失处理及资源向量；旧已知根只能归因，下一独立面板按原授权/历史认证流程，不自动追加根。

### 10.1 最小数据对象

每个根/条件保留一条模型事前选择 B、原要求的 zero/greedy 和强控制；same-bank 机制分支保留同库 greedy。每条见证只用一次封存 raw 动作和各自完整历史，保留 root、prefix-end 与 terminal 的同时间 RGB/状态/任务标签。后继能力沿原已声明两尺度/方向/探针合同，不依真实结果增加方向或重选根；如果原合同没有足够能力观测，则记录 unknown 并向 MAIN 提出精确缺项，不从终态收益反推能力。

对所有必需成对比较，构造同一行：

`identity → predicted-stage / observed-stage / physical-stage → predicted-V / actual-local-V → predicted-terminal / observed-terminal / physical-task / success → resource-vector / freshness`。

这是一张归因表，不是新 selector；evaluator-only 物理坐标、实际误差、能力探针全部不得流入选择。

### 10.2 判别矩阵

| 判据 | 最小固定比较与数学输出 | 能判什么 / 不能判什么 |
|---|---|---|
| T0 身份/接线 | 原合同完整 root/history/raw/goal/model+buffer/normalizer/renderer/frame 绑定；相同模型 API 逐元素重放；已有 AD–model-FD | 违约是实现接线失败；通过只排这些具体接缝，不证明模型正确 |
| T1 输入/评价协议 | 同真实状态/同动作/同时间下，预先固定外观或历史协议的合法配对；所有条件保留；确认干预确实改变目标输入 | 合法单因素差异与输出变化支持协议贡献；zero-first-stage 不可推无敏感；P5 当前只支持 terminal 外观轴 |
| T2 数学选择与支配 | 原全部候选重算 (16)，逐候选 own/common、无界/有界/非线性、rank-retention 分栏 | 精确评分不一致是实现错误；精确但无 eligible/被支配是该库条件未满足，非全局理论无解 |
| T3 预测排序 | 固定所有原可比 pair，精确计算 (10)–(12)，保留 margin/误差投影/二次项及 unknown | 恒等式不闭合先查坐标/时刻/单位接线；闭合且反转可定位预测误差如何越过 margin，不是新的模型训练结论 |
| T4 编码到任务 | 同实际终态的 observed latent cost 与封存物理代价/成功逐对比较；有 (15) 证书才报保证 | latent 顺序对而物理错是本批 metric mismatch；不能仅因速度项/native7 降低称位置获益 |
| T5 后继能力 | 按原事前 own-state 物理任务及 common response 诊断分别比较；相同预算/合法动作域，核残差、名义点、被动/zero 参照 | 只终态好、common好或rank好都不够；两尺度 secant 稳定不证明全局能力。历史未知量保持 unknown |
| T6 原 P1 合取 | 同 eligible 根检查原机制 gate、真实能力链、原各强控制真实 paired outcome/CI、完整 provenance 与独立 split | 只有符合原门和科学闭包才能升级；不是把前五项测试数汇总为通过 |

T0/T1/T3 有一项身份/代数不合法时，该行对相应科学假设为 unidentifiable，而非理论失败。若 T0 合法、同任务 T3 反转且 T4 一致，支持该冻结模型/域上的预测排序条件不足；若 T3 正确而 T4 错，优先是 metric bridge 不满足；二者可同时存在，不能强迫互斥归因。若 T2 无非贪心 selected，先标“任务/候选/选择前提未满足”，没有理由把下游缺失写成能力反例。

要把一种具体代码/协议错误称为历史反转的原因，还需 **只改变该接缝并逐项复现原基线** 的固定干预，观察其能否消除原反转且保留其他条件；看到另一次 encoder 颜色敏感不能完成该识别。一个合法对照可检出接线错，却不足估计独立分布上的收益。

### 10.3 最小停止/解释规则

1. 数据不齐/身份不等/时间不同：unknown 或协议失败，不补标签、不删分母、不再做物理优劣推论。
2. 原模型条件 cohort 不足：保持空队列与失败门，不用易根填满；平台只叫有限预算平台，不叫已证驻点/不可达。
3. 已选 B 不满足真实非贪心：该根原机制 gate 失败；不能换参考点、换 Q 或换 observability 分支救回。
4. 有非贪心但无原声明真实能力改善：保留为非贪心行为，不叫 reconfiguration 机制成立；common/own 不一致如实报告。
5. 有机制但无独立真实收益或强控制 CI 不严格有利：原 P1 kill experiment 未成立；不能用局部正例、模型 Q 或更多测试替代。
6. 当前所有已知根只能诊断；一旦未来另获合法独立 outcome，按原已冻结版本一次评价，失败也全部保留。本文没有授权任何新 CUDA/物理派发。

## 11. 仍未知、且确实会改变判断的量

- 最终固定 P1 cohort 是否真正覆盖直接局部方法停滞而两阶段可达的任务分布；两个 known roots 不是这一前提或 fresh outcome 的证书。
- 对 selected prefix 和实际 suffix 的加权成对误差，尤其与 goal residual/response 的投影；平均 MSE 和余弦不足。旧 missing index12 的终态编码仍 unknown。
- 最终候选库中模型/真实 own-state 和 common 物理能力的同根合取、被动漂移/时间/接触中介的分解；现有有限 secant 不给全局最优值。
- 统一合法 trust domain 内的 \(M,L,\epsilon_K\) 和长时误差界；接触、像素量化、历史非充分性使它们不能默认存在。
- 冻结编码 metric 与物理任务之间可部署、事前的排序校准/鲁棒性；P5 当前不提供 frame15 同时间物理任务排序，更不是 P1 的替代证据。
- 完整强基线同反馈/支持/资源的独立配对结果、freshness/reset/loaded-byte 闭包。634/634 API、物理复用、多个 context 不能补齐。

### 本轮数学复核

仅在 RAM 用 Python 标准库 `fractions.Fraction` 做精确有理数断言，全部通过：§7 的 \(V_B=4/41\)、合法 ridge 动作 \(u=40/41\)、其实际任务成本严格低于 zero、beta=1 的 stage-score 拒绝及精确阈值 \(123/37\)；§6.2 的目标方向 ridge 次序；§4 的任意小误差排序反转取 \(\epsilon=1/1000\)。未导入 torch、模型或环境，未新建结果/脚本文件。正例见证用 u=1 可达真实任务成本0；ridge 动作虽有正则偏差，也有严格真实收益。

文本与公式作人工复核；目标文件空白检查通过。RAM 断言只核算术，不是 P1 数据/实现完整性、freshness、真实机制或统计 gate 的回执。

最终判断：**P1 的条件机制可推导成立；仅从局部 SVD/ridge 与现有准确性假设推导当前强 gate 必过，不成立。当前实证尚未完成所需条件验证，且确有协议因素，既不能宣布理论已成功，也不能宣布原方向已被否定。** 下一步有区分力的是固定同根、同见证、同时间的 T0–T6 分层判据，而不是改 gate、更多事后筛根或把另一个 P 阶段当作 P1。
