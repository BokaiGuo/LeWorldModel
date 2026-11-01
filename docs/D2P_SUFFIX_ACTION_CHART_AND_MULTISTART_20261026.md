# D2P：后缀动作坐标、局部盆地与多起点修复

## 结论与合同边界

后续已完成全8根/全136词的评分/执行数值合同干预，旧轨迹及完整物理终态
精确复现。统一评分未改善历史多起点runtime，且新6个witness仅2个保持首段
真实RGB非贪心，不能作为P1修复。新来源确认已启动，详见
[统一数值后缀合同](D2P_STATE_NATIVE_SUFFIX_CONTRACT_20261026.md)。

新证据再次说明，不能把“冻结模型仍有负结果”直接等同于理论失败。
在完全相同的真实 RGB successor、相同20步前缀与10步后缀时域下，
`202643006` 的局部求解距离为1.476133，同模型CEM后缀为0.028642。
当前冻结模型能够找到并选择有效动作；该根的严重后缀失败有明确的
数值搜索/盆地限制，而非由模型不存在这条有效动作路径推出。

但本次固定了前缀，因而没有修复旧的首段RGB排序。此根也是greedy fallback，
不能把后缀恢复当作非贪心重配置机制证据。对其他根，更低模型成本仍可对应
更差物理终态。P1及研究总闭环保持未完成；既不宣称理论失败，也不宣称代码
问题已全部穷尽。

## 已完成：同前缀、同初始动作的全部8根因果对照

证据：
`le-wm-main/results/reacher_d2p_suffix_action_chart_all8_v2_20261026_cuda.json`，
以及同名protocol、selection、roots与implementation。根为上一轮新根确认的
`202643000..007`，本轮属于已知根诊断，不是再次独立确认。

固定每根原dispatch-incumbent的前20步，从其原后10步动作开始，比较：

| 后缀 | 全8根平均真实关节距离 | 相对原incumbent胜/负 |
|---|---:|---:|
| 原cold/warm incumbent | 0.2152529155 | — |
| 原tanh坐标再8次迭代 | 0.2317557149 | 4/4 |
| 原tanh坐标再32次迭代 | 0.2306023246 | 4/4 |
| 直接command-box再8次迭代 | 0.2315076502 | 5/3 |
| 直接command-box再32次迭代 | 0.2305694698 | 5/3 |
| 同前缀CEM | 0.0429868351 | 3/5 |
| 只按当前模型成本从上述六个动作选小 | 0.0357004607 | 6/2 |

最后一行为封存前的model-only选择，不是按真实结果取oracle。七根选择box32，
一根选择CEM。对原incumbent的均差为-0.1795524548，根配对描述性95%CI
[-0.5483064442,+0.0184313892]，跨零；没有独立效能、显著性或等预算结论。
全部48条完整30步真实分支保留，没有只报告成功根。

根`202643006`的具体分离：

| 后缀 | 模型终点成本 | 真实关节距离 |
|---|---:|---:|
| 原incumbent | 1.8595237732 | 1.4761333493 |
| tanh再32次 | 1.6353343725 | 1.5896717947 |
| box再32次 | 1.6348440647 | 1.5875940244 |
| 同前缀CEM | 0.0067835040 | 0.0286419516 |

该根约80%初始command接近饱和，但仅更换坐标并未解决其盆地问题。
另一根`202643007`模型成本由0.020738降至0.016932，物理距离却由0.020526
增至0.090249；不是“多迭代必然好”，也不是“换box必然修复”。

## 代码与数值审计

旧参数映射是`command=tanh(params)`，模型动作是
`midpoint + radius * command`。其导数含`1-command²`，接近饱和时很小。
新增`reacher_d2p_box_action_suffix`直接优化`command∈[-1,1]`，保持同一模型递推、
历史索引、flatten时序、仿射动作标准化与真实解码，不做`atanh`近似。

每个根均精确通过：原根/目标、四次prefix历史、prefix RGB成本、原后缀动作和
模型成本、原后缀真实关节距离；两种坐标的初始动作与前向预测逐值相同。
Jacobian链式法则的相对误差最大约3.7e-7；8个固定坐标、epsilon=0.001的
有界单侧/中心FD中位相对误差最高约0.001404，低于预封存0.1门槛。
权重未改、主机CUDA、完整来源SHA、实现与选择封存检查全部通过。
完整审计共13126个实际predict样本；所有候选/检查开销包含在内，不是单策略预算。

box每步求有界阻尼GN的绝对可行动作端点。令当前command为c、目标残差r、
响应J、G=JᵀJ+λI，则求解

`min_{v∈[-1,1]^d} vᵀGv − 2(Jᵀr+Gc)ᵀv`，再沿`v−c`做原非线性下降接受。

QP必须通过可行性、active-set终止与目标duality-gap证书，不放宽非线性门槛。
换坐标也改变了阻尼正则的度量，因此这是“坐标/正则化整体”干预，不能把收益
全部精确归因于一个tanh导数因子。相同迭代数不是相同总计算资源。

## 落地的独立runtime与算法映射

新模块`reacher_d2p_suffix_multistart`保留原观测状态的cold/warm incumbent，
在同一个RGB历史中比较两种坐标与CEM候选。所有根都使用相同规则，不按qpos、
真实终态RGB、已知成功与否选择重启，也不拟合阈值。原compiler及其历史SHA未改。

```mermaid
sequenceDiagram
    participant R as run_reacher_d2p_suffix_multistart_confirmation
    participant C as reacher_d2p_suffix_continuation
    participant M as reacher_d2p_suffix_multistart
    participant E as reacher_d2p_context
    R->>E: restore root; advance fixed prefix; observe RGB
    R->>C: compile_continuation(planned_suffix)
    C-->>R: cold/warm model incumbent
    R->>M: compile_candidates(incumbent, observed RGB/history)
    M-->>R: all bounded commands, costs, model-only selection
    R->>R: seal all roots and all suffix candidates
    R->>E: execute every sealed terminal branch
    E-->>R: physical outcomes for evaluation only
```

```latex
\begin{algorithm}[t]
\caption{Model-only numerical suffix repair}
\begin{algorithmic}[1]
\Require Frozen model $F$, observed RGB/action history $h$, incumbent parameters $p$, action bounds, fixed seed
\Ensure Bounded candidate commands and model-cost selection
\State $c\gets\tanh(p)$; retain $c$ unchanged as incumbent
\State $\mathcal T\gets\textsc{CompileSnapshots}(F,h,p,\{8,32\},\lambda=0.01)$
\State $\mathcal B\gets\textsc{CompileBoxSnapshots}(F,h,c,\{8,32\},\lambda=0.01)$
\State $p_C\gets\textsc{CEM}(F,h,128,3,16,\mathrm{seed})$
\State $\mathcal C\gets\{c,\tanh(\mathcal T_8),\tanh(\mathcal T_{32}),\mathcal B_8,\mathcal B_{32},\tanh(p_C)\}$
\State $j\gets\arg\min_i\textsc{FrozenModelCost}(h,\mathcal C_i)$; ties prefer incumbent, then lexical name
\State \Return $\mathcal C,j$ \Comment{No physical-outcome input}
\end{algorithmic}
\end{algorithm}
```

| 伪代码步骤 | 实现 |
|---|---|
| 两种32次局部精化，保留8次snapshot | `compile_snapshots` / `compile_box_snapshots` |
| 独立CEM盆地，保留best-ever | `reacher_d2p_cem_incumbent.search_with_incumbent` |
| singleton模型成本与tie规则 | `compile_candidates` / `choose_candidate` |
| 全根、全候选封存后执行 | `run_reacher_d2p_suffix_multistart_confirmation` |

令输出维m、动作维d、后缀H包，每次线性化存储O(md+d²)；显式VJP按m行核账，
有界QP每次active-set线性系统最坏O(d³)，总轮数由实际证书记载，不声称固定常数。
CEM使用128×3×H个predict样本；H=2时768。8次是32次内部snapshot，不重复计数。
两个singleton候选的重评分另计4个predict样本。神经计算均为主机CUDA。

## 独立新根确认

入口`run_reacher_d2p_suffix_multistart_confirmation`已完成8个新根
`202644000..007`，排除之前160个source episode。对四类prefix
dispatch/同库greedy/neutral/完整CEM，均执行六种数值后缀，共192条真实30步分支。
保留原prefix selector；所有prefix选择和全部32组suffix选择分别封存后，
才测任何终态。CEM基线也接受同样的模型成本候选比较，避免只给D2P修复。

结果路径：`le-wm-main/results/reacher_d2p_suffix_multistart_fresh8_20261026_cuda.json`。
全部192条真实分支、107432个predict样本完成；来源隔离、根/目标身份、初值解码、
后缀计数、模型分段、原权重、两级封存与实现SHA门全部通过。全部8根保留。

| 固定prefix族 | 原local incumbent均距 | 多起点model-only均距 | 胜/负 |
|---|---:|---:|---:|
| dispatch | 0.3613584872 | 0.1240538912 | 6/2 |
| 同库greedy | 0.7838119676 | 0.1213398272 | 7/1 |
| neutral | 0.8874153719 | 0.2118802611 | 7/1 |
| 完整CEM | 0.1224174584 | 0.1233060207 | 4/4 |

dispatch修复减原local的均差-0.2373045960，描述性根配对95%CI为
[-0.6360745206,-0.0091651810]。这是一项预封存的新根修复pilot，不是经多重
比较校正的正式效能证明。不能以小样本区间不跨零代替研究准入。

新根`202644003`提供独立病例：原local距离1.623534，CEM候选由模型成本选择，
实际为0.038918；不是针对旧坏根调参。与此同时，新根`202644006`仍未解决：
dispatch距离0.884740→0.794063，所有控制都仍有高误差。该根不被删除，不能据此
宣称所有代码可能性均已排除或把剩余误差直接定性为理论失败。

修复后的dispatch对同样修复的greedy为2胜/5负/1平，均差+0.0027140639、CI
[-0.0027991352,+0.0076871243]；对完整CEM为4胜/4负，均差+0.0007478705、CI
[-0.0122309089,+0.0133264282]。先前一部分“重配置胜过greedy”的表象来自
greedy后缀本身未充分求解；不能把通用数值修复算成D2P独有机制收益。
七个原模型witness中六个首段真实RGB非贪心，`202644006`仍反转，P1保持未准入。

新根图包位于`le-wm-main/results/reacher_d2p_suffix_multistart_confirmation_figures_20261026/`，
包含全部根、四族配对、两张PNG/PDF、独立summary、图注与SHA manifest，已目视检查。
最新相关回归535项通过，`git diff --check`无错误；fixture不替代CUDA实验证据。

## 失败回执、图表与复现

首版`reacher_d2p_suffix_action_chart_all8_20261026_cuda`已封存全部模型选择，
但在首次真实终态RGB预处理时因renderer负stride数组异常中断。v2仅对RGB做
`ascontiguousarray`，不翻转或修改像素；回归测试覆盖图像值/朝向/stride。
v1实现、协议与选择保留。该新审计脚本错误不是历史P1失败的解释。

图表与独立标量复核：
`le-wm-main/results/reacher_d2p_suffix_action_chart_figures_20261026/`。
Figure_1保留所有根/所有六分支的实际距离；Figure_2并列模型成本、真实RGB成本、
真实距离的配对变化，明确显示三者不总一致。PNG/PDF、图注和SHA manifest齐全，
已目视检查。图表不由fixture或CPU替代神经实验生成。

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
.venv/bin/python -u -m experiments.spectral_reachability.audit_reacher_d2p_suffix_action_chart \
  --reference results/reacher_d2p_suffix_continuation_fresh8_20261026_cuda.json \
  --policy /home/gbk/.stable_worldmodel/checkpoints/reacher/lewm \
  --adapter-artifact results/reacher_d2p_p1_distribution_repair_v1_20261024_cuda/reacher_training_action_stats.json \
  --output /absolute/new-action-chart.json
.venv/bin/python -u -m experiments.spectral_reachability.run_reacher_d2p_suffix_multistart_confirmation \
  --policy /home/gbk/.stable_worldmodel/checkpoints/reacher/lewm \
  --adapter-artifact results/reacher_d2p_p1_distribution_repair_v1_20261024_cuda/reacher_training_action_stats.json \
  --disjoint-reference results/reacher_d2p_suffix_continuation_fresh8_20261026_cuda.json \
  --seed 202644000 --roots 8 --output /absolute/new-multistart-confirmation.json
```

以上两个命令都必须在获授权的主机CUDA环境运行；输出路径及全部sidecars需不存在。
