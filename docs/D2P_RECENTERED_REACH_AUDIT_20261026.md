# D2P：零切点可达代理与非线性可行动作的审计

## 运行前固定的研究问题

这不是继续拟合上一包偏移。检验对象是prefix选择器中的`V_reach`实现近似：
同一冻结模型，零后缀切点的有界线性QP，是否被当成了整个非线性动作盒的可达
代价？QP最优性证书只能证明局部仿射问题解正确，不能证明真实非线性模型无法
到达某个目标。该区分也不能直接证明D2P理论通过或失败。

初步只读既有模型封存记录发现：两批全部816个prefix/任务单元中，812个零切点
QP值高于已经存在的非线性可行动作目标值。前一批/后一批全驱动的比值中位数约
320.70/298.97。这里非线性目标是完整192维残差平方和加0.01倍后缀绝对动作
平方和，非MSE与sum混用；它是已知可行解，不是全局最优值。这个差异证明代理
不能当成非线性不可达证书，尚不证明它导致了全部历史P1反转。

## 显式新合同，不回写旧选择器

保持全部16个已知根、三种精确执行器限制、全部17个谱prefix、原目标及官方
原始冻结权重。前缀不变，以既有`cem_then_gn4`后缀为incumbent，固定两次
重线性化。绝不重新筛根、拟合系数或用物理outcome决定参数。

绝对动作u仍在[-1,1]，原始正则为`0.01 ||u||²`。在a处令预测为F(a)、响应J，
正确的仿射残差为`g - F(a) + J a`，故局部问题为：

`min_u ||g - F(a) + Ja - Ju||² + 0.01 ||u||²`。

不能遗漏`Ja`，不能把delta直接限制在[-1,1]，也不能误加入步长正则对应的
`lambda*a`。这些是新实现必须检验的代数合同，不声称旧步长阻尼求解器写错。
每步QP只产生候选；用固定六尺度及精确原解，对完整冻结非线性模型逐个singleton
评分，只有原始正则化目标降低才接受。局部QP证书与非线性可行上界分字段保存，
不把前者升级为全局非线性下界。

前缀score显式设为`prefix_MSE + nonlinear_feasible_objective / 192`，beta固定1，
平局按原索引。保留旧zero-tangent selector与只采用原非线性incumbent的selector，
另对greedy/zero施加同样重线性化。这个新诊断合同不替代原P1协议。

## 主机CUDA执行合同

先完成全部16根×三任务×17程序的模型计算，封存所有选择；之后才执行五种
规则对应的240条真实15步分支。旧两种选择的96条分支必须与既有完整动作、
逐包192维RGB、qpos及执行哈希精确重放。所有神经计算仅主机CUDA，参数、
buffers、eval状态、官方权重、源数据receipt与源码哈希全部绑定。

实测35088个predict样本、2448个完整Jacobian、470016行VJP；这些是前向样本
和反向行数两个独立计数，不宣称免费谱计算或等资源优势。不整读93GB HDF5
计算hash，沿用绑定来源receipt。所有失败版本和历史输出保留。

新增实现：`reacher_d2p_recentered_reach.py`和
`audit_reacher_d2p_recentered_reach.py`。合成测试只验证代数与合同，不能替代
CUDA效能、物理能力改变或P1准入。**本轮16已知根CUDA已完整完成，15项完整性门
全通过；P1仍关闭**。独立分析器复算1632个新QP，核验全部816个程序与96条旧选择
完整物理重放，未把局部QP的数值正确性当成非线性可达性证书。

## 已完成的因果结果

关键全驱动根202650004：旧零切点selector选index9，真实终距1.654116001。
仅将评分换成**已经存在的非线性可行动作代价**，不增加搜索、不改变候选动作，
改选index10，真实终距降到0.024495188；对应真实RGB代价从1.809507835降到
0.003712761。这是一处明确的代理评分/分支选择失败，不能归为理论负结果。
进一步固定两次重线性化后，index10终距0.018548710、RGB0.001663959；同样优化
的greedy仍选index9，终距1.648683244。因此“换评分”与“增加优化”效果必须分开。

全16根终距均值如下，所有根保留：

|任务|旧零切点选择|仅可行incumbent评分|重线性化选择|同优化greedy|同优化zero|
|---|---:|---:|---:|---:|---:|
|全驱动|0.128258|0.024917|0.024553|0.126436|0.027925|
|仅肩|0.531334|0.527163|0.527732|0.588006|0.520035|
|仅腕|0.693284|0.677796|0.678297|0.678297|0.724138|

全驱动重线性化对同优化greedy为1胜/0负/15平，描述性根级bootstrap区间
[-0.305650225,0]；对旧selector为7胜/9负，区间[-0.309545898,0.001225973]。
均值变化主要来自上述已知坏根，不冒称一般化或显著优势。
三任务仍各0个相对zero前缀代价更差的被选根。关键index10虽然差于greedy，
但前缀RGB2.020156467仍优于zero的2.124598573，故不是原提案要求的非贪心
P1见证。此修复不自动打开P1/P2/P3/P5，也不证明全局非线性最优。

报告：`le-wm-main/results/reacher_d2p_recentered_reach16_20261026_cuda.json`；
独立全根图/统计：`le-wm-main/results/reacher_d2p_recentered_reach_figures16_v2_20261026/`。
原16根相关广覆盖测试1177项通过；加入确认身份与八根统计合同后1179项通过。
v2图目录七项产物SHA与生成器SHA已复核。

## 来源隔离的新根确认（运行前封存）

不改变runtime源码、beta、lambda、两次迭代或六尺度。固定新seed202652000–007，
仅用HDF5 episode元数据确认与216个历史episode不重叠；新episode依次为
2915、1513、8470、3221、2892、5988、5958、7171。先运行完整旧基线与完整后缀
搜索，再以202651000–007作为八个对齐控制，形成新8+旧8的完整审计。
新增`--confirmation-reference`绑定首次16根报告及相同runtime哈希，显式区分
confirmation/alignment；独立分析器要求旧八根全部五种执行逐位重放，新八根
单独给出统计，不能把合并16根区间当成独立确认。所有模型决定均先于真实未来。
新八根原基线已完成456条真实分支/59568个CUDA预测样本，14项完整性门全通过；
完整后缀对照也已完成1224条真实分支/349320个CUDA预测样本，16项完整性门全通过，
408条原分支完整重放。修复版混合16根确认现已完整完成，runtime源码SHA保持
`9cde8223482e2ca356b03cc2bfb63674b8fffef5dfb22716bbb5123964f7e57d`。
240条真实分支、35088个CUDA预测样本，15门全部通过；独立复算1632个新QP与
96条原控制重放，旧八根五种执行的完整物理行与首次报告逐位相同。七项产物SHA
及生成器SHA复核完成。新八根单独统计如下，不使用合并16根均值冒充独立证据：

|新八根任务|旧零切点评分|仅已有可行代价评分|重线性化评分|同优化greedy|同优化zero|
|---|---:|---:|---:|---:|---:|
|全驱动|0.027820|0.027820|0.025874|0.025874|0.034209|
|仅肩|0.697934|0.560880|0.561447|0.966011|0.648722|
|仅腕|0.588883|0.588883|0.588799|0.588799|0.575244|

新仅肩任务只换评分即2根改善、6根相同、无变差，差值-0.137054、描述性CI
[-0.404990,0]，说明旧根之外也有具体分支误选；不能称为普遍显著优势。
重线性化选择对同优化greedy仅肩3胜/0负/5平，差值-0.404564、描述性CI
[-0.799799,-0.124412]，但选中前缀仍比zero更接近目标，不是原P1非贪心见证。
全驱动重线性化对旧评分6胜/2负、CI[-0.004734,0.000777]；仅腕2胜/5负/1平、
CI[-0.000451,0.000232]，不可宣称两者稳定改善。全驱动/仅腕与同优化greedy
全部相同；三任务均0个相对zero非贪心被选根。**代理错误获得独立复现且局部
选择修复有效，但P1仍关闭，原六根同动作排序不能据此宣布已全部修复。**

确认报告：`results/reacher_d2p_recentered_confirmation_mixed16_20261026_cuda.json`；
独立统计：`results/reacher_d2p_recentered_confirmation_figures16_20261026/summary.json`。

### 独立完成的新八根后缀诊断

816个完整响应QP由SciPy独立复算；固定前缀的全根平均终距从旧GN4到CEM后GN4：
全驱动0.725070→0.119198、仅肩0.865705→0.709404、仅腕0.832868→0.579602。
根级胜负分别8/0、8/0、7/1；描述性配对区间分别
[-0.737193,-0.478512]、[-0.239312,-0.076529]、[-0.387457,-0.117257]。
这是固定候选库后缀求解不足的独立确认，不能混为完整planner或同资源优势。

旧common-residual诊断在全驱动新根202652000封存选择index9、greedy为12，
首次得到该新队列一个物理方向一致的非贪心候选：真实前缀RGB2.318323（greedy
1.041644），终态RGB0.011965（greedy0.015115），终距0.036685（greedy0.042652）。
这只是一根诊断见证，不满足原P1独立校准/任务族/完整机制要求；proposal-native
selector仍无合格见证，不能将旧common合同的阳性结果回填新合同。
独立产物位于`results/reacher_d2p_recentered_confirmation_suffix_figures8_20261026/`。

### 对“stage项必然排除非贪心”的反证性检查

新增纯数值审计`audit_reacher_d2p_stage_objective_contract.py`，仅读取三批已封存
模型预测、不读取真实指标。对固定beta=1、`prefix_goal_MSE + suffix_objective/192`，
zero-prefix已有非线性可行解给出总分上界U；某候选即使suffix代价为零，其总分
仍至少为prefix代价。因此prefix≥U的候选不能击败zero（平局按index0）。
全部24根检查中，仅全驱动202651005和202652000的**所有**非贪心候选被此界
排除；仅肩/腕各0/24。所以“stage项在所有根都结构性禁止非贪心”的解释不成立，
不得用它代替其余根的根因分析。它确实解释为什么上述新阳性common候选不能
被当前beta=1的proposal-native目标选择，但不能据此事后改beta或挑任务。
这不是D2P一般理论不可行性；原提案没有规定stage项必须取当前prefix目标MSE。
报告`results/reacher_d2p_stage_objective_contract24_20261026.json`；七项数值测试通过。
全部新增合同与历史相关回归现为**1186项通过**；这些逻辑/数值测试不充当神经
效能证据。神经证据仅上述原始官方冻结权重的授权主机CUDA与真实分支回执。

确认执行命令如下，所有神经命令均须授权主机CUDA执行；依赖报告必须先完整结束，
每个输出路径必须首次使用，不覆盖已有报告：

```bash
.venv/bin/python -u -m experiments.spectral_reachability.run_reacher_d2p_masked_program_pilot \
  --policy /home/gbk/.stable_worldmodel/checkpoints/reacher/lewm \
  --adapter-artifact results/reacher_d2p_p1_distribution_repair_v1_20261024_cuda/reacher_training_action_stats.json \
  --disjoint-reference results/reacher_d2p_masked_program_confirmation_fresh8_v2_20261026_cuda.json \
  --goal-contract joint-feasible-v2-soft-source --seed 202652000 \
  --output results/reacher_d2p_recentered_confirmation_baseline_fresh8_20261026_cuda.json

.venv/bin/python -u -m experiments.spectral_reachability.audit_reacher_d2p_masked_suffix_search \
  --reference results/reacher_d2p_recentered_confirmation_baseline_fresh8_20261026_cuda.json \
  --policy /home/gbk/.stable_worldmodel/checkpoints/reacher/lewm \
  --adapter-artifact results/reacher_d2p_p1_distribution_repair_v1_20261024_cuda/reacher_training_action_stats.json \
  --proposal-native-selector --source-disjoint-confirmation \
  --output results/reacher_d2p_recentered_confirmation_suffix_fresh8_20261026_cuda.json

.venv/bin/python -u -m experiments.spectral_reachability.audit_reacher_d2p_recentered_reach \
  --reference results/reacher_d2p_masked_suffix_confirmation_fresh8_20261026_cuda.json \
  --reference results/reacher_d2p_recentered_confirmation_suffix_fresh8_20261026_cuda.json \
  --confirmation-reference results/reacher_d2p_recentered_reach16_20261026_cuda.json \
  --policy /home/gbk/.stable_worldmodel/checkpoints/reacher/lewm \
  --adapter-artifact results/reacher_d2p_p1_distribution_repair_v1_20261024_cuda/reacher_training_action_stats.json \
  --output results/reacher_d2p_recentered_confirmation_mixed16_20261026_cuda.json

.venv/bin/python -m experiments.spectral_reachability.analyze_reacher_d2p_recentered_reach \
  --report results/reacher_d2p_recentered_confirmation_mixed16_20261026_cuda.json \
  --output-dir results/reacher_d2p_recentered_confirmation_figures16_20261026
```

工作目录`/home/gbk/code/LeWorldModel/le-wm-main`，以下神经命令只能在授权主机
CUDA执行，输出需不存在：

```bash
.venv/bin/python -u -m experiments.spectral_reachability.audit_reacher_d2p_recentered_reach \
  --reference results/reacher_d2p_masked_suffix_search8_20261026_cuda.json \
  --reference results/reacher_d2p_masked_suffix_confirmation_fresh8_20261026_cuda.json \
  --policy /home/gbk/.stable_worldmodel/checkpoints/reacher/lewm \
  --adapter-artifact results/reacher_d2p_p1_distribution_repair_v1_20261024_cuda/reacher_training_action_stats.json \
  --output results/reacher_d2p_recentered_reach16_20261026_cuda.json
```
