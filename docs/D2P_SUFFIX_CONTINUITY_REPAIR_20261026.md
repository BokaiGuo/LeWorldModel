# D2P：想象后缀丢弃与真实观测后的计划连续性修复

后续已完成同prefix/同初始动作的动作坐标审计：根202643006仅增加32次局部
迭代仍失败，但同模型CEM后缀将距离1.476133降至0.028642。说明计划连续性
之外仍有独立的数值盆地限制；详细全根证据和新增独立确认见
[动作坐标与多起点修复](D2P_SUFFIX_ACTION_CHART_AND_MULTISTART_20261026.md)。

## 当前结论

先前“负结果已经排除代码原因”的表述过强。新的主机 CUDA 全候选因果干预确认：
选前缀时精化得到的后缀被丢弃，真实观测后固定从零重编译，会在部分根上破坏原本
可执行的计划。保留该后缀作为第二个优化起点、并按当前观测的模型成本保留
incumbent，修复了这批根的一部分闭环负结果。

这不是已经解决所有 P1 排序问题。前缀的预测/真实 RGB 排序与终态后缀执行是不同
合同；本次固定前缀，不会改变旧首段排序。冻结模型的前缀反转、真实终态 RGB
排序误差、独立准入和重配置独有收益仍需分别验证，不能归结为理论被证伪。

## 代码定位与单因素干预

原 `reacher_d2p_action_words.compile_words` 对每个 imagined successor 求解 fine8
后缀，以其终点成本选择前缀，但未保存该后缀参数。实际执行前缀后，
`run_reacher_d2p_action_words` 再次调用 `compile_snapshots`，后者无初值时固定
创建全零参数。“被评分的完整动作计划”和“实际执行的完整动作计划”因此不同。

修复没有训练任何模型，也不使用 qpos/Jacobian 选动作：

1. `compile_snapshots(..., initial_params=...)` 接收形状、dtype、device一致的
   有限初值，detach/clone 后重线性化；默认调用保留历史动作与回执。
2. `reacher_d2p_suffix_continuation.compile_continuation` 对同一个真实 RGB 历史
   分别运行 cold fine8 与以原 imagined suffix 初始化的 warm fine8。
3. 按两者在该观测状态下的非线性冻结模型成本取小；平局保留 cold。warm 每步
   仍须非线性下降接受，因此其模型成本不高于 carried tail；incumbent也不高于
   cold。这是模型内保证，不是物理性能保证。
4. 原 prefix selector、非贪心 margin、权重、动作界、物理时域均不修改。

```text
compile prefix + imagined tail -> seal both
execute fixed prefix -> observe RGB
    cold: zero -> fine8
    warm: saved imagined tail -> fine8
choose min(current-model-cost(cold), current-model-cost(warm))
seal suffix choice -> execute -> evaluator measures physical outcome
```

## 已完成：全部旧根/全部词的主机 CUDA 因果实验

证据：`le-wm-main/results/reacher_d2p_suffix_continuity_all136_20261026_cuda.json`，
以及同名 protocol、selection、roots、implementation 回执。固定原8根
`202642000..007`、全部17词，共136个前缀和408条真实10-step后缀分支。
每个完整计划仍为20+10个原始物理步。

- 重核官方完整数据 SHA、动作统计与冻结权重。
- 在执行任何候选前统一重建并封存136条 imagined suffix；每条优化回执与旧版
  精确一致，未删根、未筛词。按 root identity 配对，不按数组位置。
- 每条前缀的动作、四次 RGB/action 历史更新、根/目标身份、RGB成本均精确复现；
  原 cold 后缀的优化轨迹、动作和终态关节距离全部精确复现。
- 在同一精确物理 successor 比较直接 carry、cold、warm。observed-RGB 后缀选择
  在终态结果产生前单独封存，不利用未来 RGB 或 qpos 改选择。
- RTX 5060 Ti / 主机 CUDA；无神经 CPU 回退，权重未变。实际 predict 样本67398，
  包括全部候选/诊断/物理分支，不是单个部署策略的预算。全部完整性 gate 通过。

封存 prefix 对应的全8根终态结果（距离越小越好）：

| 后缀执行方式 | 平均真实关节距离 | 对 cold 的配对结果 |
|---|---:|---|
| 原 cold fine8 | 0.4580018073 | — |
| 直接 carry 原计划 | 0.3087624834 | 7胜/1负，描述性CI跨零 |
| 一律 warm fine8 | 0.3037127143 | 7胜/1负，描述性CI跨零 |
| 模型成本选择 cold/warm incumbent | 0.1247325416 | 6胜/0负/2平 |

incumbent减cold的均差-0.3332692656；按根重采样的描述性95%CI为
[-0.6747357485,-0.0681311754]。这是已知根诊断，不是独立校准或正式显著性主张。

两个可定位病例：

- `202642000`：同prefix的cold距离1.385696，carry为0.038055，warm为0.021932；
  当前观测的模型成本分别1.835845/0.011528/0.008077，模型自己选warm。
- `202642005`：cold为0.018870，warm却为1.454644；模型成本选择cold，避免了
  一律热启动的灾难分支。保留计划不等于无条件执行旧计划。

## 对误差环节的独立分解

所有非neutral词相对同根neutral的128个margin，不把neutral平局计为正确：

| 对比 | 严格同号 | 严格反转 |
|---|---:|---:|
| 同一imagined tail：imagined状态成本 vs observed状态模型成本 | 119 | 9 |
| imagined最优tail成本 vs observed状态重新cold求解后的成本 | 68 | 60 |
| carry：observed模型终点 vs 真实终态RGB成本 | 105 | 23 |
| cold：observed模型终点 vs 真实终态RGB成本 | 106 | 22 |
| warm：observed模型终点 vs 真实终态RGB成本 | 102 | 26 |
| incumbent：observed模型终点 vs 真实终态RGB成本 | 101 | 27 |

9与60不是可加的误差分量，不能称“精确51个反转来自丢弃”。但同一前缀/相同观测
下替换初值的物理干预确实建立了部分执行失败的代码因果关系。incumbent改善所选
动作的关节距离，不意味着它改善了整个词库的真实RGB排序，后者仍有27个反转。

后处理与两张已目视检查的PNG/PDF：
`le-wm-main/results/reacher_d2p_suffix_continuity_figures_20261026/`，包含全部根结果、
全词margin分解、根配对不确定性、图注与SHA manifest。
入口 `analyze_reacher_d2p_suffix_continuity` 只读取既有证据，不是CPU替代实验。

## 已完成：8个来源episode不重叠的新根确认

`run_reacher_d2p_suffix_continuation_confirmation` 已完成主机 CUDA 确认，固定8根
`202643000..007`，source episode与此前152个不重叠，先封存全部模型选择再执行。
每根10条30-step分支：dispatch、同库greedy、neutral、flat六包GN各做
cold/continuation两个后缀；CEM另做observed重搜索与保留原CEM tail的模型成本
incumbent。CEM使用自己的后缀，不偷换成DLS。

为不改写历史compiler的实现SHA，确认脚本对dispatch/greedy/neutral所需的少量
imagined tail做确定性重建，并要求旧优化回执精确复现；额外成本单独记录。
这不是等规划预算实验，也不提出效率优势。P1 margin gate保持关闭，不因运行时
修复跳过非贪心前缀与重配置机制验证。

结果位于 `le-wm-main/results/reacher_d2p_suffix_continuation_fresh8_20261026_cuda.json`。
80条30-step物理分支全部完成，实际总predict样本64484。来源隔离、有限差分、
模型分段、重建尾部、精确预测计数、逐包历史、封存、权重和实现gate全部通过。

| 前缀/搜索族 | 原cold或重新搜索的均距 | 保留计划incumbent的均距 |
|---|---:|---:|
| 动作词dispatch | 0.4674752136 | 0.2152529155 |
| 同词库greedy | 0.6020788523 | 0.4574585261 |
| neutral | 0.6392372451 | 0.4892144406 |
| 平坦六包GN | 0.8725857696 | 0.7024615536 |
| 完整CEM | 0.0445688325 | 0.0390506401 |

dispatch修复对原cold为3胜/2负/3平；新根平均改善不能表述成逐根可靠性保证。
对同样修复后的greedy为5胜/0负/3平，均差-0.2422056107；但对CEM incumbent为
5胜/3负，均差+0.1762022753，不能声称超越强基线。五个模型witness只有4个保持
真实首段RGB非贪心，`202643004`仍反转；该根后缀也从0.009651恶化至0.037583。
另一失败根`202643006`是greedy fallback，cold/continued距离为1.470618/1.476133，
而模型自身的后缀成本仍高达1.869520/1.859524；其优化轨迹有约80%的tanh饱和
坐标。它是下一项优化盆地/动作参数chart审计的具体入口，不是已证明的理论边界。

不选择有利置信区间：不可变原报告沿用预定10000次bootstrap，dispatch修复对
cold的95%CI为[-0.6660568684,-0.0020015123]；独立后处理用同seed的20000次
复核得到[-0.6667291104,+0.0006894790]，上端符号会变化。这个小样本区间不稳健，
不构成正式显著性主张。后处理summary保留两套结果并显式注明，不能仅挑前者。

新根两张PNG/PDF、标量复核和SHA包位于
`le-wm-main/results/reacher_d2p_suffix_continuation_confirmation_figures_v2_20261026/`。
首版图包保留；v2新增bootstrap敏感性披露，不更改CUDA结果或任何模型选择。

## 复现

所有输出必须使用不存在的新路径；神经运行需主机CUDA授权。

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
.venv/bin/python -u -m experiments.spectral_reachability.audit_reacher_d2p_suffix_continuity \
  --reference results/reacher_d2p_action_words_fresh8_20261026_cuda.json \
  --policy /home/gbk/.stable_worldmodel/checkpoints/reacher/lewm \
  --adapter-artifact results/reacher_d2p_p1_distribution_repair_v1_20261024_cuda/reacher_training_action_stats.json \
  --output /absolute/new-suffix-continuity.json
.venv/bin/python -m experiments.spectral_reachability.analyze_reacher_d2p_suffix_continuity \
  --report /absolute/new-suffix-continuity.json --output-dir /absolute/new-continuity-figures
.venv/bin/python -u -m experiments.spectral_reachability.run_reacher_d2p_suffix_continuation_confirmation \
  --policy /home/gbk/.stable_worldmodel/checkpoints/reacher/lewm \
  --adapter-artifact results/reacher_d2p_p1_distribution_repair_v1_20261024_cuda/reacher_training_action_stats.json \
  --disjoint-reference results/reacher_d2p_action_words_fresh8_20261026_cuda.json \
  --seed 202643000 --roots 8 --output /absolute/new-continuation-confirmation.json
.venv/bin/python -m experiments.spectral_reachability.analyze_reacher_d2p_suffix_continuation_confirmation \
  --report /absolute/new-continuation-confirmation.json --output-dir /absolute/new-confirmation-figures
```

最新相关广覆盖回归482项通过。逻辑/代数fixture仅验证实现，不是
神经CPU实验，不替代CUDA证据；正式P1及完整研究闭环仍未完成。
