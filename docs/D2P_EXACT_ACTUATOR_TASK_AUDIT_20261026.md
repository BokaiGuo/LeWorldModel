# D2P 严格执行器子空间：代码合同与真实任务前提

## 结论边界

不能把旧的 `weak_shoulder=(0.05,1)` / `weak_elbow=(1,0.05)` 当成关闭一个
执行器。它们仍是双驱动任务，第二关节实际为 wrist；旧报告仍保留其原名与原合同。
本轮新增明确的动作合同，不回填历史结果，不宣称它解释全部历史 RGB 排序反转。

官方原始冻结 checkpoint 上，严格单驱动确实使短时物理响应条件变差。全部八个新根、
两时域、两幅度都保留，未按结果选择任务或根。这支持继续进行重配置机制试验，
但没有证明某方向数学不可控、非贪心前缀有效或 P1 已通过。

## 代码级实现与审计

新增 `reacher_d2p_masked_action_chart.py`：

- 全驱动：每包十个原始力矩坐标；仅肩/仅腕：每包五个坐标，按真实时间顺序散射。
- 未启用执行器的**原始力矩严格为 0**，不是 5% 力矩，也不是归一化坐标为 0。
  字面原始零归一化后是 `-mean/std`；与历史动作采用同一统计、只归一化一次。
- 真实三帧/三动作历史不屏蔽；任务限制只作用于规划根之后的未来动作。
- 不用上下界相等的退化 affine box；模型 Jacobian 精确等于完整原始动作
  Jacobian 对应的列，最大允许差 `2e-6`。
- 未来一包、两包递推均使用原始官方 predictor、完整 192 维编码，不训练额外模块。

新增 `run_reacher_d2p_masked_task_preflight.py`：先封存全部八根、全部三任务的
模型预测，才执行任何未来物理动作。每根一包为 41 条、两包为 81 条真实未来，
共 976 条；两幅度为 0.1/0.25，全部原始动作坐标均测正负方向。单驱动只能复用
其活动坐标上的已执行探针，不推断任何未执行的复合动作结果。

八根 `202649000..007` 的来源 episode 为
`2356,6859,8467,2024,3060,5207,6839,1429`，与此前 192 个 episode 无重叠。
来源由元数据在读取新结果前确定，完整源文件 SHA 使用已有可追溯回执，不对 93 GB
文件重新整块读入内存。

## 主机 CUDA 与独立复算

主报告：
`le-wm-main/results/reacher_d2p_masked_task_preflight_fresh8_20261026_cuda.json`

全部 976 条真实未来完成，**2,272 个 CUDA predict 样本、12 项完整性门全部通过**。
原始权重 SHA、模型参数、全部运行 buffer、eval 状态和封存预测均未变化。
官方权重 SHA 为 `eb70b1fd5409f8f81875d62f5ee5a20dd220a3128a477de66b5760f475f0f469`。

独立 NumPy/SciPy 分析器 `analyze_reacher_d2p_masked_task_preflight.py` 校验：
根身份集合、源 episode、合法关节目标、全部原始力矩、逐包动作 SHA、完整端点、
方向有限差分、模型 AD 子列、物理 RGB/qpos 响应，以及 384 次有界线性拟合。
有限位 wrist 的关节误差使用直接差，只有周期肩关节包裹角度；这不是修改 RGB
判据，也不是把关节标签提供给规划器。实际方向 FD 最大相对误差为 0.013623。

下表为全八根均值；幅度 0.25，物理响应对原始力矩求差分。最小奇异值单位依赖
所选物理坐标；不同 horizon 不应直接当成同一控制器的成功率比较。

| 任务 | 一包最小响应奇异值 | 两包最小响应奇异值 | 两包有界线性拟合剩余目标残差比例 |
|---|---:|---:|---:|
| 全驱动 | 0.274672 | 0.489252 | 0.092225 |
| 仅肩 | 0.022857 | 0.032425 | 0.559978 |
| 仅腕 | 0.013982 | 0.024709 | 0.478930 |

单驱动与全驱动最小响应奇异值比值的根间中位数，一包约 0.07747/0.04884，
两包约 0.05914/0.04699。所有数值均非零；不能写成已经发现“不可控维度”。
有界线性拟合使用相应幅度的原始力矩 box、固定阻尼 0.01，仅作 evaluator 诊断，
**拟合动作没有被实际执行，不是闭环达成目标的证据**。

双图与完整分析产物位于：
`le-wm-main/results/reacher_d2p_masked_task_preflight_figures_fresh8_v2_20261026/`。
`Figure_1` 展示全部根的实际响应谱，`Figure_2` 展示全部根的线性残差诊断；PNG/PDF
均生成，目视检查后修正首版过长轴标签，保留首版文件、不覆盖原始实验。

## 后续机制试验的封存合同

另一组八根 `202650000..007` 已仅依源元数据确定，来源 episode 为
`7235,5406,4095,4268,4632,6353,6733,6440`，与前述全部 200 个 episode 隔离。
不选择某个有利任务：仍在全驱动/仅肩/仅腕全部运行。

新增 masked-program compiler 与 runner，固定每任务 17 个谱前缀（零、前四个
响应右奇异方向、正负号、幅度 0.25/0.75），两包后缀使用固定四步带证书 box GN。
比较同库 greedy、flat GN 与带 GN incumbent 的 flat CEM；后者不能冒称纯 CEM
或等计算量基线。选择重配置必须同时满足：

1. 比字面零前缀的预测目标误差更大，差超过固定 `1e-6`；
2. 完整预测终点优于同库 greedy，差超过固定 `1e-6`；
3. 公共目标残差上的有界响应增益下界严格超过零/greedy 的增益上界。

分段 imagined-prefix/suffix 必须与**同一完整三包动作**的递推逐位相同。
全部 17 个程序与两个 flat 对照在物理执行前封存，保留所有 456 条真实分支；
不只执行“通过”的根，不用真实结果调 selector。这里的共同残差增益资格是显式
诊断策略，不偷偷替换原 P1 的校准/效能门，也不通过移除该条件事后挑出成功根。

## 完整动作程序审计已完成

最终主报告：
`le-wm-main/results/reacher_d2p_masked_program_pilot_fresh8_v2_20261026_cuda.json`。
全部 8 根 × 3 任务 × 19 分支 = **456 条真实执行**、1,368 个物理动作包、
59,568 个 CUDA predict 样本，**14 项完整性门全部通过**。全部 408 个谱程序
的分段预测与完整三包预测逐位一致，所有原始动作符合严格执行器屏蔽。
全部预测和选择在任何未来物理结果产生前封存。

| 任务 | 模型合格重配置根 | 非零前缀RGB排序符号一致 | native/greedy平均真实关节距离 | flat GN | flat CEM + GN incumbent |
|---|---:|---:|---:|---:|---:|
| 全驱动 | 0/8 | 94/128 | 0.227556 | 0.865052 | 0.028534 |
| 仅肩 | 0/8 | 77/128 | 0.548005 | 0.911161 | 0.552913 |
| 仅腕 | 0/8 | 91/128 | 0.681375 | 0.842350 | 0.659048 |

同一根的16个非零前缀互相依赖，128不能当成128个独立样本；符号统计保留全部
margin，包括接近零的情况，不是新的稳健校准通过率。native全部回退同库greedy，
不能将greedy收益写成重配置收益。预算不相等，上表不是正式效能排名，有GN
incumbent的对照也不是纯CEM。

| 任务 | 预测非贪心候选 | 终点优于greedy | 共同响应增益合格 | 非贪心且终点更好 | 三条件同时满足 |
|---|---:|---:|---:|---:|---:|
| 全驱动 | 43 | 9 | 13 | 1 | 0 |
| 仅肩 | 51 | 4 | 1 | 0 | 0 |
| 仅腕 | 47 | 15 | 18 | 3 | 0 |

这不是“执行了合格重配置程序但机制被物理否定”，而是**固定候选生成及资格条件
没有产生该程序**。短时条件变差不能直接推出这17个前缀应当成功。
全驱动202650004的greedy距离1.610768；预测两条件候选bank15可达0.611236，
但其真实前缀比zero更贪心（margin=-0.001090），不构成非贪心见证。仅腕
202650001的bank3/7保留真实非贪心，并有很小终点收益，但不满足共同响应增益
资格；bank15则发生首段排序反转。全部候选保留，不事后更换selector。

上述结果继续要求检查候选覆盖、后缀数值盆地、状态相关残差与真实后继响应。
4步从零GN的失败不能当作模型不知道有效动作：同一冻结模型下，全驱动flat GN
与CEM+GN incumbent均距0.865052/0.028534已表明搜索路径显著影响结果。当前
reduced-chart小预算诊断不替换已有多起点/carry默认实现，不把已知冷启动限制
重新包装成理论负结果；P1/P2/P3/P5仍不开放。

分析器独立校验完整192维成本、原始动作、逐包SHA、关节拓扑误差和model-only
选择。**本轮未保存响应矩阵，增益只复核CUDA证书代数，不宣称另做SciPy增益
求解**；预检的384个独立SciPy拟合是另一份证据。最终双图位于
`le-wm-main/results/reacher_d2p_masked_program_pilot_figures_fresh8_v3_20261026/`。
PNG/PDF均已目视检查，使用共同非负距离轴，全部产物SHA已复核。

## 新审计writer错误的独立归因

首版`reacher_d2p_masked_program_pilot_fresh8_20261026_cuda.json`保留`status=fail`：
实际逐根执行正确，但writer误用上一循环最后的root，全部八行身份标成202650007。
门准确拒绝了它，失败报告、全部sidecar和实现副本均保留。

修复为从当前封存record显式提取seed/episode/start，增加跨八根所有权回归，随后
在同根、同动作/目标/阈值/模型上完整主机CUDA重跑，不事后改写元数据。
`results/reacher_d2p_masked_program_writer_repair_20261026.json`独立确认：协议仅
writer实现绑定改变；全部模型预测/动作/决定除walltime外不变；两次456条分支及
1,368个packet的RGB SHA、原始动作、qpos与完整192维观察逐位相同。两次实际
CUDA predict合计119,136个。这仅解释新writer失败，不冒充历史P1的RGB根因。

最新相关全覆盖**989项测试通过**；新增83项为synthetic合同/反篡改检查，不能
替代实际CUDA与物理效能证据。研究目标仍未完整闭环。

## 复现命令

从 `le-wm-main` 执行；神经实验须在已授权的主机 CUDA 环境运行，输出路径必须不存在。

```bash
.venv/bin/python -u -m experiments.spectral_reachability.run_reacher_d2p_masked_task_preflight \
  --policy /home/gbk/.stable_worldmodel/checkpoints/reacher/lewm \
  --adapter-artifact results/reacher_d2p_p1_distribution_repair_v1_20261024_cuda/reacher_training_action_stats.json \
  --disjoint-reference results/reacher_d2p_waypoint_confirmation_fresh8_20261026_cuda/planning.json \
  --seed 202649000 \
  --output results/reacher_d2p_masked_task_preflight_fresh8_20261026_cuda.json

.venv/bin/python -m experiments.spectral_reachability.analyze_reacher_d2p_masked_task_preflight \
  --report results/reacher_d2p_masked_task_preflight_fresh8_20261026_cuda.json \
  --output-dir results/reacher_d2p_masked_task_preflight_figures_fresh8_v2_20261026

.venv/bin/python -u -m experiments.spectral_reachability.run_reacher_d2p_masked_program_pilot \
  --policy /home/gbk/.stable_worldmodel/checkpoints/reacher/lewm \
  --adapter-artifact results/reacher_d2p_p1_distribution_repair_v1_20261024_cuda/reacher_training_action_stats.json \
  --disjoint-reference results/reacher_d2p_masked_task_preflight_fresh8_20261026_cuda.json \
  --seed 202650000 \
  --output results/reacher_d2p_masked_program_pilot_fresh8_v2_20261026_cuda.json

.venv/bin/python -m experiments.spectral_reachability.analyze_reacher_d2p_masked_program_pilot \
  --report results/reacher_d2p_masked_program_pilot_fresh8_v2_20261026_cuda.json \
  --output-dir results/reacher_d2p_masked_program_pilot_figures_fresh8_v3_20261026

.venv/bin/python -m experiments.spectral_reachability.audit_reacher_d2p_masked_program_writer_repair \
  --failed-report results/reacher_d2p_masked_program_pilot_fresh8_20261026_cuda.json \
  --repaired-report results/reacher_d2p_masked_program_pilot_fresh8_v2_20261026_cuda.json \
  --output results/reacher_d2p_masked_program_writer_repair_20261026.json

.venv/bin/python -m pytest -q \
  tests/test_reacher_d2p_masked_action_chart.py \
  tests/test_analyze_reacher_d2p_masked_task_preflight.py \
  tests/test_reacher_d2p_masked_programs.py \
  tests/test_analyze_reacher_d2p_masked_program_pilot.py \
  tests/test_reacher_d2p_masked_program_writer_repair.py
```

上述 83 项为 synthetic 合同测试，不能替代实际 CUDA/物理效能证据。

## 最新后续：固定前缀求解与软限位源状态

后续全八根/三任务/17前缀的主机CUDA配对干预已完成：1224条真实分支、349320
predict样本、16门全通过，原408分支逐位重放。保留原解的CEM后GN4显著改善
全前缀集合的真实终点，但旧选择器整体未改善，不能由此开放P1。另将原提案的
own-state reach cost与common-gain诊断分开，显式新增独立选择合同。来源隔离
确认发现源数据真实软限位状态被旧起点检查错误拒绝，已保留失败、全部根和完整
偏移，用显式v2任务合同重跑；16根源状态/渲染审计已通过，15个原有效上下文
逐位不变，唯一拒绝精确复现。来源隔离新八根现已完成全部456基线及1224后缀
分支，14/16门均通过；全驱动/仅肩全前缀均距改善在新来源复现，但新选择合同
没有非贪心见证，P1仍未通过。另已修复两个数值保护漏洞，16根三任务CUDA差分
回归保持正常输入下原动作/预测逐位不变，不冒称历史反转根因。最新1072项回归
通过。详见
[后缀求解与残差合同审计](D2P_MASKED_SUFFIX_SEARCH_AUDIT_20261026.md)。

