# D2P：有限位关节被错误环绕的目标构造缺陷

## 代码根因，不是模型可靠性结论

`experiments/run_live_reacher_planning.py`中的历史目标事务对两个qpos都执行
`(qpos + delta + pi) % (2*pi) - pi`。但实际DMControl Reacher的shoulder连续、
wrist有限位±160°（±2.7925268032 rad）。wrist不是允许穿过限位的圆周。
源XML与HDF5逐根元数据检查发现以下两例：

| 原始根 | 源wrist | 请求偏移 | 历史目标wrist | 问题 |
|---|---:|---:|---:|---|
| 202644006 | -2.6023704185 | -0.65 | +3.0308148887 | 超过+160°限位 |
| 202646001 | -2.5009439938 | -0.65 | +3.1322413133 | 超过+160°限位 |

二者恰为各自8根批中终态误差最大的根。其余14个根目标合法。该关联及实际
越界由源码、官方HDF5和实际编译模型验证，主机CUDA配对干预也已完整完成，
两根严重误差均明显恢复（数值见下表）。MuJoCo限位为物理约束而非此处的可达性
定理，不借此声称任意时间/任意控制的数学不可能性。

该代码事实覆盖此前把这些严重失败统称checkpoint可靠性边界的归因。历史结果
和所有失败根保留；合法目标上的首段RGB排序反转也依然存在，不能用这个缺陷
解释全部P1失败。

## 显式修复版本与不可变协议

新增`reacher_d2p_feasible_goal.ReacherFeasibleGoalImageWrapper`，不静默改历史
wrapper、目标、哈希或结果。目标生成使用实际编译模型的joint类型、地址、
`jnt_limited/jnt_range`，而不是假设两个坐标都周期：

1. 连续shoulder仍按原角度环绕。
2. 有限位wrist若`q+delta`合法，保留原delta的每一位；否则若`q-delta`合法，
   翻转方向但保留完整偏移幅度。不裁剪为更近目标，不删除根。
3. 根本身越界、两个完整方向都越界或不是两标量hinge模型时fail closed。
4. 复用原render/restore事务；这些物理元数据只用于环境任务构造，规划器仍只收
   RGB和实际动作历史，没有新增qpos readout、adapter、policy或训练。

这是新任务协议，不是同任务性能改善，不能把新goal的正结果追认成旧P1。
历史qpos evaluator也对两个坐标环绕，其有限位wrist解释需谨慎；本干预保留
原指标以隔离目标构造差异，不趁机换metric制造通过。

主机CUDA审计`audit_reacher_d2p_goal_joint_limits`已完整完成，全部门通过：

- 全部16原始根、旧/新目标共32条完整30步分支，固定原CEM128×3/16elite。
- 全32条完整动作先统一封存，再测任何终态；不按outcome挑选目标规则。
- 原16个CEM轨迹、动作、根/goal RGB及终态必须精确复现。
- 未改任务的14根，其输入图像、搜索轨迹、动作和终态必须精确不变。
- SHA绑定原权重、动作stats、官方完整source与实现，CUDA神经执行，禁止CPU回退。

## 配对干预结果

证据输出：`le-wm-main/results/reacher_d2p_goal_joint_limits_all16_v2_20261026_cuda.json`。
全16根32条完整30步分支、73728个实际predict样本，原权重不变、无CPU神经回退。
原16根的CEM搜索轨迹、动作及完整物理终态全部精确复现；未改目标的14根，
root/goal/history RGB、搜索轨迹、动作、物理终态均精确不变。

| 根 | 旧任务关节误差 | 合法目标任务误差 | 旧终态RGB成本 | 新终态RGB成本 |
|---|---:|---:|---:|---:|
| 202644006 | 0.8313750442 | 0.0756297241 | 0.8334125280 | 0.0378283560 |
| 202646001 | 0.7537108360 | 0.0865524211 | 0.7453063726 | 0.0544506833 |

全16根平均关节误差0.1475055983→0.0585741149。表中比较是显式改变非法目标
任务后的诊断，不是同任务规划器优势；不提供有误导性的正式效能p值或P1 pass。
冻结模型已经能在这两个来源状态的合法同幅度目标上控制到较小误差，因此此前
把该严重失败直接归为模型可靠性边界的说法不成立。并未证明所有历史失败来自
同一缺陷，也未解决合法目标202646007的首段RGB排序反转。

## 与用户引用的原6根margin失败的关系

后续已在固定原六根/动作下定位另一原因：joint-repair的预测头BN运行统计漂移；
保留其全部学习参数而恢复官方统计即可从4/6恢复6/6。此结论不来自关节限位
干预，详情见[预测归一化修复](D2P_PREDICTION_NORMALIZATION_REPAIR_20261026.md)。
下述元数据重放仍保留，不能继续据此说原两个反转尚无任何代码级定位。

另用`audit_reacher_d2p_calibration_goal_limits`精确重放原始margin校准
20274800..005的root/goal RGB和warmup动作，核对实际编译关节范围。全6根目标
均合法，原两个排序反转根20274802/20274804不受本次限位修复影响。
证据：`le-wm-main/results/reacher_d2p_original_calibration_goal_limits_20261026_cuda.json`。
该项是CUDA可分配前提下的模拟器/图像身份/目标元数据核对，原排序证据按SHA
链接而非重新执行神经预测；不把它说成新的神经效能结果。

因此仅凭关节限位修复不能解释原六根；后续预测头BN干预才在固定原六根上消除了
两个已知反转。两项原因独立，均不能升级为P1通过或理论否定。

独立后处理`analyze_reacher_d2p_goal_joint_limits`从官方HDF5的原始root qpos
重算每条goal admission，对齐全部16根/32条封存决定，验证实际joint范围、
目标、动作、原基线重放、14根不变性与compute。两张全根PNG/PDF、summary、
图注及已核验SHA manifest见
`le-wm-main/results/reacher_d2p_goal_joint_limits_figures_20261026/`，均已目视检查。
限位/绑定/后处理26项回归通过；最新广覆盖684项通过，`git diff --check`无错误。

首次新入口在创建任何输出前误把自身auditor SHA与旧auditor比较，已排除自身
并保持核心SHA检查。随后一次在首根结果前发现已知根审计继承CEM封存、而新根
确认直接持有CEM封存，两种schema不能同读；已增加hash-bound继承解析及5项
绑定回归。该中断的原protocol/implementation保留，v2使用新输出路径；未覆盖
旧科学结果，也没有放宽任何执行门。

## 使用边界与后续准入

### 合法目标新八根确认：完整运行与独立复核

`results/reacher_d2p_feasible_goal_confirmation_fresh8_20261026_cuda.json`已完成：
固定新根202647000..007，8个唯一source episode，排除此前176个episode；使用
官方原始冻结权重、显式`joint-feasible-v1`，逐根验证实际simulator目标和joint范围。
全136动作词、40个后缀决定分别封存后，执行全部128条30步终态分支，398540个
实际predict样本，14项完整性门全部通过，无CPU神经回退。

本批8个原偏移均合法，**没有目标翻转**，因此它验证新任务准入在新来源根上的
运行及现存数值runtime，不是限位反射相对旧目标的效能消融，更不是BN修复的
泛化实验（本批没有使用joint-repair权重）。

| 完整分支 | 全八根平均关节误差 |
|---|---:|
|历史prefix/历史runtime|0.0284481211|
|历史prefix/native后缀|0.0336716360|
|native prefix/native后缀|0.0328023265|
|同库greedy/native后缀|0.0327610261|
|neutral/native后缀|0.0289083538|
|完整CEM prefix/native后缀|0.0248581260|

native dispatch对同库greedy为2胜/4负/2平，均差+0.0000413004，描述性95%根
bootstrap区间[-0.0110819662,+0.0103568487]；对CEM为4胜/4负，区间同样跨零。
规划计算并不相等，不把这些pilot区间升级成正式效能/无效证明。

6个新模型witness中5个实际首段RGB非贪心。剩余根202647003模型margin
+0.0165028572，真实RGB为-0.0516061783；目标合法，全部根保留，不调阈值。
原六根的BN原因不能解释这条官方原始权重、多包前缀的新失配，也不能把这条
新反转直接归为理论失败。P1机制前提、独立校准及outcome仍未完成。

独立`analyze_reacher_d2p_state_native_selector`已按root/family身份重算全部决定、
动作与指标，核验protocol/prefix/suffix封存和每根goal admission。全八根/十七词
PNG/PDF、图注、summary及SHA manifest位于
`results/reacher_d2p_feasible_goal_confirmation_figures_20261026/`，两张图均已目视，
产物哈希全部通过。相关广覆盖757项通过，`git diff --check`无错误。

历史`ReacherGoalImageWrapper`仍用于精确回放，不能继续把它当作全域合法目标
生成器。后续新任务必须显式采用`ReacherFeasibleGoalImageWrapper`并绑定其代码、
joint-range与goal-admission回执。不得把这次对已知根的新任务修复冒充独立校准；
P0/FD、P1校准及outcome必须在明确的新任务合同下重新封存。当前P1及研究总
闭环仍未完成；合法目标上的排序负结果保持开放诊断，不升级为理论否定。

## 复现

神经入口须获授权在主机CUDA执行，输出及sidecars须不存在：

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
.venv/bin/python -u -m experiments.spectral_reachability.audit_reacher_d2p_goal_joint_limits \
  --reference results/reacher_d2p_state_native_selector_all136_20261026_cuda.json \
  --reference results/reacher_d2p_state_native_confirmation_fresh8_20261026_cuda.json \
  --policy /home/gbk/.stable_worldmodel/checkpoints/reacher/lewm \
  --adapter-artifact results/reacher_d2p_p1_distribution_repair_v1_20261024_cuda/reacher_training_action_stats.json \
  --output /absolute/new-goal-joint-limits-audit.json
.venv/bin/python -m experiments.spectral_reachability.analyze_reacher_d2p_goal_joint_limits \
  --report /absolute/new-goal-joint-limits-audit.json \
  --adapter-artifact results/reacher_d2p_p1_distribution_repair_v1_20261024_cuda/reacher_training_action_stats.json \
  --output-dir /absolute/new-goal-joint-limits-figures
.venv/bin/python -u -m experiments.spectral_reachability.run_reacher_d2p_state_native_confirmation \
  --goal-contract joint-feasible-v1 \
  --policy /home/gbk/.stable_worldmodel/checkpoints/reacher/lewm \
  --adapter-artifact results/reacher_d2p_p1_distribution_repair_v1_20261024_cuda/reacher_training_action_stats.json \
  --alignment-reference results/reacher_d2p_state_native_selector_all136_20261026_cuda.json \
  --disjoint-reference results/reacher_d2p_state_native_confirmation_fresh8_20261026_cuda.json \
  --seed 202647000 --roots 8 --output /absolute/new-feasible-goal-confirmation.json
.venv/bin/python -m experiments.spectral_reachability.analyze_reacher_d2p_state_native_selector \
  --report /absolute/new-feasible-goal-confirmation.json \
  --output-dir /absolute/new-feasible-goal-confirmation-figures
```
