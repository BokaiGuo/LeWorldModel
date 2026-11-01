# P1 fixed40 任务指标机制面板（saved-only）

独占新增source/tests/本文及独占saved报告，不改旧封存、MAIN或共享文档。继承goal为旧blocked P4，不能覆盖或完成它；本有界交付为实现、RAM回归、真实saved analyze/verify及冻结。使用experiment-code技能组织验证，但用户限制优先：无神经、物理、网络、拟合、筛根或阈值调整。测试不是科学证据。

## 输入与固定合同

唯一上层输入为已验收`agent_p1_teacher_library40_task_metric_20261028_v1_analysis_NASH01.json`，SHA `52c0739d8e2b6f996dc751bfe3f085074549120cfa0355eb6283e80d653d6373`。重新调用其完整saved build，继承并验证全部原JSON、prepare、模型、源码、数组、物理artifact/journal与native validator闭包，而非只信上层summary。新报告再绑定本三文件及各boundary实际NPZ数组字节SHA。原Q/goal完全不变。

40槽严格4task（root0off/on、root1off/on）×2role(Z/S)×5word原顺序。输入只有2个known roots，不是40个独立根；没有fresh确认。index12只有terminal-on RGB/latent缺失，actual Q保持null；真实20步native返回完整，任务指标可读，绝不补图或用off代on。全部成功值原样报告，不成功过滤。

固定规则在本分析实现中声明，不是追溯声称在观察旧结果前预注册。每个同task十槽组全部45对，共180对（四组，不跨根/condition）。8个role内五词组共80对及同词Z/S20对是前者重叠视图，不能把它们相加当独立样本。跨role不同word的80对也完整保留，但prefix/context/suffix都可能不同，绝非隔离因果对照。各对显式列身份、词、raw后缀是否相同、缺失槽。原role/sameword匹配器仍严格复用，不放宽旧matching API。

每对报告predicted Q→actual Q、predicted Q→native metric、actual Q→native metric，以及Q与各段物理变化的方向。逐指标输出精确delta、取向与符号分类；distance/误差lower-better，success/coverage higher-better只反转取向。零tie不是反转；unknown不填0。分类含concordant、discordant、both/left/right tie、unknown，完整fixed/available/unknown/ties/nonzero分母。没有合成加权分数或winner。

## 真实首段/终段与不可识别边界

从每槽真实20条raw_return中读取warmup10（绝对10）、prefix15（绝对15）、suffix5（绝对20）三边界，精确核phase/index/未提前终止、真实info NPZ的goal_state7、goal_pose3、完整unwrapped block_pose、agent position/velocity和state。每边界用原whole-body wrapper合同重算真实返回success/distance/reward；终边界必须与已验收分析整个physical_metrics字典完全一致。首段=10→15，终段=15→20，总段=10→20。每指标输出raw end-minus-start与取向后的变化，不凭终态捏造首段。

真实saved接线确认：原episode在warmup10完成后才bind task goal，因此该条raw_return的goal_state仍为reset原生goal，不能要求它已等于task goal或直接拿其reward作首段基线。面板单列该步`returned_goal/returned_native_metrics/native_reward`，再以相同实际state、原task goal和原wrapper算术计算`metrics`；并要求与真实post-bind root checkpoint0的native success/distance、root_state完全一致。prefix15/suffix5必须原返回goal_state严格等于task goal。该基线是saved算术不是新环境调用，不改原goal/状态/评分；返回指标与重算指标不混称。

可识别的是各真实轨迹每个指标的“首段变差且终段变好”描述性布尔值；这不等于真正物理非贪心控制或能力改变。缺少匹配的能力干预/贪心反事实或局部可控性测量，所有40槽逐指标`truly_physical_non_greedy`、`capability_change`必须null，不能从末态或单一object投影推断。不同指标分别保留，不能挑有利子模式认定机制。

成功使用原agent+object联合XY与原angle公式，不拿object-only误差当完整任务失败。原success在全库均false，因而所有success对均ties（涉及缺latent的latent→success轴仍unknown）。native/object-only coverage原authority没有返回，全部null；不另造覆盖几何。latent失配不推断C2失败。P1=false、无physical release、无科学确认。

## 命令与验收

工作目录 `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p1_task_metric_mechanism_panel_20261028_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.agent_p1_task_metric_mechanism_panel_20261028_v1 analyze --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p1_task_metric_mechanism_panel_20261028_v1_analysis_MAIN01.json
.venv/bin/python -B -m experiments.spectral_reachability.agent_p1_task_metric_mechanism_panel_20261028_v1 verify --analysis /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p1_task_metric_mechanism_panel_20261028_v1_analysis_NASH02.json --analysis-sha <实际报告SHA>
```

只独占新建，不覆盖已有报告；verify重建完整内容再比较。输出上限32MiB。精确未决依赖为index12缺失原on latent、原native coverage authority，以及识别机制所需的前瞻干预/反事实；不能用新生成伪补历史证据。MAIN负责最终审读，本工具无任何实验dispatch入口。

本次工程保留：NASH01分析文件已写出但verify因本新代码的tuple/list JSON round-trip不等而失败，原文件不改不删除、不能作为最终验收。冻结前将段合同统一为JSON原生list并增加完整panel+protocol round-trip回归，最终saved交付使用独占NASH02。此为工程失败保留，不是模型/物理结果改变。
