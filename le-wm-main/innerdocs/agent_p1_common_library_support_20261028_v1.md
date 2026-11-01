# 共同五词库有限支持与部署交叉项（saved-only）

本独占stem为原最小机制试验的结果分析，不新建实验、不做泛化代码审计。只新增source/tests/本文及独占saved结果，不修改MAIN、封存源码或已有结果。继承旧blocked P4 goal不覆盖；有界个人交付为代码、RAM测试、实际saved analyze/verify和冻结。使用experiment-code技能组织验证，用户saved-only边界优先，不生成神经/物理、图像、网络、训练/Q拟合或新评分门槛。

## Authority与闭包

唯一40/180输入为最终`agent_p1_task_metric_mechanism_panel_20261028_v1_analysis_NASH02.json`，SHA `7eda1333a3bf2c8ef1d46deaf67d74c6d464498c018b52cf35e9802204501e5d`；不是失败的NASH01。重新完整调用panel.build，继承native consume、原始数组、action/history、模型/prepare/源码/artifact/journal全闭包。绑定原`innerdocs/agent_p1_next_mechanism_test_20261028_v1.md` SHA `622bc124a5b08f787acac569f7b1875ed1424508d87e9a35bc0670b819d06423`，复用其中指定的旧feature/wrap/task_direction纯算术并核其源码已在闭包。

再次核原compiler strict receipt40plans与每行task/anchor/word/plan身份及完整float32 raw10、prefix5、suffix5字节。共同库严格为teacher.LIBRARY原顺序`zero,uZ,-uZ,uS,-uS`；zero必须正0 float32，反向为literal raw精确符号翻转，不时间逆序、不重新tanh/缩放/clip。两角色必须是完全相同五个suffix buffers，重复/碰撞保留不去重。封存raw10/suffix5数组SHA与上游实际模型生成receipt；不能用末态重选动作。

## 四轴支持：完整五值，不取缺失子集

每task原root10→原task goal方向，固定对Z/S共用。四轴`joint_xy`（agentXY+objectXY四维）、`agent_xy`、`object_xy`和单独`object_angle_wrapped`。使用原数学的wrapped root→goal角差和wrapped endpoint−zero角响应，单位为px或rad，不混合加权。原root不是prefix15状态，更不是after残差。

每role五项独立实际suffix endpoint20相对自身zero endpoint20的响应`r_p(b)`，全部输出vector与`dot(d,r_p(b))`，含zero项。支持`H_p=max(all five projections)`只作为 **evaluation upper envelope**，输出所有精确并列attaining indices，不输出可部署winner。任一必需endpoint（含zero）缺失或原方向完全为0，support=null，不对已完成子集取max，也不从after residual补方向。全部五投影均保留，未知明确null。逐task报告H_Z、H_S和H_S−H_Z，各轴独立。

全40物理末态真实存在，包括index12；该槽缺terminal-on RGB/latent所以actual_Q保持null，但不能抹掉其真实state。物理支持不依赖latent可用性。若未来合法输入缺物理state，纯算术接口维持40槽并将相关支持置null；生产输入仍必须原NASH02完整重建，不能用户任意换报告。native coverage没有authority，相关收益/upper envelope保持null。

## 模型部署对角线与交叉项

每task固定列`(Z,uZ)`、`(S,uS)`及交叉`(Z,uS)`、`(S,uZ)`，索引分别1/8/3/6（相对task十槽）。这些是事前模型生成的两条正向suffix，绝不按物理收益挑反向词或upper envelope。报告原predicted Q/actual latent Q、原物理指标/returned reward、四轴响应，以及每项相对**自身role zero suffix**的真实任务收益。

所有40行同样报告zero对照：误差收益=zero误差−candidate误差，success/coverage收益=candidate−zero；reward增量单列，Q下降另列而非物理收益。joint/agent/object/angle/完整state7等原轴不拟合综合分。成功是原whole-body合同，不能用object-only投影替代。原step10/15/20信息和first/last段变化整行保留；warmup10旧returned goal/reward与bind后当前taskgoal基线仍严格分栏，不重贴旧reward。

## 解释限制与分母

固定4task、8role、40槽、160投影，只有2known roots，没有增加独立样本或fresh认证；index12 latent缺失，成功全ties保留。有限物理oracle只是evaluation upper envelope，不是planner、连续动作域可控集合证书、可部署策略或非贪心前段见证。H增大不证明能力改变，更不意味着P1准入。位置/速度/接触/残差中介未隔离；真正物理非贪心/能力改变仍unknown。P1=false、无physical release、无C2失败推断。只有原库与原根的描述性结果，不调科学阈值。

## Saved CLI

目录 `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p1_common_library_support_20261028_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.agent_p1_common_library_support_20261028_v1 analyze --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p1_common_library_support_20261028_v1_analysis_MAIN01.json
.venv/bin/python -B -m experiments.spectral_reachability.agent_p1_common_library_support_20261028_v1 verify --analysis /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p1_common_library_support_20261028_v1_analysis_NASH01.json --analysis-sha <实际NASH01报告SHA>
```

只独占新建输出，不覆盖任何结果；上限32MiB。verify独立重建全内容而非仅日志/文件hash。RAM回归不是科学结果；最终SHA和实际测试终态随交付发出，冻结后不原地修。本轮不打开reserved4fresh roots。
