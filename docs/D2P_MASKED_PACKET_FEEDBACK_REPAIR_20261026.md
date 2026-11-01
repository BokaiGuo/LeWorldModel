# D2P：逐包排序定位与真实RGB反馈执行修正

后续已完成[因果上一包残差补偿对照](D2P_CAUSAL_RESIDUAL_AUDIT_20261026.md)：
576条真实分支、29376个CUDA样本、12门全通过，原三方式432分支精确复现。
固定动作上的004预测margin纠正没有转化为同反馈greedy之上的执行收益；三个
任务均无稳定补偿收益，未作为P1修复。下述原反馈结果和证据边界保持不变。

## 本轮改变了什么

本轮不仅重复诊断，还实际执行了保留原前缀的反馈修正。官方原始冻结checkpoint、
全部16个已知根、三种严格执行器掩码、原共同残差选择的native/同库greedy/零前缀
全部保留。关键根202651004原终态native劣于greedy，在对两者实施同样的反馈后
转为native更好。但这是动作改变后的执行收益，不是原同动作预测排序已恢复；
不开放P1，不回填历史outcome，也不声称全部剩余负结果都是代码错误。

## 固定动作逐包定位：已完成

权威回执：
`le-wm-main/results/reacher_d2p_masked_packet_fidelity16_20261026_cuda.json`。

根202650000..007及202651000..007，沿用各自v1/显式软源v2目标合同，三任务
全部三类程序的原CEM后GN4完整动作，不更换索引。144条真实15步分支、432包、
1728个主机CUDA predict样本，10项完整性门全部通过。所有包的完整192维真实
表示、qpos、RGB和动作SHA与历史逐位一致；手写递推和发布模型原生rollout的
全部包也逐位一致。首包free/actual-history预测一致，完整终态预测精确重现。
参数、buffers、eval状态、原文件与实现封存均检查通过。

每包另从该分支当时真实已观察的历史计算单步预测。这个诊断能定位多步误差，
但原始根上的selector不可能同时拥有多个未执行分支的这些观察；因此它不作为
可部署的反事实排序修正。202651004全驱动、native索引11与greedy索引10的
native减greedy成本：

| 包边界 | 根出发自由递推 | 该分支真实历史单步预测 | 真实RGB |
|---|---:|---:|---:|
|1|+0.550107|+0.550107|+0.588530|
|2|-0.829120|-0.863751|-0.852832|
|3|-0.000671|-0.000350|+0.002648|

最后一包即使用真实历史仍反转。因此“只修多步rollout接线”不足以解释/恢复本根，
但不据此排除所有其他代码/目标/模型协议原因，更不是理论否定。

## 完整坐标的精确归因

独立NumPy按root identity配对全部任务、全部包，同时保留native-vs-greedy和
native-vs-zero。令预测差为dp、真实差为do、参照预测/真实为pz/oz、目标g，则
真实margin减预测margin精确等于：

`2 mean((oz-pz)*dp) + 2 mean((oz-g)*(do-dp)) + mean(do²-dp²)`。

全部恒等式残差最大1.083e-15。202651004终态自由递推总margin误差0.003319331，
其中公共参照偏移0.002120239、动作响应线性误差0.001110405、二次误差0.000088688。
用真实未来greedy作oracle公共锚点可把预测margin改为+0.001449266，但这个锚点
在部署时不可用，绝不能把此诊断实现成读取真实未来的修复。

同批真正不同native/greedy只有全驱动202651004、仅腕202650000/001；其余根
两个程序相同，零margin保留在图中，不把它们算作额外独立排序成功。仅腕两根
终态排序本轮无反转，不等于首段非贪心条件已恢复。

## 因果反馈执行：已完成

权威回执：`results/reacher_d2p_masked_feedback16_20261026_cuda.json`。

所有原计划、原预测waypoint在执行前先统一封存。首包不改变，第二/三包分别
只读取自身分支已经执行的真实RGB/动作历史，使用固定四步GN与原动作incumbent，
damping=0.01、原box约束和线搜索，比较：原动作重放、面向最终目标的一包重规划、
面向原想象waypoint的一包跟踪。没有未来标签、权重训练、跨分支观察或挑根。
每次选择在执行前写入不可变回执，记录初始计划、父执行哈希和观察frontier。

共432条真实15步分支、1296包、19728个主机CUDA predict样本，12项完整性门
全部通过。全部144条原动作重放及所有模式首包的完整真实观察精确一致，全部
replay单步预测也与上一审计逐位一致。独立分析逐项验证1296组选择/执行链、
动作SHA、目标、完整192维成本、四步下降receipt与原参考。

| 任务 | 原动作native均距 | 目标重规划 | 原路标跟踪 | 路标跟踪根级胜/负 |
|---|---:|---:|---:|---:|
|全驱动|0.127644|0.170523|0.113647|12/4|
|仅肩|0.587835|0.765244|0.591332|4/12|
|仅腕|0.675340|0.701611|0.674113|11/5|

指标为全部16已知根的真实终态关节距离，按周期肩/有限位腕的拓扑计算。全驱动
路标跟踪相对原重放均差-0.013997，描述性配对bootstrap95%区间
[-0.024235,-0.005307]；肩/腕两任务区间均跨零。全驱动RGB成本均差的区间也
跨零；不把关节均距改善偷换为RGB排序或正式成功率。直接目标重规划虽然全驱动
12根改善，四根恶化使总体变差，不能只报胜根数或将此路径默认部署。

### 原终态反转根的实际修正

全驱动202651004：

| 执行规则 | native真实均距 | greedy真实均距 | native RGB成本 | greedy RGB成本 |
|---|---:|---:|---:|---:|
|原重放|0.027885717|0.013872700|0.003654499|0.001006140|
|同样目标重规划|0.012497715|0.027893350|0.001050549|0.003726097|
|同样原路标跟踪|0.020818979|0.029828828|0.002902807|0.004306852|

这证明通过执行时自身观察可以改变该根的物理比较结果。相同反馈也作用于greedy，
没有以强反馈native对比无反馈greedy。但两者中greedy也变差，且全驱动只有这
一个不同前缀，不能据这一个差异根宣称稳定重配置优势。仅腕202650000首段
非贪心失败仍在（首包未动）；目标重规划还让该根原本有利的终态比较恶化。
本批是已知根运行时诊断，计算量不等，不是独立校准、同动作排序恢复或C2机制证明。

## 复核时发现的指标复用错误

新增反馈入口最初复用了旧p0的`_qpos_distance`，会将两个关节都wrap；腕有限位，
这个通用helper不应作为新合同的最终指标。独立分析从保存的完整qpos和原目标
metadata重新计算，保留旧双wrap值，新增明确拓扑的纯数值helper，并修正后续
入口。全部432个实际终态的新旧值最大差3.33e-16，因此这是真实的潜在指标
错误，但**不是本批反转或反馈收益的原因**。纯数值对抗测试以腕+2.7到-2.7
的5.4rad距离拒绝错误环绕，不以这种fixture当作CUDA效果证据。

原CUDA报告和其归档实现保持不动；新源码的纯指标改动不回写该报告。神经预测、
规划动作和真实执行均未因此改变，也没有对432条完整轨迹重复计数。

## 产物和复现

逐包归因的最终双图包：
`results/reacher_d2p_masked_packet_fidelity_figures16_v3_20261026/`。
反馈的最终双图包：`results/reacher_d2p_masked_feedback_figures16_v2_20261026/`。
均含PNG/PDF、summary、notes、LaTeX和SHA manifest；首版保留，v2/v3只完善
独立拓扑指标和小margin图中文字，不修改物理证据。图按figure-generation技能
全根展示并目视，未只展示成功根。

工作目录`/home/gbk/code/LeWorldModel/le-wm-main`。已存在路径不可覆盖，实际重跑
须改新输出路径；神经入口仅主机CUDA：

```bash
.venv/bin/python -u -m experiments.spectral_reachability.audit_reacher_d2p_masked_packet_fidelity \
  --reference results/reacher_d2p_masked_suffix_search8_20261026_cuda.json \
  --reference results/reacher_d2p_masked_suffix_confirmation_fresh8_20261026_cuda.json \
  --policy /home/gbk/.stable_worldmodel/checkpoints/reacher/lewm \
  --adapter-artifact results/reacher_d2p_p1_distribution_repair_v1_20261024_cuda/reacher_training_action_stats.json \
  --output results/reacher_d2p_masked_packet_fidelity16_20261026_cuda.json

.venv/bin/python -u -m experiments.spectral_reachability.audit_reacher_d2p_masked_feedback \
  --reference results/reacher_d2p_masked_packet_fidelity16_20261026_cuda.json \
  --policy /home/gbk/.stable_worldmodel/checkpoints/reacher/lewm \
  --adapter-artifact results/reacher_d2p_p1_distribution_repair_v1_20261024_cuda/reacher_training_action_stats.json \
  --output results/reacher_d2p_masked_feedback16_20261026_cuda.json

.venv/bin/python -m experiments.spectral_reachability.analyze_reacher_d2p_masked_packet_fidelity \
  --report results/reacher_d2p_masked_packet_fidelity16_20261026_cuda.json \
  --output-dir results/reacher_d2p_masked_packet_fidelity_figures16_v3_20261026
.venv/bin/python -m experiments.spectral_reachability.analyze_reacher_d2p_masked_feedback \
  --report results/reacher_d2p_masked_feedback16_20261026_cuda.json \
  --output-dir results/reacher_d2p_masked_feedback_figures16_v2_20261026
```

新增59项逐包/完整面板/因果frontier/数值拓扑合同测试通过，相关广覆盖1131项
通过（41.68s）。这些回归和独立后处理不执行CPU神经实验。整套研究尚未闭环：
仍缺独立P1校准/outcome、真实能力改变的C2、几何调度C3及完整资源匹配强对照。
