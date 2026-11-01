# Reacher：反馈后继的真实动作响应与重放身份审计

## 研究问题和预先固定的测量

本页补原提案C2/实验2的直接物理机制证据，不以预测rank增加或终态误差改善替代
真实控制能力改变，也不改变P1的门槛。使用官方原始冻结模型及此前独立八根
202648000..007的全部四类计划、replay/tracking两种执行，共64个真实prefix后继。
这些根的已有结果已经打开，因此本次是全根机制诊断，不是新的独立效能确认。

每个后继使用同一两包/十步未来command坐标；围绕中心command，对全部二十个
坐标使用正负方向及固定幅度0.1/0.25，保留完整192维真实RGB、模型预测与两维
真实qpos端点。每个后继81条未来分支，完整协议为5184条未来执行、10496个CUDA
predict样本。qpos及测量出的物理响应均只用于evaluator，不输入planner。

每根使用固定neutral重放后继的中心command未来目标残差，投影全部分支的响应；
这是共同任务方向的对照，避免把不同剩余目标距离误写为几何收益。报告同时保留
真实RGB与物理qpos的响应矩阵，不把有限幅度差分称为无穷小导数或可控性定理。
有界响应增益固定damping=0.01，在CUDA求解且要求box-QP全局误差证书，独立
NumPy/SciPy后处理复算完整端点、矩阵、奇异值和448个有界增益。正gain差有利于
native，与“距离差越负越好”的旧统计语义明确分开。

## 首次运行中断与隔离检查

首次响应审计在根202648004的CEM tracking分支前缀身份检查停止；之前四根的
完整回执及封存实现位于
`le-wm-main/results/reacher_d2p_feedback_response_geometry8_20261026_cuda.roots/`
及同名`.protocol.json`、`.implementation/`。它没有完整终态报告，不能用局部
结果冒称八根完成，也不把该中断解释为模型或理论负结果。中断时只记录了身份
不符，尚不能从该日志本身判断是物理状态变化还是像素量化差异。

随后单独执行完整重放稳定性隔离：全8根×8分支×2次×4包，共512个边界，
全部真实root/history/action和192维表示与历史逐位一致，两次完整integration/
scene/RGB快照也逐位一致。权威回执：
`results/reacher_d2p_prefix_replay_stability8_20261026_cuda.json`，6项完整性门
通过、`historical_mismatches=[]`、`repeat_differences=[]`。这既不否认首次中断，
也不以偶尔成功的重试替代修复。

修订后的响应审计直接消费这两轮均严格绑定历史的真实RGB和完整物理状态，
每次恢复验证integration/scene精确身份，不从重渲染图像中选择近似替代。
同时在大量探针干预后重新执行原prefix，逐包比对完整物理状态并记录像素差异，
以区别渲染生命周期和物理重放问题。任何物理状态差异仍立即拒绝；没有放宽原
RGB阈值，没有修改安装库、checkpoint、原报告或原P1阶段消费者。

## 完整测量与独立复核已完成

修订版完整执行并exit 0：
`results/reacher_d2p_feedback_response_geometry8_v2_20261026_cuda.json`，12项完整性
门全部通过，5184条真实未来分支、10496个主机CUDA predict样本；官方权重、
全部参数/buffer及eval状态不变。所有64个prefix后继均使用两轮精确重放得到的
真实观察/完整状态，256项额外重建诊断的integration/scene及RGB本次也全部精确，
`rerender_differences_at_exact_physical_state=[]`。**首次失配未重现，其直接原因
仍未确定**；不能事后补写成“已经证明是一像素量化bug”，更不能据它解释历史P1。
改进的是新审计的观察/状态持久化合同，不是宣称找到了历史反转的又一个根因。

NumPy/SciPy分析独立验证全部完整坐标差分矩阵、奇异值和448个有界优化增益；
分析器同时按root identity绑定原动作和第四包真实表示，检查所有真实81未来结果，
不只信任报告的true标志。产物位于
`results/reacher_d2p_feedback_response_geometry_figures_v2_20261026/`，包含两张
全根PNG/PDF、summary、notes、LaTeX图注及SHA manifest。两图已逐图目视。
32项新增数据/快照/frontier/增益合同测试纳入广覆盖，906项通过（25.39s）；这些
是代码/证据合同测试，不是CPU神经实验。

### 动作导数一致不等于真实响应一致

下列误差均为64个后继上逐矩阵相对Frobenius误差的均值；model FD和physical
FD使用相同完整二十维command探针。它们从真实已观察的prefix history出发，
**不是原selector在想象prefix上的几何**，不能用来追认原选择规则已经通过。

| 探针command幅度 | 模型解析响应 vs 模型FD | 模型解析响应 vs 真实RGB FD |
|---|---:|---:|
|0.1|0.001126|0.189516|
|0.25|0.007038|0.136191|

同模型差分随幅度增加而偏离解析导数，符合有限幅度非线性的诊断方向，但不构成
无限精度梯度证明。真实RGB误差在较大幅度反而减少，需区别渲染量化、有限尺度
与学习动力学失配；本实验没有将它们全部归因到一个机制，也没有发现足以支持
“仅修解析梯度索引就解决”的证据。

### 当前全驱动任务尚未建立稳定的控制能力增益

共同任务残差固定来自本根neutral replay的中心command未来，包含真实未来观察，
因此它只能用于evaluator归因，**不是可部署的oracle修正**。增益差为native减
同执行模式greedy，正号有利于native；95%CI仅为已知八根的描述性配对bootstrap。

| 执行模式 | 幅度 | 模型FD RGB增益差 | 真实RGB增益差 [CI] | 真实qpos增益差 |
|---|---:|---:|---:|---:|
|replay|0.1|+0.548344|-2.940349 [-6.537586,+0.147882]|-0.000000342|
|replay|0.25|+0.611896|-5.416002 [-8.783101,-2.223689]|-0.000000384|
|tracking|0.1|-4.562030|-10.933250 [-17.076944,-4.442985]|-0.000001162|
|tracking|0.25|-4.421320|-5.174381 [-18.325939,+8.525990]|-0.000001175|

全部qpos增益差CI均跨零；这些是不同坐标/尺度的增益，不能横向比较RGB和qpos
数值大小。replay在幅度0.25下真实RGB增益为0正/7负/1平，不能只取模型正向
均值声称物理机制成立；tracking也没有稳定改善。完整表另保留对固定neutral及
同模式CEM的全部比较，不挑选有利参照或幅度。

物理qpos响应的平均最小奇异值，在幅度0.1下native/greedy/neutral replay为
0.467045/0.466224/0.467531，幅度0.25为0.467052/0.466231/0.467538。
各方法本已具备双关节短期响应；这些值本身不是可控性定理，但不支持把这批
普通全驱动Reacher自动视为明显的一阶控制瓶颈任务族。

还独立计算了全部81条已执行未来中的最佳实际结果，避免线性surrogate在±1
command箱内外推就被当作真实可达证明。native相对greedy的最佳实测RGB成本差
为replay +0.005736（0胜/7负/1平）、tracking +0.053769（1胜/6负/1平）；对应
最佳实测关节距离差+0.002924/+0.025351，CI均跨零。这是固定探针集的evaluator
上界比较，不是部署planner、正式success指标或完整优化器排名。

### 与上一轮反馈收益的关系

此前native跟踪终态均距0.063315→0.027490、八根全改善依然有效；那是在跟踪
原封存的六包计划。本次在第四包后改为统一的低幅度坐标探针，检验的是后续局部
动作响应，**两者不是同一个未来动作合同**。反馈减少原计划执行漂移，不自动
意味着其prefix比greedy创造更好的真实控制能力；不能把两批指标互相替代。

本轮补齐了直接物理机制测量，结果要求继续修正/验证任务与编译合同，而不是
宣布D2P理论被否定，也不是放宽P1。下一阶段仍须在预声明的controllability-gap
任务族上同时证明非贪心前缀、真实能力变化及闭环收益，并完成独立校准/outcome、
无组合及强规划器等资源对照。原研究整体未闭环，P1/P2/P3/P5不由本审计开放。
见[闭环审计](D2P_RESEARCH_CLOSURE_AUDIT_20261025.md)。

## 完整存档命令

下面三个命令均已成功执行。输出文件/目录已存在，重跑必须指定新输出，禁止
覆盖证据；对同八根重放不能称为新的独立效能确认。神经入口均须主机CUDA；
最后一项只有NumPy/SciPy算术和绘图，没有CPU神经前向。

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
.venv/bin/python -u -m experiments.spectral_reachability.audit_reacher_d2p_prefix_replay_stability \
  --reference results/reacher_d2p_waypoint_confirmation_fresh8_20261026_cuda/tracking.json \
  --policy /home/gbk/.stable_worldmodel/checkpoints/reacher/lewm \
  --adapter-artifact results/reacher_d2p_p1_distribution_repair_v1_20261024_cuda/reacher_training_action_stats.json \
  --output results/reacher_d2p_prefix_replay_stability8_20261026_cuda.json
.venv/bin/python -u -m experiments.spectral_reachability.audit_reacher_d2p_feedback_response_geometry \
  --reference results/reacher_d2p_waypoint_confirmation_fresh8_20261026_cuda/tracking.json \
  --prefix-snapshot-reference results/reacher_d2p_prefix_replay_stability8_20261026_cuda.json \
  --policy /home/gbk/.stable_worldmodel/checkpoints/reacher/lewm \
  --adapter-artifact results/reacher_d2p_p1_distribution_repair_v1_20261024_cuda/reacher_training_action_stats.json \
  --output results/reacher_d2p_feedback_response_geometry8_v2_20261026_cuda.json
.venv/bin/python -m experiments.spectral_reachability.analyze_reacher_d2p_feedback_response_geometry \
  --report results/reacher_d2p_feedback_response_geometry8_v2_20261026_cuda.json \
  --output-dir results/reacher_d2p_feedback_response_geometry_figures_v2_20261026
```
