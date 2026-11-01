# D2P：目标编码器度量的新根验证

## 研究问题和边界

已知202653000–007动作库上，固定目标输入co-metric改变了greedy选择，并减少
部分物理关节终态误差，但不构成C2/P1。新实验只检验同一评分干预在未看过的
源episode上是否保留收益，不把旧库收益、测试通过或完整性门当正式研究闭环。

固定候选为`Q=.01*(J_E J_E^T+.01I)^-1`，`J_E`是原冻结官方编码器在目标RGB
处的完整192维输出对归一化像素的Jacobian。只用目标RGB和模型，不用未来
物理状态、关节标签、readout训练或阈值调优。同一目标在三种动作限制下共用Q。

## 预先冻结的执行合同

- 根为202654000–202654007，排除既有232个源episode；源身份在推理前检查。
- 全驱动、仅肩、仅腕全部保留，每根每任务17个程序，加flat GN及flat CEM+GN。
- 保留原谱程序、原欧氏后缀CEM/GN优化与root-response度量评分，额外评分两种
  固定模式：仅改common gain，以及prefix/terminal/common gain一致使用目标Q。
- 不是后缀重新优化实验；不将它称为“完整Q一致compiler”。
- 每目标192个单独输入VJP与13次编码forward，3个固定随机方向、两个差分步长；
  相对误差不超过0.15、余弦不低于0.98，失败停止，不重抽方向。
- 全部8根的模型预测、原/新选择与目标Gram先写入同一预测封存文件，之后才
  执行456个物理分支。不按模型见证或outcome删根、删任务。
- 所有神经计算主机CUDA；预测样本按实际batch大小记账，不允许CPU回退。
- 独立分析重建所有408个新度量QP、48个模式选择和全部原基线；按根身份对齐。
  同时报新方法自身greedy与原固定greedy对照，防止基线随评分变化掩盖事实。
- 双终态/首段合取不含真实能力增益，不是完整C2，更不打开正式P1/P2/P3/P5。
  八根配对区间仅作描述性证据，不作多重性控制后的效能确认。

## 运行状态与代码修正

首次v1在模型加载后、任何forward/物理未来前，被新增源码绑定检查停止：
误将仓库`jepa.py`当官方checkpoint实际的encoder类。官方配置声明的是
`stable_worldmodel.wm.lewm.LeWM`。这是本轮新增审计路径错误，不是模型负结果。
v1协议/归档/失败说明全部保留：
`results/reacher_d2p_goal_metric_confirmation8_v1_20261026_cuda.failure.json`。

修正为从checkpoint的`_target_`解析类源码，不实例化模型、不做CPU推理；
在推理前封存实际源码，加载后比较实际`encode`来源，另绑定checkpoint配置。
新增回归测试确保来源遵循声明类，而不是仓库同名文件，且解析时不实例化模型。

同一批根以不可覆盖v2输出在主机CUDA完成；403920个predict样本、456物理分支，
1536目标输入VJP、104次目标编码forward、408个额外QP，23项完整性门全部通过。
正式产物前缀：
`results/reacher_d2p_goal_metric_confirmation8_v2_20261026_cuda`。

实际执行命令（在`le-wm-main`目录、主机环境运行）：

```bash
.venv/bin/python -u -m experiments.spectral_reachability.run_reacher_d2p_root_metric_confirmation \
  --policy /home/gbk/.stable_worldmodel/checkpoints/reacher/lewm \
  --adapter-artifact results/reacher_d2p_p1_distribution_repair_v1_20261024_cuda/reacher_training_action_stats.json \
  --disjoint-reference results/reacher_d2p_root_metric_confirmation8_v1_20261026_cuda.json \
  --seed 202654000 --goal-encoder-metric \
  --output results/reacher_d2p_goal_metric_confirmation8_v2_20261026_cuda.json
```

独立分析已完成，重建408个新QP、48个模式选择，并复核原基线与封存物理动作：

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.spectral_reachability.analyze_reacher_d2p_goal_metric_confirmation \
  --report results/reacher_d2p_goal_metric_confirmation8_v2_20261026_cuda.json \
  --output results/reacher_d2p_goal_metric_confirmation8_analysis_v2_20261026.json
```

分析产物：`results/reacher_d2p_goal_metric_confirmation8_analysis_v2_20261026.json`。
相对原固定greedy的关节终距均值差（负数改善）：

|任务|仅改gain|一致Q评分|
|---|---:|---:|
|全驱动|−0.00412070|+0.00248632|
|仅肩|−0.00545927|−0.20227365|
|仅腕|−0.00599442|+0.07572794|

一致Q仅肩描述性配对区间为[−0.63872,+0.04282]，跨零；仅腕保留
202654001的+0.654825损害，不删除根，不选择有利任务重定义方法。
一致Q模型见证数为5/2/2，首段与RGB/关节双终态合取为4/2/1；
这些合取没有真实能力改变证据，不等于C2/P1。分别报告相对新方法自身greedy
和原固定greedy，不能把改变greedy产生的收益全部归给重配置机制。

## 完整Q一致编译器：独立新根已完成

新入口`run_reacher_d2p_goal_metric_compiler`固定根202655000–007，排除240个
既有源episode。两套编译器各保留17程序与2个flat控制，共计划912个物理分支。
固定Q用于谱Jacobian/SVD、阶段代价、后缀CEM/GN、own-state reach与选择规则，
不训练模型，不读取物理未来来选择动作。原始方法逐根同行运行。
这仍是封存动作执行，不包含观察反馈后的重新编译，也尚非等计算预算比较。

v1因适配器将原`flat_cem_control`三返回值误接成两个，在任何物理执行前停止。
修复返回值接线并增加回归测试，原根和参数不变；保留
`results/reacher_d2p_full_goal_metric_compiler8_v1_20261026_cuda.failure.json`。
v2原始会话14993已正常退出，完成805536个预测样本、912个物理分支、
1536输入VJP及104次目标编码forward，18项完整性门全通过。全部八根预测
封存后才开始物理执行；先前空物理目录并不表示作业死亡，没有重复启动。

主机CUDA实际命令（`le-wm-main`目录）：

```bash
.venv/bin/python -u -m experiments.spectral_reachability.run_reacher_d2p_goal_metric_compiler \
  --policy /home/gbk/.stable_worldmodel/checkpoints/reacher/lewm \
  --adapter-artifact results/reacher_d2p_p1_distribution_repair_v1_20261024_cuda/reacher_training_action_stats.json \
  --disjoint-reference results/reacher_d2p_goal_metric_confirmation8_v2_20261026_cuda.json \
  --seed 202655000 \
  --output results/reacher_d2p_full_goal_metric_compiler8_v2_20261026_cuda.json
```

独立审计已执行完成，检查加权谱动作见证、1632个独立QP、两种选择规则、
根身份与912条封存动作执行。全部项目内数值辅助源码另行归档并绑定SHA。
回归测试与真实数据审计分别核账，不相互替代。

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.spectral_reachability.analyze_reacher_d2p_goal_metric_compiler \
  --report results/reacher_d2p_full_goal_metric_compiler8_v2_20261026_cuda.json \
  --output results/reacher_d2p_full_goal_metric_compiler8_analysis_v2_20261026.json
```

|任务|模型见证|选中自身greedy|关节终距均差对原固定greedy|描述性95%区间|
|---|---:|---:|---:|---|
|全驱动|0/8|8/8|+0.003010|[−0.006567,+0.013972]|
|仅肩|0/8|8/8|+0.001808|[−0.000529,+0.003973]|
|仅腕|0/8|7/8|+0.016933|[+0.004951,+0.029363]|

负数才表示改善。本候选未修复P1；不能以它否定一般理论或排尽代码问题。

全408程序逐项归因另在
`results/reacher_d2p_full_goal_metric_compiler8_coverage_v1_20261026.json`：

- 每任务136程序，真实关节首段退步且终态胜自身greedy分别25/44/45个；
  再要求原RGB首段/终态同方向，则16/29/21个。
- 模型own-state reach胜zero/greedy仅0/2/1个；总分胜控制0/0/1个，
  全部模型条件同时成立均0个。不能只归因为“库里没有动作”。
- 同Q的模型/真实首段回归谓词分歧24/20/34次，终态改善谓词分歧64/14/19次；
  真实Q与关节亦有分歧，故预测误差与任务代理问题必须分开。
- 这些是固定1e-6谓词的分歧计数，不是完整排名反转率、真实能力证明或因果bug定论；
  不按这些物理候选选择新部署程序、不改参数、不打开P1。

实际归因命令：

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.spectral_reachability.audit_reacher_d2p_goal_metric_compiler_coverage \
  --analysis results/reacher_d2p_full_goal_metric_compiler8_analysis_v2_20261026.json \
  --output results/reacher_d2p_full_goal_metric_compiler8_coverage_v1_20261026.json
```

## 完整编译器的真实能力探针：已完成

新增显式`--full-goal-compiler`源合同，直接读原嵌套方法与原物理分支，不伪造或
改写成旧报告。全部八根三任务，角色仅为预测封存的native/common/greedy/zero，
仅按相同prefix index去重；两个固定扰动尺度0.1/0.25，要求原首段观察、
预测zero后缀和完整响应矩阵精确复现。所有神经计算仍在主机CUDA，不作CPU回退。
这不是新的效能校准，不按上述事后物理候选挑角色或更改选择。

主报告`results/reacher_d2p_full_goal_metric_capability8_v1_20261026_cuda.json`
已完成58单元、3116物理未来、6522主机CUDA预测样本，12项完整性门全通过。
分析`results/reacher_d2p_full_goal_metric_capability8_analysis_v1_20261026.json`
完成696原空间与348固定Q拟合，共1044个独立拟合。native/common分别报告，
不把两种选择规则或不同根的要件拼起来。

|任务|common模型见证|见证且固定Q真实RGB能力双尺度改善|见证且原物理关节C2诊断|全部要件|
|---|---:|---:|---:|---:|
|全驱动|1|0|0|0|
|仅肩|2|2|0|0|
|仅腕|6|5|0|0|

每任务分母8，后三列限common模型见证；native仍均0个模型见证。仅肩两个
common见证虽终态关节/RGB均改善，却都不满足原真实首段退步和真实关节共同增益；
仅腕六见证中真实首段退步2个、关节共同增益2个，但未在同一根同时成立。
不能改成只要求某一条，不能回填P1。

全部封存角色的模型AD/模型FD相对误差中位数在0.0009–0.0067之间；
实际历史模型/真实RGB响应余弦中位数在0.981–0.995之间（两固定尺度分开计算）。
这些局部证据未发现明显AD实现错误，不等于全程非线性预测或任务代理已经正确，
也不能宣称已经排尽代码问题。完整Q一致化不是本问题的充分修复。
相关广覆盖回归1557项通过（49.79秒），仅是数值/逻辑合同证据。

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -u -m experiments.spectral_reachability.audit_reacher_d2p_prefix_capability \
  --full-goal-compiler --render-policy fixed-warm-render-v1 \
  --reference results/reacher_d2p_full_goal_metric_compiler8_v2_20261026_cuda.json \
  --policy /home/gbk/.stable_worldmodel/checkpoints/reacher/lewm \
  --adapter-artifact results/reacher_d2p_p1_distribution_repair_v1_20261024_cuda/reacher_training_action_stats.json \
  --output results/reacher_d2p_full_goal_metric_capability8_v1_20261026_cuda.json
```

实际独立复算命令（仅保存数组上的NumPy/SciPy，不作CPU神经推理）：

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.spectral_reachability.analyze_reacher_d2p_metric_capability \
  --report results/reacher_d2p_full_goal_metric_capability8_v1_20261026_cuda.json \
  --output results/reacher_d2p_full_goal_metric_capability8_analysis_v1_20261026.json
```

## 对原方案的影响

即使本轮控制收益复现，也仍须把Q接入谱方向生成、动作/后缀优化和own-state
reach目标，再用独立物理能力测量检验C2；仅换score不能冒称完成这些模块。
原C1–C3、三个实验、强基线和等预算要求保留，不以这次诊断缩小研究目标。
