# D2P：机制评价合同与模型条件任务准入

## 范围不变

原始提案要求冻结模型生成带动作见证的程序、非贪心重配置带来真实能力和任务收益，
并在P1成立后再做组合与动态计算。当前C1--C3、实验1--3和强基线要求均保留。
本轮不更新模型、readout或adapter，不修改历史阈值或P1准入，不筛选真实受益根。

## 已完成：同度量、不同残差合同的物理复算

编译器的proposal-native规则使用每个后继自己的残差与有界QP目标值；历史物理
机制诊断则使用共同zero-prefix未来残差下的关节gain fraction，并要求原RGB
首段代价与RGB/关节终态同时满足条件。共同残差隔离响应方向，但并不等价于
原提案的own-state可达目标。单独只读代码复核也确认它是显式更强的另一诊断，
不能把其否定等同于原科学命题被否定。这里不是把它改成较易通过的门。

新增独立分析保留全部八根、三任务与封存native/common/greedy/zero角色，在
原RGB、固定目标Q-RGB、关节三个空间分别复算两尺度、common/own两种残差。
共696个新有界拟合，另验证原696个拟合与全部原数据/源码/动作回放合同。
各空间分别比较首段、终态、own objective、common gain以及stage+own objective。

产物：`le-wm-main/results/reacher_d2p_mechanism_metric_contract8_v1_20261026.json`。
实际命令（在`le-wm-main`目录，仅保存数组的NumPy/SciPy计算）：

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.spectral_reachability.audit_reacher_d2p_mechanism_metric_contract \
  --capability results/reacher_d2p_full_goal_metric_capability8_v1_20261026_cuda.json \
  --output results/reacher_d2p_mechanism_metric_contract8_v1_20261026.json
```

结果没有恢复native规则的机制见证。common规则在仅肩/仅腕的Q-RGB空间中，
模型见证分别2/6个，真实首段回归和终态改善分别2/6个，但双尺度own objective
胜两个对照均0；关节空间仅腕有2个“首段/终态/own objective”合取，却没有任何
stage+own objective也胜两个对照的根。因此度量/残差合同差异真实存在，
但不能称其已解释或修复这批负结果，也不回填P1。

另明确区分零动作coasting后继与真实初始状态。当前保存数据可以精确计算初始
关节代价；初始RGB模型上下文向量没有保存，此项保持unknown，而非以zero-prefix
冒充。所有旧zero-relative条件保留，不能把相对于零动作更差直接写成相对初始
状态退步。该语义差异也不足以使本批P1成立。

## 尚缺的原提案前提：模型条件任务准入

原提案P1指定“当前直接梯度容易停滞、但两阶段动作可达”的状态。此前固定来源
八根并未按这个条件准入；它们是合法诊断来源，却不能保证覆盖这个机制任务族。
不能把普通状态没有非贪心选择自动解释为理论失败。

本轮首先建立模型预测条件的有限预算gradient-plateau/escape队列，不执行真实未来：

- 固定从seed202656000开始的128个候选seed扫描，按episode元数据顺序取前32个
  与全部248个既有source episode不重叠、彼此不重叠的历史。所有元数据碰撞留档；
  神经或渲染失败后不替换已经准入的根。
- 官方冻结checkpoint、原目标偏移[0.65,-0.65]、原动作限制、固定目标Q均保持；
  每根都保留全驱动、仅肩、仅腕，Q使用原完整192输入Jacobian与固定FD合同。
- 同一三包动作从零开始GN4，保留动作继续到GN8，再到GN16。每一阶段保持
  原解；随后128样本、3轮、16 elites的真正CEM保留GN16原解。目标和动作盒相同。
- 模型准入条件：至少一个单执行器任务的GN8至GN16成本下降不超过1e-6，且
  CEM比GN16的成本严格降低超过1e-6。只称“固定预算平台与采样逃逸”，不称
  数学平稳性、强梯度规划器失败、真实能力改变或两阶段机制已经成立。
- 对两个单执行器任务对称排序：先取合格任务中最大的`(GN16-CEM)/zero_cost`，
  再取两任务最小值，最后按seed排序。全驱动始终保留为控制，不按物理收益删任务。
- 选前8个合格根；不足8个则fail-closed并输出空准入列表，不用容易根填满。
- 所有32根模型动作、完整加权192维预测、优化回执和GN16标量投影梯度留存。
  准入函数不读取真实未来、任务关节标签或旧实验性能来排序。
- 预期149184个主机CUDA predict样本、416目标编码forward、6144目标输入VJP行、
  295008动作逻辑VJP行，分别核账。初始RGB/source状态恢复不是规划程序的真实未来。

新准入只定义一个明确的model-conditioned任务分布。它不能冒充随机来源的无条件
泛化；后续任何真实校准/outcome仍须单独冻结协议、保留全部被选根和三任务、
补足等资源强基线、真实能力探针与闭环，并通过原阶段准入。全部原失败根与旧报告
仍保留；这不是追认旧P1通过，也不是放开P2/P3/P5。

## 主机CUDA执行命令

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -u -m experiments.spectral_reachability.build_reacher_d2p_model_escape_cohort \
  --policy /home/gbk/.stable_worldmodel/checkpoints/reacher/lewm \
  --adapter-artifact results/reacher_d2p_p1_distribution_repair_v1_20261024_cuda/reacher_training_action_stats.json \
  --disjoint-reference results/reacher_d2p_full_goal_metric_compiler8_v2_20261026_cuda.json \
  --seed 202656000 \
  --output results/reacher_d2p_model_escape_pool32_v1_20261026_cuda.json
```

协议与源码必须在推理前归档，所有神经计算仅主机CUDA。运行或测试完成均不等于
P1通过，必须以最终完整产物与独立逐项复算为准。

## 完成结果与独立复算

主机作业正常退出0；32根、三任务全部完成，15项完整性门全通过。
实际149184个CUDA预测样本、416个目标编码forward、6144个输入VJP行、
295008个动作逻辑VJP行；没有执行或读取真实未来分支。

独立分析重新构造元数据来源扫描、全部96个任务的完整坐标代价、GN4/4/8
连续轨迹、CEM原解保留、动作盒和mask、投影梯度及最终根排序，均通过。

| 任务 | 有限预算平台根 | 采样逃逸根 | 两者同时满足 |
| --- | ---: | ---: | ---: |
| 全驱动 | 1 | 2 | 1 |
| 仅肩 | 7 | 2 | 0 |
| 仅腕 | 4 | 4 | 1 |

至少一个单执行器任务合格的根仅1个，少于预先规定的8个，最终
`selected_seeds=[]`、`insufficient_model_eligible_roots`。不补根、不调门槛，
不把全驱动合格冒充单执行器合格。本批没有建立所需的模型条件任务队列，
更不是原P1的物理机制否定；也不能据此宣布代码已全部正确。

产物：

- `le-wm-main/results/reacher_d2p_model_escape_pool32_v1_20261026_cuda.json`
- `le-wm-main/results/reacher_d2p_model_escape_pool32_analysis_v1_20261026.json`

独立分析实际命令：

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.spectral_reachability.analyze_reacher_d2p_model_escape_cohort \
  --report results/reacher_d2p_model_escape_pool32_v1_20261026_cuda.json \
  --adapter-artifact results/reacher_d2p_p1_distribution_repair_v1_20261024_cuda/reacher_training_action_stats.json \
  --output results/reacher_d2p_model_escape_pool32_analysis_v1_20261026.json
```

新增独立分析器10项测试覆盖正确平台/逃逸和mask、动作界、代价、轨迹、设备、
投影梯度、物理标签、预算、原解守护损坏。与准入/机制合同测试合计34项通过；
完整D2P相关回归1591项通过（49.47秒）。这些是实现证据，不是效能证据。

## 并行环境前置条件审计：没有靠换环境跳过P1

PushT修复后的root-anchored校准已经执行：4/4模型见证、4/4 RGB非贪心，
但仅3/4 pose非贪心，因此物理门未过。完整目标度量/GN诊断也已执行，
192个物理分支、5056个CUDA预测，未准入P1；不能将其写成待跑的新实验。
PointMaze已有SRA P1不等于原D2P P1，后者仍缺独立任务协议、重配置前缀、
真实后继能力和重新编译后缀的正式准入链。

本轮进一步只读检查PushT训练/导出路径，未加载神经权重、未扫描大数据：

- `le-wm-main/config/train/data/pusht.yaml`指定`pusht_expert_train.lance`、
  frameskip=5。现有HDF5不能仅凭文件名假定与训练Lance内容完全一致。
- `le-wm-main/train.py`在数据划分前调用`get_column_normalizer`；
  `le-wm-main/utils.py`从非NaN动作行计算均值与PyTorch默认样本标准差。
  这是当前训练源码事实，不是已证明该公开checkpoint使用这版源码的事实。
- 安装库`stable_worldmodel/wm/utils.py:save_pretrained`保存
  `model.state_dict()`和传入config，没有单独序列化dataset transform。
  官方本地`pusht/lewm/config.json`只有模型结构，action input_dim=10，
  没有训练动作均值/标准差、数据版本或训练运行绑定。
- 公开在线示例常数mean=[-0.0087,0.0068]、std=[0.2019,0.2002]，
  不能由这些资产认证为该权重实际训练统计。环境自身action_scale=100
  是相对动作转像素位移，不意味着传入模型前应再次除100。

结论：本地可见资产尚不能闭合官方权重到训练normalizer的精确溯源。
既有“额外除100”错误已有独立CUDA证据并修复；这不能同时证明公开统计
一定正确或一定导致当前剩余失败。不能用重新拟合常数、根据未来效果选择
normalizer或默认换成HDF5统计来伪造解决。P1与研究总目标继续保持未闭环。
