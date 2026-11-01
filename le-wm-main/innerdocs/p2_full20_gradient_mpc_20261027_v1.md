# Internal full20 projected-gradient MPC baseline v1

独立实现/新文件，非官方论文实现。复用 full20 原四个 fixed contexts、192维原
goal_vectors[0]/Q、between-boundary recurrence、raw20 time-major H2 packets、
winner witness kernel、实际 predict/action-encoder instrumentation 与 source sealing。
单根从固定上下文优化一条开环H2轨迹，不是receding-horizon反馈或真实MPC闭环证据。
模型全部 frozen eval；优化的仅20个动作logits，不训练参数或readout。
使用 experiment-code 的协议/日志/测试要求，不进行技能模板中的训练或效果图生成。

## 结果前冻结算法

- optimizer：projected vanilla gradient descent，不是Adam/AdamW。
- logits u0=0，固定39次更新，learning_rate=0.1，无line search、随机重启或early stop。
- seed=20261027，random draws=0；无每root候选重抽样。
- u维度20，joint L2 cap=sqrt(10)*0.75；raw=NumPy float64 tanh(u) 转 float32 后reshape(2,5,2)。
- 候选为u0至u39，共40条完整H2。前39条各一次scalar cost VJP与更新；最后一条仅评分。
- 更新为 project_ball(u - 0.1 * grad_raw * (1-tanh(u)^2))，float64纯数组运算。
- 所有40条保存cost、终点context、logits、raw commands；39条保存raw/logit gradient、
  next_logits、derivative request/completion。按cost稳定first argmin选择所有访问候选中的最优，
  不仅是最终iterate。零候选已包含在40内，不另加baseline或nominal评估。

旧评估配置 config/eval/solver/adam.yaml 指向stable_worldmodel.GradientSolver，
其30步/100samples/AdamW不能直接满足此40candidate预算；本实现不冒称官方复现，
不把该配置改名为budget-matched实现。现有sampling kernel无梯度trajectory接口；
复用的是证据/recurrence/witness/instrumentation，而非虚称重用旧梯度优化器。

## 梯度、预算与数值口径

每candidate仅两次完整模型forward。raw packet是CUDA float32 leaf，两个packet的
latent recurrence不detach，原Q terminal MSE在CUDA float64形成一个scalar。
torch.autograd.grad(scalar,raw)在已有图上一次求20维raw梯度，无新的模型forward；
随后在NumPy float64以解析tanh链式法则更新logits。float32动作量化cast按普通可微cast
的identity梯度处理，不声称量化映射的数学精确导数。无finite differences/Jacobian/SVD。
用于选择的cost仍为保存终点的原NumPy goal0/Q算术；与GPU可微scalar校验
atol=1e-10/rtol=1e-12，这只是实现算术校验，不改变任何P2/科学门槛。

| 每root单位 | 数量 |
| --- | ---: |
| full H2 candidates | 40 |
| candidate predict / action encoder | 80 / 80 |
| segmented2 + independent full2 winner replay | 4 |
| total predict / action encoder | 84 / 84 |
| autograd.grad API calls / scalar-objective VJP directions | 39 / 39 |
| action-logit optimizer steps / model optimizer steps | 39 / 0 |
| predictor processed-context-token samples | 252 |
| extra derivative forward / Jacobian / SVD | 0 / 0 / 0 |

四root完成时为336predict、336action encoder、1008context-token samples、156次
autograd.grad/156数学scalar VJP directions与156动作logit更新。每次VJP穿过完整两步
predictor；方向数不是实测kernel backward次数，39 API calls不等于全部计算量。
记录requested/completed/failed derivative、packet ledger、wall time和peak CUDA memory。
FLOPs、backward引擎内部、上游RGB/建bank和共同wall-time cap仍未测量，不能说等全预算。
只有candidate/predict/replay单位与CEM/random/MPPI一致；SRA额外4compilepredict及其
Jacobian/SVD不能隐去，且两者求导对象和资源完全不同。

为固定有grad/无grad/replay的attention数值路径，明确CUDA SDPA MATH backend，
TF32 off、torch deterministic algorithms on、cuDNN benchmark off、deterministic on，
CUBLAS_WORKSPACE_CONFIG=:4096:8。与旧sampling默认backend不同，不称bitwise跨方法
预测等价或等运行时间。此backend为执行前固定规则；若CUDA runtime不支持确定性
或反传，不fallback、不改算法/门槛、不补候选，报告失败后需要独立新version。
CUDA行为尚待main真实执行验证，CPU数学fixtures不能证明CUDA神经路径可用。

## 支持域、失败与证据边界

20个raw坐标/动作box/H2与sampling一致，但tanh联合logit球只覆盖raw box的真子集；
与SRA相同joint cap，而不是旧packet各自cap。CEM/random在full raw box的proposal
support/分布不相同，不能声称同proposal或只隔离优化器/组合因素。
这是单轨迹局部梯度baseline，不是multi-start方法，也不保证改善或成功。

失败不退款：已有cost与导数/forward记录均保留，其余固定slots显式blocked；不能
重启、填40个替代候选或用部分轨迹winner部署。严格80+4及39次反传/更新且所有
候选完成才有效；search、segmented、full root replay上下文必须精确相同，否则拒绝。
失败root全部保留，不删根。journal每事件flush/fsync；source封存后绝不改字节。

prepare只解析full20已封存protocol/prepared和原compilation，绑定原依赖/source；
不解析sampling report分数，不读physical/admission/comparison outcomes，不调参。
不运行CUDA、不构造env、不执行physics，不读取未来heldout。
P1/P2/composition_admission/physical_efficacy_verified/fair_total_budget_comparison=false。
即使main将来规划成功也只是四已知calibration roots的工程证据。

## 执行命令

实现线程仅运行非神经tests与prepare/verify：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p2_full20_gradient_mpc_20261027_v1.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p2_full20_gradient_mpc_20261027_v1 prepare --output results/p2_full20_gradient_mpc_20261027_v1_prepared01
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p2_full20_gradient_mpc_20261027_v1 verify --output results/p2_full20_gradient_mpc_20261027_v1_prepared01
```

main review后显式CUDA，其他线程不自行运行：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p2_full20_gradient_mpc_20261027_v1 cuda --output results/p2_full20_gradient_mpc_20261027_v1_prepared01
```

prepare output独占、不得存在。cuda在该prepared目录只创建started、四root journals/
rows与report，重复启动会拒绝；失败保留全部文件，不在原目录重试或追认。
只读selected_actions出口在本次source seal前提供，用于未来独立consumer；
重建39步数组更新并验证所有candidate cost/Q/winner/receipt与实际模型预算，不执行神经。
该出口检查保存的gradient轨迹，哈希与CPU算术不能独立证明梯度由真实模型产生。
