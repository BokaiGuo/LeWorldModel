# MAIN批准协议：goal-conditioned R bank 数值核

本文件先于数值核实现写入。仅本stem source/tests/本文新增，原spectral8、R gate、
normalizer、Q、模型、源任务和封存结果全部不改。2026-10-29 MAIN明确批准以下精确公式，
不是看新结果后调整候选。worker不NN/physics/fullstrict、训练、网络、后代或生产生成。

## 数学与输入合同

仅H1/H3，n=10h；task为原192维native任务，完整3latent/2action-token context。
原task字段名projection在本协议表示**投影factor Q**，定义损失为
`mean((Q @ (z-goal))**2)`；不是声称输入Q为任意度量矩阵的平方根。
本v1只允许原identity192 factor，不扩展非identity权重。

接受同一冻结task、同完整parent context已经付费的raw-zero Jacobian J192×n和nominal z0，
两者必须literal float32。task/goal/Q/normalizer/context/domain、signal binding和来源指针均绑定。
原normalizer已经在raw J的AD图中，不再次除std。不得拿control0 J冒充actual cp5或另一parent J。
输入不接受future预测、physics、success、reach、资格结果、候选分数或效果标签。

float64计算：

```
A = Q @ J
r = Q @ (goal - z0)
b = A.T @ r
U, s, Vt = svd(A, full_matrices=False)
lambda = 0.002
w = solve(diag(s*s + lambda), s * (U.T @ r))
p = Vt.T @ w
d0 = p / norm(p)
t = p - b * ((b.T @ p) / (b.T @ b))
d1 = t / norm(t)
```

这是一次全经济SVD，U192×n/s[n]/Vt[n,n]，保留全部n右奇异方向，包括零奇异值；
无rank截断、top-k、伪逆阈值、预筛、root-DLS别名、候选选择或迭代。
solve是**一次真实np.linalg.solve**，不是把矩阵向量乘虚记solve或另跑额外ridge。
保存A/r/b、全部U/s/Vt、solve输入/输出、p/t、范数、实际b·t及系数，数值环境版本留痕。
不另做SVD符号规范化：两个方向由目标投影决定，正负候选本身都有完整固定槽位。
每个有效候选另存`b_dot_param`及`b_dot_cast32_action`（cast32动作先展平再转float64内积）。
`b·t`只是raw-zero线性关系；有限param经tanh/float32运输后不保证`b·action=0`，
更不保证非贪心或接触变化。这两个量仅解释，不增加筛选、资格或阈值。

退化规则完全固定：b或p为零，两个方向均null；否则d0存在。
`norm(t) <= 64*eps_float64*norm(p)`时d1=null。固定64eps仅避免归一化投影残余噪声，
不是R资格阈值，也不能按数据改它。其余非finite/结构/数值失败保留失败，不换算法或方向。
即使零残差、零J或退化，正常数值路径仍执行一次SVD和一次solve。

半径严格为sqrt(5)*.25、sqrt(5)*.75，H3不乘sqrt(h)。顺序
`radius -> d0/d1 -> sign(-1,+1)`固定8槽，params=sign*radius*direction。
raw严格NumPy tanh(float64)→一次float32 cast→reshape(5h,2)，不第二次scale、clip、
逆atanh、inward projection、epsilon放宽或静默修复。保存实际param norm和相对名义半径舍入差，
不新增一个postcast/logit-norm硬gate误拒原字面乘法；原动作域仍是rawbox[-1,1]。
null方向对应4槽blocked且params/actions/rawbits为null，不补zero、不换词、不去重。
单独给出zero控制定义，但**未执行、未付新edge/后续诊断费用**，不能把继承nominal当完整zero资格。

## 费用、失败与边界

每次正常核构造固定program_compile_calls=1、array_SVD_calls=1、solve_calls=1。
new predict/encoder/projector/J/VJP/backward/physics均0，source signal已付向量另列，未知为null；
不将继承signal和新核费用重复加总为总作业，更不称等FLOPs。实际CPU边界/各操作inclusive时间留存，
嵌套compile含SVD/solve，时间不能直接相加。核计时不含调用前输入文件I/O或最终写盘。

先对三项费用整体reserve；任何cap不足，整个bank不dispatch，8槽保留blocked、actual attempts=0。
dispatch后失败不退款：SVD失败则compile/SVD各1attempt、solve0；solve失败则compile/SVD/solve
各1attempt而SVD已有1return。父compile随后也记failed，不将合法嵌套失败账误拒为额外操作。
RAM ledger是本次数值调用的内存账，不保证进程崩溃时durable；生产持久intent应由未来own runner负责。

本核不求任何新postcontext、不承诺local J在tanh有限步下的误差方向、不保证非线性terminal Q、
postcontext reach、严格R资格或真实任务收益。原bank资格交集为空不等于全空间无解；本bank
也不据此证明存在解。original_R_equivalence、P1/P3/C3/fresh/physical_release始终false。

## 接口与认证边界（实施固定）

`build_bank(task, signal, provenance, *, caps=None, clock=...)`为数值入口；独立新schema。
signal严格使用原native response六字段：binding_sha256、parent_context_sha256、horizon、
nominal、jacobian、producer_sha256。binding必须属于传入task的当前context，不能重贴标签。
provenance明确RAM_FIXTURE或SAVED_REFERENCE、task/signal摘要、原artifact path/SHA与call/array索引、
继承资源向量。只绑定这些外部指针，不在本核读取/认证原artifact或调用原fullstrict；
`source_artifact_authenticated_here=False`及`bank_authenticated=False`不可伪装为HOST/native认证。
未来runner先完成真实来源认证再调用本核，不能把生成JSON本身当封存神经证据。

CLI仅contract、build、verify，读有界显式输入JSON/path/SHA，不发现/猜测真实数据源。
输入schema为本`.input`，严格四项schema/task/signal/provenance；无outcome扩展字段。
verify只重算本次数值payload并独立检查原本地ledger，不嵌套重做上游strict。
本schema的`C3=False`只是本核独立声明，**不是**producer的`full_C3_demonstrated`别名；
不翻译、不重签producer科学声明，也不把缺该producer字段解释为其已认证。

### 原保存ABI的窄兼容检查

只读检查实际`agent_p3_parent_native_cuda_20261029_v1_cuda_MAIN01/report.json`：
四行共66条`rows[*].result.signal_mappings`，H1/H3均存在。
每条采用`parent_task`与`native_response`，已核六字段接口、当前context/binding摘要、
goal/Q/normalizer/raw-domain摘要、identity192及float32 nominal/J/history形状。
此检查不调用fullstrict，不生成实际bank，不认证artifact全部数组/账目/来源。
`outward_response`会为外层root重写binding，不能代替上述native response与local task配对。
完整来源认证仍归调用方；仅给artifact path/SHA或已付资源指针不等于认证完整artifact。

### MAIN命令与测试

工作目录`/home/gbk/code/LeWorldModel/le-wm-main`。先由MAIN将已认证的同一local
task/native signal及真实来源索引写入显式input；本核不猜索引或替调用方补认证。
input的`schema`为`agent_p3_goal_conditioned_R_bank_20261029_v1.input`，其余严格为
`task`、`signal`、`provenance`。provenance含`kind,artifact_path,artifact_sha256,call_id,
origin_index,nominal_index,jacobian_index,task_sha256,signal_sha256,paid_source_resources`；
摘要按本模块`digest`，未知继承资源用null，不用零。
以下大写路径/SHA是MAIN实际参数占位，不是预填未来authority：

```sh
.venv/bin/python -B experiments/spectral_reachability/agent_p3_goal_conditioned_R_bank_20261029_v1.py contract
.venv/bin/python -B experiments/spectral_reachability/agent_p3_goal_conditioned_R_bank_20261029_v1.py build --input ACTUAL_INPUT.json --input-sha ACTUAL_INPUT_SHA256 --output NEW_BANK.json
.venv/bin/python -B experiments/spectral_reachability/agent_p3_goal_conditioned_R_bank_20261029_v1.py verify --report NEW_BANK.json --report-sha ACTUAL_BANK_SHA256
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q tests/test_agent_p3_goal_conditioned_R_bank_20261029_v1.py
```

RAM/临时文件回归34项通过；覆盖公式、退化、实际费用/失败展开、严格输入/账目、
真实exclusive writer与fsync、重算篡改拒绝。输入/输出各16MiB上限，磁盘floor200MiB。
build前后绑定三件套及输入SHA；verify额外重算CPU费用单列，不称零成本。
未实现生产神经runner、postcontext资格或物理handoff；MAIN未批准任何worker生产执行。
冻结SHA在交付消息中列出，避免文档自摘要递归。
