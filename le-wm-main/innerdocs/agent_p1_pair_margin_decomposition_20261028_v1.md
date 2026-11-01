# T3/T4：固定 40 槽、80 对的 saved-only margin 分解

源文件：`experiments/spectral_reachability/agent_p1_pair_margin_decomposition_20261028_v1.py`。
测试：`tests/test_agent_p1_pair_margin_decomposition_20261028_v1.py`。本轮独占新增这三件套；冻结理论文档 SHA `94ef31f06f49e11195ede31cec8fe9dfe2ce807c0b37bf02c9d5de1836222246` 保持不变。

交付性质：source-first，供 MAIN 审读后执行；本 worker 不运行生产 saved CLI，不生成生产结果。仅用小型合成数组/保存文件运行测试，无 neural、CUDA、物理、训练、Q 拟合、选根或新策略选择；不修改任何原 producer、结果、门和阈值。这里的结果是描述性归因，不是 P1/机制/freshness/任务效益放行。

## 1. 固定 authority，不信单个 true 标志

唯一生产输入入口固定为 MAIN 的
`results/agent_p1_model_library_selector_evaluation_20261028_v1_MAIN01.json`，SHA
`d8b3bedd38e9f9a56d0156f2452d5a20f363acbdad42ea5e3ee4d91f9626ff67`。

先读固定 SHA，再调用其原 `evaluation.build()` 完整 saved-only 重建，按 canonical JSON bytes 比较全部内容，**包括 signed zero**，而非 Python dict 的 `0.0 == -0.0` 等价。原严格链重建模型 selector、panel/native physical consumer、original reports/receipts、prepare/model/source/artifacts/arrays/journal；不是只核报告文件 hash 或相信 `P1_admission=false`。不调用任何 run/execute/model/physics 路径。

关键原 strict authorities：

- model report `d5ab99d55bacd73c8d9cf0fb774819bb46af63100053198858e639d8fa746738`；model strict receipt `d82109eb8de582e59a4f0485307c01c9b91f9f84551c218c61cca4da24335213`。
- actual RGB39 report `cb08d580b3ee2e8d15d2cc83065a8dc68a74672df0e2b602479a836469cb15f3`；strict receipt `9dfb1081711505cef281f1ff547b1dad46b59cb0ea207e985497ecbf9cecf0a9`。
- panel NASH02 `7eda1333a3bf2c8ef1d46deaf67d74c6d464498c018b52cf35e9802204501e5d`。

本 consumer 继承并额外核验每槽原 plan/prediction/job identity、完整 context digest、prefix/suffix/full10 raw float32 signed bits、goal192/identity operator、actual root10/start15、同一个 terminal20 返回事件。旧 native full3 与 shifted context 不误比：只提取 shifted `[3,192]` 的**最后一行**作为预测终态，实际是同 condition 的 terminal20 RGB 编码192。

独立数组 loader 核 compressed SHA、gzip 解压上限、NPY SHA、禁止 pickle、无 trailing data、shape/dtype、finite 和 array SHA。缺失12保持无 actual array/null，不拿 incident/off/旧帧或预测补图。index12 的 native return20 和物理指标仍有合法 authority，不能因为 latent unknown 抹掉物理比较。

输出只保存 authority 路径/SHA、完整旧 closure 的 count/digest 和明确的 full-rebuilt 标志；不复制约49k bindings、不复制原完整报告/巨型重复字符串。每个槽保存紧凑身份、数组 receipts、物理 scalar axes 和数学 scalar terms；每对仅引用 i/j/anchor。默认新报告 cap 4MiB，超出则明确拒绝，禁止截断。verify/consume 始终重新走完整原 authority+数组+全部数学重建，不能由 hash-only 开门。

## 2. 坐标、归一化、符号与精确恒等式

原模型评分：`mean((I192 @ (p-g))**2)`，以 float64 对原 float32 endpoint/goal 计算。因此原始 squared norm **除以192一次**。若使用理论写法 `||A(p-g)||²`，对应 `A=I/sqrt(192)`、有效 co-metric `I/192`；原 producer 把算子命名 Q=I，并不意味着要再乘一次 co-metric 或再除192。实现直接算 raw sums/192，不数值构造 sqrt(192)，不改变 Q/动作/选择。

设 `rhat_i=p_i-g`、`e_i=a_i-p_i`，lower-is-better，所有 pair 固定 `i<j`，margin 为 **cost_i-cost_j**（不是另一文献常用的 j-i）。

`Qa_i-Qp_i = 2mean(rhat_i*e_i)+mean(e_i²)`。

pair 精确 correction：

`(Qa_i-Qa_j)-(Qp_i-Qp_j) = cross_i-cross_j+quadratic_i-quadratic_j`。

另存共同偏置/相对响应分解，令 `s=rhat_i-rhat_j,c=e_j,delta=e_i-e_j`：

`correction = 2mean(s*c) + 2mean((rhat_i+c)*delta) + mean(delta²)`。

两条恒等式均在原 float32 值对应的**精确有理数**上逐项计算（`Fraction.from_float` 精确承接 binary32），所有 exact scalar 输出 numerator/denominator 字符串。每槽与每对 exact identity residual 必须为0；不设容差。不是从旧交叉项字段改符号复制，而是重新读数组计算。共同偏置投影与响应误差可同时存在，不能只凭误差项大小宣称唯一物理原因。

同时重算原 NumPy float64 MSE，并与 authority 的 predicted_Q/actual_Q 逐 binary64 bits 比较。浮点展开残差、float64-vs-exact-rounded 差和原 scalar margin 的 sign 是否与有理数一致单独报告；这些是舍入诊断，**不作为可调容差、不 snap sign、不把近零并入 tie**，不 relaxation 原 gate。T3/T4 主分类沿原 float64 评分差的严格符号；即便发现 exact/float64 符号不同也保留原分类并显式报告，不悄悄替换研究合同。

## 3. 全分母与 T4

固定8个 original anchor，各5词的全部10无序 pair，共80；只在同 root/condition/role/task/anchor/context/prefix/goal/time 内比较。后缀动作在 pair 两臂之间本来就不同；“同动作”指每一臂预测与其 observed/native outcome 精确绑定同一原 raw action，不是要求 i/j 后缀相同。无 cross-role、cross-condition、跨时间或跨根拼接。

39 actual slots 提供76可比分解；另4对 `(10,12),(11,12),(12,13),(12,14)` 原位保留，`exact=null,float64=null`，observed-to-physical unknown。model-to-physical 仍可用40条实际返回标签。没有按反转/成功/词/根挑选，原12/76不会被删除或事先硬编码成验收目标。

T3 分列 predicted→observed 编码排序；T4 对每个原物理轴独立分列 predicted→physical 与 observed→physical。继承方向：success/coverage 越高越好，其余距离越低越好。native_distance、allXY、agent/objectXY、angle、velocity、不具 authority 的 coverage 全部分开；coverage仍null，success平局不算改善，不合成分数或拟合权重。全80保留 concordant/discordant/both-tie/单边tie/unknown 分类与计数，禁止将76对、8 context或40槽当独立根；full-library reversal rate 保持null。

因输入完整性/代数绑定失效而拒绝不是理论负结果；精确分解通过也不证明模型足够、真实能力改变或强基线优势。fresh roots=0；理论与本 consumer 都不能签发 freshness。

## 4. MAIN 审读后的可执行命令

工作目录 `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p1_pair_margin_decomposition_20261028_v1.py

.venv/bin/python -B -m experiments.spectral_reachability.agent_p1_pair_margin_decomposition_20261028_v1 analyze --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p1_pair_margin_decomposition_20261028_v1_MAIN01.json

.venv/bin/python -B -m experiments.spectral_reachability.agent_p1_pair_margin_decomposition_20261028_v1 verify --analysis /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p1_pair_margin_decomposition_20261028_v1_MAIN01.json --analysis-sha <analyze返回的实际SHA>

.venv/bin/python -B -m experiments.spectral_reachability.agent_p1_pair_margin_decomposition_20261028_v1 consume --analysis /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p1_pair_margin_decomposition_20261028_v1_MAIN01.json --analysis-sha <analyze返回的实际SHA>
```

analyze 使用 exclusive `xb`，只允许results下自有命名前缀，不覆盖旧文件；verify/consume不写报告、stdout只输出简短回执，不打印巨型原closure。调用原 saved validator 可能比本数学循环昂贵，必须计入实际运行耗时，但不产生新的模型/物理调用。未运行前不声明生产接线通过、不提供虚构结果SHA或重现反转数。

## 5. 本轮验证与未决

先创建测试并观察缺实现的失败，再实现。RAM/合成保存文件覆盖：192归一化、精确pair恒等式、共同偏置/相对响应分解、4unknown与全80、各物理轴、signed-zero raw/goal/context/state、不合法Q/shape/dtype、单ULP原标量漂移、完整compressed/NPY/array hash、无trailing bytes、完整原布局的合成build、原authority必须完整重建、报告重签名篡改和独占写入。fixture原先把独立JSON数据共享为同一可变列表，已隔离，不能用这种共享别名让变异同时污染比较双方。

未决：MAIN完整源码审读与生产saved analyze/verify/consume。测试只验证本消费者的逻辑与模拟接缝，不替代真实原closure检查。当前不启动新实验，不触碰MAIN的Hi-LeWM/P3任务，不修改冻结理论文档。
