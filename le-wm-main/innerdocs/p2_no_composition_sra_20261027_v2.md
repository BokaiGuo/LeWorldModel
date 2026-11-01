# v2 联合 H2 无组合 SRA：新授权固定算法

保留 v1 拒绝与全部旧源码/封存。本版依据主线程在用户授权推进范围内预声明的新协议实现，不能因
“旧合同没有”重复拒绝。只新增 v2 module/test/本文和独立 results；不训练、不下载、
不按新 outcome 选根/选阈值。experiment-code 技能用于固定配置、日志和测试，
本轮只做 prepare，不运行 CPU/CUDA 神经 forward。参数并非用户逐项亲选；
归因为 main-thread predeclared under authorized implementation。
项目/祖先检查没有 AGENTS.md。

## 固定算法

原四个已用根完整三帧 latent / 两包 between-state action history、原 goal0、Q、
normalizer、checkpoint 和 root/goal RGB SHA 均沿用。联合20维两packet logits零点
计算 H2 nominal 和一个192×20终点 Jacobian；第二步通过第一步完整图反传，
没有 detach、prefix 分支、suffix Jacobian 或 intermediate reach-value 评分。

全20维 `Q@J` 与 `Q@(goal-nominal)` 做现有精确有界 ridge，damping=.002，
一个联合 logit L2 球 cap=sqrt(10)*.75。候选顺序固定为 zero；DLS×.25/.5/1；
mode0..8，每个按 radius=sqrt(10)*.25 / cap、sign=-/+，共1+3+36=40。
SVD 使用 NumPy LAPACK 返回顺序，奇异向量最大绝对坐标的首个位置取正；
退化/零奇异值不重抽、不调秩，保留并保存返回基。不同平台的退化子空间基不保证
逐位相同，需用保存的 bank/logits/commands/runtime复核，而非冒称跨机器确定性。

raw=float32(tanh(logits))，time-major [2,5,2]→十步命令，仍在原[-1,1]箱内。
本版是一个联合球，不是每packet各一个球；不声称与 full20 box 或旧bank的
搜索分布/支持完全等价。全40候选分别非线性评估完整H2，按原Q终点mean-square
stable-first选择，不复用 nominal 免费充当zero候选，也不额外评估DLS/均值。
赢家按原baseline做分段H2及独立完整H2双replay，共4predict；只生成一条十步edge。

## 资源账本

每根 nominal=2、Jacobian forward=2、40候选=80、winner replay=4，共88predict；
四根352，不是84/root或336总数。实际 action_encoder 入口也应为88/root；
固定三帧上下文共264 processed context-token samples/root、1056总数，
predict/encoder分别留入口元数据，不能把两种入口重复相加当context预测成本。

每根Jacobian调用1、SVD2（ridge1+mode1）、bounded solve1、selection evaluation40；
四根Jacobian4/SVD8/数学标量输出VJP方向768。Jacobian采用vectorize=True，
192方向不是192次autograd API backward；内部kernel/engine backward次数未知。
操作journal记录请求、完成、失败和SVD/solve时间；实际predict入口由既有
actual_entries与PredictionBudget计数，统一sample_limit=88覆盖编译/候选/replay。
既有guard仅接受planning/evaluation：编译用独立4次planning scope，实际入口的
role metadata仍明确为compilation，之后是80次planning和4次evaluation，不改旧guard。
失败不退款、不补候选；任何失败不可部署winner，全部根及失败记录保留。
FLOPs、上游RGB encoding/bank构建成本和共同wall cap仍未知；逐根含同步wall time、
全局peak CUDA memory。本版不是严格总算力配对；P1/P2 admission全部false。

## Prepare、复验、主线程未来运行

cwd `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p2_no_composition_sra_20261027_v2.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p2_no_composition_sra_20261027_v2 prepare --output results/p2_no_composition_sra_20261027_v2_prepared01
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p2_no_composition_sra_20261027_v2 verify --output results/p2_no_composition_sra_20261027_v2_prepared01
```

prepare/verify成功退出0只表示执行资产/来源核验，不是科学准入。新算法及新来源在
本版新outcome之前绑定，四根已经用过，绝不冒称独立prospective cohort。
输出必须全新；prepare独占protocol/prepared/source_seal目录，不产神经report。
source seal包含本版、v1只读依赖、baseline recurrence/normalizer/model实现、
instrumentation和现有config/checkpoint哈希绑定；任一输入/当前源码/镜像改变则拒绝。
源码哈希只证明字节绑定，不证明执行真实性。

仅供主线程review后另行运行，本轮未执行：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p2_no_composition_sra_20261027_v2 cuda --output results/p2_no_composition_sra_20261027_v2_prepared01
```

CUDA分配前verify，独占started防重跑；CUDA缺失拒绝，无CPU神经fallback。
每根独占journal、bank与结果，report记录实际入口/编译/失败账本；不包含物理执行器，
旧physical consumer schema不能直接当本版输入。主线程完成此校准后再设计严格
总算力配对，不能把本版预期预算、纯数组测试或prepare称为真实CUDA/物理结果。

只读 `selected_actions(output)` 可在未来真实完整report存在时重新校验并导出四条
动作；返回actions、report SHA以及额外verification_work（SVD8/solve4/neural0），
这项额外数组验证不是原352-call校准预算的一部分。该接口本轮没有真实report可消费，
不执行物理，也不将已存receipt/hash当独立神经执行真实性证明。

本轮验证：新增20项纯数组/metadata/保存合同测试通过；连同v1拒绝、MPPI、full20
sampling、composition admission和严格P2合同联合168 passed（26.07秒）。
v1 prepared01来源复验仍按原语义退出2。无CUDA/CPU神经forward或物理实验。

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p2_no_composition_sra_20261027_v2.py tests/test_p2_no_composition_sra_20261027_v1.py tests/test_p2_full20_mppi_20261027_v1.py tests/test_p2_full20_sampling_20261027_v2.py tests/test_p2_composition_admission_20261027_v1.py tests/test_d2p_p2_comparison_contract.py
```
