# P4 真实／想象反馈：独立接口核账与准入边界

本任务只新增同 basename 模块、测试、本文及全新独占 output；旧 sealed runner、
报告、测试、中央进度、source seal 均只读。使用 experiment-code 技能组织逐轮审计
与非神经回归；适用祖先、experiments、tests、docs 路径未发现 AGENTS.md。
已经阅读中央 `D2P_PARALLEL_CUDA_PROGRESS_20261027.md`、原研究 P4 合同、
P45 前置条件快照及 v1/v2 的协议说明和 v2 执行接口源码。

## 审计对象与不能互换的结论

唯一对象是 `results/p4_feedback_matched_20261027_v2_cuda_main_v1/report.json`
及它的固定八分支、逐步 attempt/observation、decision、candidate journal、RGB /
between / terminal NumPy 文件、三份 runtime/evaluator source seal、绑定的官方
checkpoint/config 和 compilation。权重仅 streaming SHA256，不加载/反序列化。
模块仅 stdlib/NumPy 与 source-seal 验证，不导入 torch、gym 或世界模型。
没有模型、编码、GPU、physics、下载、删除、训练或新 outcome 选择。

旧 gates 只保留为 reported_source_gates，不能代替新验证。分别报告：

1. 内容／来源：protocol 与原文件逐项相同、原输入 SHA、当前及 archive 源码／args
   seal、四个 root SHA × imagined/actual 完整 census、root RGB stack/goal 来源、
   decision 文件 SHA、前轮 decision hash、branch/report、execution/独立 step/attempt
   文件逐项一致。所有读过的文件末尾重新哈希检查，不能静默接受并行变更。
2. 逐轮接口：相同 root、相同 round、相同固定40候选及 seed、原16 packet bank、
   原 goal0/Q；round0 同一真实 context，round1 actual 使用实际编码 context，
   imagined 仅使用本臂上轮 model mid。不能把刚编码但未用于选择的 context 当反馈。
3. RGB／动作／长度：真实三帧 whole-stack 与逐帧 hash；第二轮严格为旧后两帧加
   真实执行终帧；between 为旧末包加刚执行包；float32 time-major 5×2→10，
   显式 mean/std、无额外 /100，数值逐项一致。normalizer 常量另从 sealed Python
   AST 读取而不执行源码。检查 mid/full latent 与 normalized action 的1/2包移位，
   两轮搜索均 H2，真实每轮只执行首包5步；partial packet 禁止成为后轮边界。
4. actual 成本：逐 scope 与 dispatch events 对照，planning 每轮40×2、evaluation
   每轮[2,1,1]，context3、无失败；RGB hook每轮1调用／3帧、action hook按实际
   predict调用核账；逐步 attempt/return 不混淆。回报每根每轮配对向量、耗时、
   显存记录和未测项。root collection/reset/warmup 不计入 selected-path步数。

候选 journal 不允许漏、重排、失败过滤或只留 argmin：完整检查40个路径及成本
有限性，重算稳定(cost,path) argmin、前两名 margin、top5。所选完整终态保存
latent 重新计算原 Q-MSE；未选候选没有保存 terminal latent，因此只能核验它们的
已记录成本和排序，不能宣称本轮重放了全部 nonlinear predictions。

保存 latent 数值、独立 replay 标志与 RGB编码 context 的内容一致性，不等于
重新神经编码或 transition replay；必须明确 `new_model_replay=false` 和
`rgb_reencoding_verified=false`。保存的 physical official/pose 标签由 branch/report
互证，按root连接描述，不冒称本轮重新运行官方 evaluator 或接触力验证。

## 科学准入保持关闭

即使全部接口核验通过，P1/P4 admission 均恒为 false，科学状态仍
blocked_upstream。两轮、四个已反复使用的校准根、固定 bank，不满足完整 P4。
同策不证明反馈无效；不同 context 不保证改策；不同动作或某根改善也不证明机制
成立。必须报告 actual−imagined 的已保存终距及所有不利结果，不筛选根、不做小
样本显著性／等价性推断，不以 predict 数等同 FLOPs／Joules／总算力或 wall cap。

尚缺 PushT 自己的 P1 及采用组件 P2/P3 科学前提、独立冻结起终分布、长程多轮
到任务终止、完整优化／latency匹配强对照、根级配对不确定性，以及可靠任务度量
与接近／接触／推送／闭合分解。其他 PointMaze 分支不能替代这些条件。

下个无需 GPU/physics 的可执行缺口：针对已经保存的80个 state/action/RGB
收据，制定独立、盲化的接近／接触／推送／闭合标注协议和阈值来源；先列明所需
几何／接触证据是否存在，不能从缺失的接触力 telemetry 捏造 contact ground truth。
协议与独立标签资格关闭前，不启动正式 P4 outcome；之后由主线程按上游准入和
独立根长程固定预算合同安排新的实验。本审计不以此建议自动开放任何阶段。

## 运行与输出

工作目录 `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p4_feedback_admission_20261027_v1.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.p4_feedback_admission_20261027_v1 --report results/p4_feedback_matched_20261027_v2_cuda_main_v1/report.json --output results/p4_feedback_admission_20261027_v1_main_v1
```

新 output 必须不存在，basename 必须以本任务名开头；exclusive/fsync 写入
report.json、report.md，运行前封存本模块／测试／本文／seal helper，末尾验证。
退出0仅指独立接口／保存收据审计成功，绝不指 P4 admission；退出2表示完整性
未通过并保存错误。新 artifact 记录精确输入 SHA／独立重算／未验证边界，不覆盖
原CUDA报告。测试均为已保存数据／元数据／算术的破坏性 fixture 副本；没有神经
或物理效能实验。具体真实结果以新输出为准，非预期值不通过改旧 artifact 修复。
