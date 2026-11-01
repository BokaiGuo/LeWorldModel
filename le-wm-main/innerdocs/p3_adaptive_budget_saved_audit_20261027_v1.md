# P3 保存轨迹的自适应接线与停止审计 v1

本工具只读封存源码、JSON 和 fsync journal，NumPy 重算，不导入任何旧 runner、torch、模型或环境。仅创建独占小报告（≤2MiB），写入前保留200MiB及完整2MiB余量。不下载、不重跑、不修改旧文件。使用 scientific-critical-thinking 的因果时序与证据比例原则：可执行反馈接线、真实执行轨迹、收益和科学准入是不同结论。

## 三种实现不能混为一谈

1. `p3_joint_horizon_goal_metric_scheduler_20261027_v1.search` 在搜索前调用 v4 `allocate`，来自同根、同Q的 H2/H3 零点 nominal/Jacobian。之后两层 `for h/for k` 完整执行已分配候选，再取稳定 argmin，whole+segmented 验证。不存在候选反馈重分配或成功提前退出。每个6-predict block内 H2有3候选/H3有2候选，所以每臂搜索恒60predict；winner H仅影响2H验证。该实现不是全16^H树枚举，而是**穷尽已分配的固定池**。own-Q ridge确实改变候选1，不可否认真实谱提案生成，也不能把它说成conditional suffix或在线预算控制。
2. `p3_action_jacobian_budget_schedule_20261027_v3.run_search` 每个保留parent确实遍历全部16bank。第一层后用继承的H1 action-Jacobian chi选择K2/K3；根和候选代价决定保留哪个parent，因此是有接线的宽度分配，不是纯伪调度。但K2+K3恒8、H3固定，故16+16K2+16K3=144，加3验证；不能宣称自适应减少计算或自动决定H。继承J获取成本保持unknown，不乘8臂也不当免费。
3. `p3_adaptive_runtime_20261027.run_search` 是不同的真实逐层反馈实现：完整观察本层后计算finite child-minus-parent response，冻结阈值选下一层宽度及是否继续；fixed臂同算signal但忽略选择。不能把joint局限外推为仓库没有自适应代码。真实8条保存轨迹中，唯一signal stop位于depth3=max_depth，故不能节省下一层；两条adaptive轨迹在16+128后已耗144，尾层0attempt是预算耗尽，不是提前信号停机，更不是实际H3探索。

## 重算与账本

固定绑定三个真实report SHA，joint的execution→prepare→protocol及来源/model文件、每根journal均验证。源码库存以AST给出精确函数行号和当前字节SHA。joint独立用保存J/nominal/goal/Q重算chi、冻结分配公式、候选own-Q代价与全部prefix-best轨迹、stable winner，逐根逐臂核对已分配census和search+verification。只在同Q比较uniform/dynamic，sameH和mixedH独立列出，不跨Q相减，不读取物理结果。

neural entry的start/return是一次attempt，不重复收费；相同动作、重试、失败及未return仍逐次保留。predict与action_encoder分别记账；AD forward已含共享predict，不能再加。Jacobian API、192数学VJP方向、signal/ridge SVD另列，backward-engine API/FLOPs未知；source-validation SVD另列，历史metric/planner成本原样只留一份。审计器自己的数组SVD不是新CUDA成本，不混入原执行账。

runtime由每个kernel attempt重建每层工作量，不按层声明推断实际深度；planning/evaluation phases与actionencoder/context-token计数互核。每个完整parent一个signal SVD，另有每臂进入搜索前一次root-vs-root验证SVD，不能遗漏。保存数据缺每个child latent（只有各kernel best context），所以只能独立复算阈值decision、消费时序及工作量，**不能**从原始数组重建全部finite-response chi/gain。程序顺序是因果接线证据，不等于已有带时间戳decision receipt证明。该signal包含自主漂移，不追认为action-Jacobian。

报告保留源errors、失败attempt、empty/partial层，检查不通过写discrepancy，不筛掉不利根；非本固定输入schema的畸形文件/hash漂移严格拒绝。已完整receipt是本次主要输入；不能据本工具推断硬杀后未保存的work为零。

## 停止条件与确切缺口

runtime重算原先固定chi≥.5宽度、gain≥.05继续条件，不新增/拟合阈值，并区分最后一层stop和真正还有后继层的stop。joint不存在相应路径；工具只提供不靠outcome挑阈值的严格诊断：若已见**精确零**非负MSE，且未来(H,index)均不能打破稳定tie，则允许零下界证书；小正数、stagnation或prefix-best下降不能证明未见候选无改进。诊断可省后续predict数不是实际执行节省，始终标realized=0，未按保存结果配置新策略。

后续可执行补缺是独立新scheduler在模型dispatch前持久化decision及依赖的完整parent/children/signal数组、debited资源向量、未执行候选列表、预声明stop理由与验证reserve；动态改变剩余调度需真实执行新receipt。正代价的无损停机还需要有效未见候选lower bound，否则只能预声明有损策略并另测质量。当前四根是calibration，不是独立P3准入，不能以少1个winner验证predict宣传收益或等总FLOPs。

## 执行（纯保存证据）

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p3_adaptive_budget_saved_audit_20261027_v1.py
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.p3_adaptive_budget_saved_audit_20261027_v1 --output results/p3_adaptive_budget_saved_audit_20261027_v1_saved01
```

mock验证账本/阈值/停止证书逻辑，saved测试核对旧真实receipt；二者均不构成新神经、物理或科研收益实验。报告自带本审计源码/tests/doc/input SHA，输出冻结后不再修改。
