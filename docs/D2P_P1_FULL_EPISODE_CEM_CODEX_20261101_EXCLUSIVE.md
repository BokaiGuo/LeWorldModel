# P1 全程 CEM：控制器与计费准备接线

范围是独占新文件中的新控制器，不改其他线路、不执行 GPU 或物理环境。
本方案从决策根 cp0 开始，不复用 Leibniz 正在处理的 cp5 前缀物理重放或其输出。
Leibniz 的 cp5 单节点 H3 CEM 与这里的四决策节点闭环是不同对象。

## 对照设计

参考 first-feedback carry 的两个已知根、相应 cp10 目标、模型、动作归一化、
投影目标和决策时点 0/5/10/15，固定 20 个最大控制步；新 CEM 每次实际反馈
独立重规划，执行选中序列首个五步包。四节点模型规划 horizon 为 H3/H2/H1/H1，
与该对照的有效选择 horizon 一致；两臂后续状态和实际 RGB 必然可以不同，
相同的是机会及任务，不是观测值。若 native 终止或截断，保留实际已执行步数，
不得填充、续跑或借用其他轨迹。环境 receipt 必须将 canvas censoring 等非原生
中止明确映射到 `stopped=true` 并额外记录原因。

每节点的 CEM：独立 PCG64 流 `seed + node_index`，四轮各 16 个 raw
`[H,5,2]` 候选，初始均值 0、标准差 0.5、clip [-1,1]，第 0 个是当前均值；
按模型终点投影误差和候选序号选 4 elite，使用 population std 并夹至 [0.05,1]。
全 64 候选中按 `(分数, 全局序号)` 选一个并额外复评；复评分数不符即停。
没有使用物理成功、终态或事后选样；这个实现不是官方 CEM。
seed 必须在正式新运行前由合同固定，控制器将其写入 journal。

完整四节点的**计划调用量**为 260 次 rollout 和 selection score，
每节点 65 次；action encoder / predictor 各 455 次（`65*(3+2+1+1)`），
predict samples 同样为 455（每次 batch=1）。这是计划量，不是实测费用：
每次调用此前写 attempt，返回后写 return，失败保留 failure，禁止将缺失视为零。
实际完成与否由事件核对；动作派发以 packet attempt 记录，物理步数须从经过
认证的环境侧逐步回执另行计量，不能把最多 20 步当作实际成本。
日志不覆盖模型构造、共享目标、校验、环境内部计算、FLOPs、时延或能耗；
没有预算硬上限、同步 GPU 时延测量或完整资源收据。与比较臂不声称等成本、
equal FLOPs、等时或公平优势。节点最外层 rollout 与内部模型调用嵌套，不能相加
作为独立算力。观测回调若声称新编码必须通过提供的 metered leaf 恰调用一次；
`inherited` 表示没有新编码尝试，不表示历史编码免费。

## 实际来源与关键错配

只读检查了两个 first-feedback 的 `PREP_MAIN02` 和其保存 report，以及既有 cp10
target receipt；脚本校验这些文件固定 SHA、节点顺序和目标像素/latent binding。
原 prepare 的 `source.tasks` 仍是原目标、原 root binding；实际运行先重绑定到
cp10 target 和新 root。来源脚本据已保存目标和实际 report 还原新 task identity，
并另外计算 float32 goal/normalizer 和 float64 projection 的字节 SHA，不能把这些
字节 SHA 与上游 JSON 语义摘要混同。没有复核所有传递来源、重新加载模型、
重新渲染根或证明运行时身份；这些仍为未来适配器的 fail-closed 门槛。

## 实现接口和未闭合门槛

控制器调用现有 `_normal` 模型 rollout，通过包装的 leaf 记录 encode、
action_encode、predict 的写前尝试，写入完整 CEM 候选、得分、选择与动作意图。
journal 使用 O_EXCL、逐事件 fsync 和链式摘要。模型或动作调用异常后永久停止；
物理派发异常可能已发生真实部分效果，严禁自动重试。

`Controller` 必须由独立审查的真实 adapter 提供同一模型 leaf、actual RGB 编码、
最近三个真实 latent 和最近十个实际 raw action 构建的归一化历史，以及逐步环境
派发/回执。adapter 必须事前核实 checkpoint/模型权重、目标像素和 latent、
root reset/warmup/状态/图像、正常化与任务源，并在每节点检查真实 RGB 与执行
历史的因果链。控制器检查 goal、normalizer、matrix 实际字节及身份字段匹配，
但不独立证明回调字符串、历史来源或动作 receipt 真实。不能仅凭源映射或
fixture 测试宣称以上门槛通过。

尚缺真实适配器、受审的新运行合同与预算/资源限额、GPU 独占和明确执行授权、
物理逐步写前/返回回执、已封存来源闭包复验与严格事后 consumer。故本代码
是可调用的控制器核心和准备接线，**不是**可直接投入真实环境的闭环 runner，
无真实强基线结果，也无 CUDA/物理执行命令。

## CPU 检查

```sh
cd /home/gbk/code/LeWorldModel/le-wm-main
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.codex_p1_full_cem_source_20261101_exclusive
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_codex_p1_full_cem_controller_20261101_exclusive.py tests/test_codex_p1_full_cem_source_20261101_exclusive.py tests/test_codex_e1_proposal_durable_events_20261101.py
```

测试中的模型、RGB、行为回执均为 CPU fixture，只验证接口/计数/失败语义。
