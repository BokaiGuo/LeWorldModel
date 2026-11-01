# P3 动力学驱动搜索运行接口（2026-10-27）

P1 尚未通过，P3 未准入。此实现只用于接口、因果调度与成本校准。当前 Q
存在 agent 动而物体静止的能力假象，沿用它绝不表示修复任务 efficacy。
不改已有 seal、源码、报告；不训练、不学 readout、不挑 outcome 根。

## 新增能力与复用

`p3_adaptive_runtime_20261027.py` 直接调用已有 `search_programs`，每个父节点
做一次 depth=1 的完整固定动作库展开，保留其原始 attempt/失败收据，汇合后
按 (Q-MSE, depth, primitive-index sequence) 排序。不是再造搜索内核或预算合同。
生产调用仍由已有 CudaReplay 和 PredictionBudget 执行并扣费。

每层完整展开后，针对每个父状态 x、已预测子状态 y 和冻结目标 g：

- R 的每行为 (y−x)Qᵀ，r=(g−x)Qᵀ。对 R 做薄 SVD，相对截断 1e-6。
- χ=||r−Proj_span(R)r||²/||r||²，gain=clip(max(Rr)/||r||²,0,1)。
- 同层 χ 取最大、gain 取最大，交给已有 FrozenSchedule。gain≥0.05 允许继续
  至深度上限 3；否则本层后停止。χ≥0.5 时下一层保留 8 个父节点，否则 4 个。
- 每层重新计算；使用本层及过去已付费预测，不访问 reward、success、终态物理
  结果或 evaluator。既非静态 early-stop，也不按 outcome 标签分配预算。
- R 未去除自主漂移，因此是有限动作响应几何，不是已识别的控制 Jacobian，
  不能称为谱控制定理、可靠可达性证书或收益。SVD 无参数训练、无 readout。

不完整/失败展开不生成调度信号，不能用成功子集伪装完整几何。全部失败与已消耗
资源保留。选全部已访问非空节点中的最好计划，不补零、不强制返回最后深度。
固定组也计算自身完整信号，但忽略调度结果，始终 beam=4、最大深度=3。
这样信号开销在两组均有记录；后续仍须另做无信号固定组和打乱/噪声信号消融。

## HOST CUDA runner 与成本范围

`p3_adaptive_runtime_cuda_20261027.py` 使用同一四个已封存校准根、16 个命令、
官方 checkpoint，全部根保留，逐根交替两组先后顺序。正确上下文为 emb3 与
两个 between-state 动作（saved actions[1:]），随后每个新五步 packet 才 predict
一次；不把三个 incoming 动作拼入 root-anchored recurrence。

每组每根 planning 最多 144 个逐边预测样本，独立预留最多 3 个完整动作串重放样本，
一个 PredictionBudget 总上限 147，跨 planning/evaluation 不重置。固定组正常工作量
16+4×16+4×16=144，重放为最佳计划真实深度 1/2/3，不能强行记成 3。
自适应组 under cap 不等于相同用量；每轴差值由既有 compare_paired 核算。
尝试次数上限不是预测收据；真实异常不退款。最多 8×147=1176 predict samples，
无 batch 放大，串行，最多 15 个真实动作长度的想象串，不做物理执行。

完整 Resources 逐项提供：实际 predict calls/samples、视觉 encoder hook 计数、
尝试 replay trajectories、新 imagined token 数 B、尝试完整前缀最大时域。
action encoder 每次输入 [B,T,10]，T=3：`imagined_action_tokens` 累加 B，
`action_encoder_samples` 也累加 B；另设 `processed_action_context_tokens` 累加 B×T。
后者含两个历史 token，不能算成三个未来预测。每根各组前两项上限 147，
processed action context 上限 441；附加 gate 核对 processed=3×samples 且
imagined=samples。失败前已进入 action encoder 的工作仍计入，不退款。
action encoder forwards/samples/context tokens 另列，不混入视觉编码计数。
无梯度路径、无物理环境，故 VJP/backward/physical 各轴为已知零。
完整 search+signal+verification 放在一个共同预定分支 `search_and_verification`；
不按 imagined candidate 或成功分支筛选。失败时 task_success 仍 false；没有测量
物理 success，两个组的 successes 不能解释为任务失败率或效益指标。

信号 SVD 用非神经 NumPy CPU 数学，计次数、响应规模与耗时；所有神经前向只在
HOST CUDA。每根实测墙钟含信号和重放，保存 action encoder 工作与峰值显存。
上游已存在 compilation、checkpoint load、source hash 是冻结的共享排除项；
模型加载时间单列，不冒充完整端到端等预算。新源码以协议内 SHA 清单前后核验，
不是替换或重封已有 archive seal；原 validate_inputs/verify_seal 始终严格执行。
若共享源码变化导致旧 seal 失败，应停止并交主线程，不能降低校验或重新盖章。
协议与所有新 JSON 文件名均含 P3_ADAPTIVE_RUNTIME，exclusive 创建，不覆盖旧目录。

## 主线程启动命令（本任务不启动 GPU）

在 `/home/gbk/code/LeWorldModel/le-wm-main`，先无分配预检：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m experiments.spectral_reachability.p3_adaptive_runtime_cuda_20261027 \
  --compilation /home/gbk/code/LeWorldModel/le-wm-main/results/pusht_d2p_metric_program_preflight4_v1_20261026_cuda/report.json \
  --policy /home/gbk/.stable_worldmodel/checkpoints/pusht/lewm \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/P3_ADAPTIVE_RUNTIME_20261027_cuda_v1 \
  --preflight-only
```

主线程确认调度后，去掉 `--preflight-only` 即为完整运行命令。只使用一张主机卡，
没有 CPU fallback。已有目录必须另取含 P3_ADAPTIVE_RUNTIME 的全新路径。
预测成本上界 1176 次 batch1 predictor 与 action encoder 前向；不重新编码真实图像。
保守预留 4 GiB 显存、100 MB 磁盘、10 分钟运行窗口（估计非实测，含 JSON、校验与
Python/SVD 开销），满足现场 16 GB 卡、2.5 GB 空间的预计需求。不复制权重/数据；
若失败保留目录和收据，不删、不覆盖、不自动重试。初始化硬件/载入异常可能只留协议，
不可把它算作完整 paired 结果。

## 验证与剩余科学缺口

测试使用纯非神经图 fixture，仅验证调度确实随预测动力学变化、目标不变时的深度
改变、不可达分量改变宽度、并非 cost stagnation early-stop、原始动作递推、完整失败
保留、硬 expansion cap、严格字段和零资源路径。不是 CPU 神经实验。

指定 `.venv/bin/python`、`PYTHONDONTWRITEBYTECODE=1`、pytest `-p no:cacheprovider`：
新增测试 35 passed；连同 bounded search、预算、配对核算、instrumentation、ledger、
原 CUDA search runner 回归共 267 passed。无分配预检通过：4 根、52 个源码文件，
旧 upstream seal verified。均未启动 CUDA；预检 fingerprint 随本次最终修订变化，
主线程启动时重新计算并封存，不能把先前预检 hash 当最终源码 hash。

仍缺实际主机 CUDA 新 runner 验证、P1 完整准入、独立前瞻根、真实物理任务后果、
可靠物体度量、噪声/打乱信号、成本曲线和更强固定/自适应基线；任何集成完成标志
均不代表 P3 admission、compute advantage、walltime advantage 或研究闭环。
