# P2 独占工程交付：saved 组合生命周期与对照消费

目标是“可执行组合及有效对照准备”，不是 P2 机制通过。本轮不使用子智能体，
不接管 P1 八计划或 moving_overlay_capability_probe，不修改任何既有 producer、
封存、阈值或 outcome。未启动神经、物理、训练、下载、删除或推送。

## 复用与新增边界

已读 README 与用户指定五份协议。既有 matched composition 是两段 beam/flat
搜索；paired v2 是真实 predicted-context suffix 编译；frontier CUDA v2 增加异常
和即时 derivative 数组保留；no-composition SRA v2 是联合 H2 单程序；full20
sampling v2 提供 CEM/random。它们不是严格全预算匹配，不能因 predict 相等
推导总算力公平。保持原始 Q、goal、动作域、候选生成及 stable-first 选择。

新增 `le-wm-main/experiments/spectral_reachability/chat_p2_20261027_lifecycle.py`
只为既有 saved graph 和 prospective queue 提供可执行生命周期，不另造 planner。
saved graph 的目标是已有非负成本的路径和，不是新 Q；不能用它替代未来神经
组合搜索或用物理 outcome 给边赋成本。它不从 latent 最近邻连接节点、不执行
动作、不补造动作可达性，不解析不透明 state/context 的物理含义。

- `prepare`：全图校验，封存所有边而非赢家；保存全部被引用 bytes 与当前三个
  本地运行源码的字节快照、固定搜索策略、完整 edge 分母。
- `verify`：要求外部 prepared canonical SHA，检查快照/current source 与全图绑定。
  SHA 证明字节一致，不证明时间先后、真实执行或身份授权。
- `run`：调用原 saved uniform-cost/beam 搜索；完整 endpoint 绑定 state kind、
  state、latent、history/context、全合同（包括 goal），原 raw 动作字节和 receipt
  全部保留。资源约束仍由原 graph 合同执行；缺测/null 不补零。
- `consume`：重算全 saved 搜索与报告的所有非计时字段，再导出原动作；不接受
  删候选、替赢家、伪造样本数或 admission。没有 path 或搜索超限不导出。
- `run_file`：输出前独占 started marker，失败写异常类型/消息与 prepared SHA，
  已开始/已输出均不得重试覆盖。进程死亡、磁盘故障无法保证 final artifact。
  saved 搜索没有模型工作退款；不把保存的生成成本当本次推理测量。
- `consume_producer(kind, output)`：只调用现有完整 exporter。SRA 和 composition
  用各自 selected_actions；sampling 先 load_prepared 复验所有 inputs/source seal，
  绑定 report source_seals 后用原三参数 exporter。没有 winner-only 降级路径。

输出仅在用户允许子树写 `chat_p2_20261027_` 新文件。没有神经模型入口；真实
world-model API calls 与 batch samples 均各自显式为本次新测量 0。保存的历史
成本与本次 consumer wall time 分开；FLOPs 公平、物理任务收益均不作声明。

## 使用

cwd `/home/gbk/code/LeWorldModel/le-wm-main`，全部命令加
`PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m` 后调用
`experiments.spectral_reachability.chat_p2_20261027_lifecycle`：

```text
prepare --graph /ABS/graph.json --input-root /ABS/evidence --algorithm shortest --max-expansions 10000 --output results/chat_p2_20261027_NEW_prepared.json
verify --prepared results/chat_p2_20261027_NEW_prepared.json --sha256 EXTERNAL_CANONICAL_SHA
run --prepared results/chat_p2_20261027_NEW_prepared.json --sha256 EXTERNAL_CANONICAL_SHA --output results/chat_p2_20261027_NEW_run.json
consume --prepared results/chat_p2_20261027_NEW_prepared.json --sha256 EXTERNAL_CANONICAL_SHA --report results/chat_p2_20261027_NEW_run.json --output results/chat_p2_20261027_NEW_actions.json
```

`run --confirmatory` 退出 2 并保存阻断。Python API 可传 manifest/load 诊断原队列，
但不接受 MAIN trusted authority，不签证、不提供确认性 dispatch；即使未来队列合同
完整，也须由主线程独立桥接 graph/queue/真实 executor 的身份和预算。
verify 成功、saved path 或 consume 成功均只表示工程 saved consistency。

## 本轮实测与现存缺口

RAM fixture 的 prepare→verify→run→consume 已实际执行并保存为
`results/chat_p2_20261027_ram_{prepared,run,consumed,validation}.json`；
确认性阻断与 started marker 另存。所有 fixture 只用于工程，绝非真实 CUDA 证据。
prepared canonical SHA：`5f5b9af24df6d491e3d236f1b00deb8a4861adf45c8770a337faf1afb6a716bb`。

纯 saved 验证记录在 `results/chat_p2_20261027_saved_validation.json`：

| 既有实现 | 本轮只读消费结果 |
| --- | --- |
| no-composition SRA v2 prepared01 | 原 exporter 完整校验通过 |
| full20 sampling v2 cuda_v1 | 原 inputs/source/完整动作 exporter 校验通过 |
| composition frontier CUDA v2 prepared01 | 拒绝：只有 prepared，无完整 report/commit 可消费 |

未读取物理 outcome 做规则选择；没有把 v1 或 paired 的历史结果替补进 v2。
SRA/sampling 通过不证明两者与 v2 已完成同一批真实配对，不证明组合优势。

回归命令：

最终联合回归：390 passed in 23.17s。最初 scope-only 联合回归 279 passed；
扩展回归首次有一项新增测试的异常类型断言与强化后的拒绝语义不一致，
修正测试后上述最终命令全部通过。没有修改既有测试。

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider \
 tests/chat_p2_20261027_test_lifecycle.py \
 tests/test_moving_p2_saved_program_graph_20261027_v1.py \
 tests/test_moving_p2_composition_prospective_queue_20261027_v1.py \
 tests/test_p2_composition_predict_frontier_cuda_20261027_v2.py \
 tests/test_p2_no_composition_sra_20261027_v2.py \
 tests/test_p2_full20_sampling_20261027_v2.py
```

## 精确待放行条件（不改变现有门槛）

1. MAIN 提供原 prospective queue 合同的真实 P1 六 gate 和预声明指标证据、
   独立统计审查、外部 authority SHA；P1 先于 P2 seal，禁止本线程自签。
2. 完整 freshroot 历史排除投影与无未解决来源，根/history/blind/task identity
   队内唯一且不复用 P1/历史根；MAIN 审查物理别名，不靠不同 SHA 宣称独立。
3. 预先冻结全部 roots/goals/history/Q/动作域/模型、规则、seeds、候选分母；
   将真实 producer 的身份逐项绑定队列，而非只通过 graph 或自述 schema。
4. MAIN 评审后仅在主机 CUDA 用现有 producer 运行，冻结 eval、前后参数/buffer
   和 source 检查、全候选生成动作见证及 segmented/full replay；失败不得补候选、
   退款或过滤分母。当前组合 v2 完整执行及 saved consumer 尚缺。
5. 实现并审查共同完整资源上界/停止条件及实际测量：API calls、batch samples、
   编码、AD/JVP/VJP/backward、solve/SVD、共享前置、失败/replay、physics、wall/CUDA
   和 FLOPs；未知不能填零。既有 SRA/sampling/组合不自动成为全预算有效对照。
6. 按原协议完成全部候选物理任务收益、非贪心机制和配对不确定性验证。
   未满足前不宣称 P2 成立；本交付不接管这些 P1/主线程工作。
