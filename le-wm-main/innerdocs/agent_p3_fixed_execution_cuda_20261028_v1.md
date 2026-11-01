# 固定五步执行请求：MAIN-only CUDA 接线 v1

独占 source/tests/本 innerdoc；不改 frozen contract/native/common/signal。只实现 root extension，不实现物理或 child 信号。源码直接调用 frozen `fixed.execute_panel`，没有另写生成器、替换目标、修改门槛或旧 globals。experiment-code 技能用于接口、持久化与回归；训练、图表、调参和非零结果要求不适用于本授权，零/负结果完整保留。

## 固定设计与真实预算

输入是 MAIN 已准备的合同 `results/agent_p3_fixed_execution_contract_20261028_v1_prepare_MAIN01.json`，外部实际 SHA `428f043c731098c5f6620b29d3bf180fc6912fd69e90b29a2d6f6166454650d6`。新 prepare 经 frozen verify 全量重建，并绑定自身三文件、该 prepare、继承全部 source/model/weight/config/原始双信号资产的 hashes。未知新 prepare/report SHA 由 MAIN 实际产生，本文不预填。

固定两个 known tasks × fixed/adaptive 四个分母；固定 H1 与 frozen model-signal 决定的自适应 H（允许 H1/H3，不强迫分离）。每臂完整独立已付 H1/H3 信号 8 API/8 samples、2 J/384 VJP；搜索 24 API/24 samples；赢家首包另付 1 API/1 sample，共 **33/33**。已付信号来自两个独立真实 panel，不重复运行信号，不把它记作免费，也不重计为新 CUDA 调用。

当实际 H=[1,3,1,3]：32 候选（12/4/12/4）、每个 whole+segmented 两次真实 replay，最后每臂独立 whole 首包一次。新 run 100 predict API/100 samples/300 context tokens；100 AE、100 predictor、100 pred_proj/300 rows；68 replay、36 metric、4 solve/4 arraySVD/4 program_compile；新 AD=0。合并已付信号后四臂总132 API/132 samples、8 J/1536 VJP。API/samples、AD/VJP、模块、tokens、SVD、嵌套 inclusive wall 分轴，不声称 FLOPs/wall 相同；后向 engine/CUDA ns/FLOPs 未计量为 null。历史采集开销仍见继承 prepare，不伪造到每臂的分摊。

请求真实执行长度均五步；实际物理尚未执行，因此 `equal_requested_execution_horizon=true`，`equal_actual_physical_execution_horizon=null`。`equal_planning_horizon` 从 frozen branches 实际计算，通常 false。H3 是根的联合15步规划，不是三个 child；只输出赢家前5步，10步后缀不执行，不 padding。H1/H3 仍按自身规划终点原 modelQ 选赢家；首包 modelQ 只是固定边界诊断，非共同任务成功。fixed 也付 H3 是 controlled shared-pipeline 消融，非最廉价 fixed-H1 强基线。PushT 原成功含 agent+object；不以 object-only 投影宣称完整任务失败。

## 运行与严格保存

物理对接 callable 为 `consume(output, actual_report_sha256)`，返回固定 slot-major、fixed/adaptive 顺序的四项 `prefixes`，每项 `{slot, mode, status, request, error}`。`request` 为 frozen contract 的 `.execution_request` 原对象：`binding/root_context_sha256/planning_horizon_packets/planned_action_steps/requested_execution_steps/commands[5,2]/commands_sha256/full_plan_actions_sha256/nonexecuted_suffix_steps/predicted_execution_endpoint_sha256/predicted_execution_modelQ` 等；commands 必须 literal float32，无归一化/重排/替换。另返回 `root_contexts=[{slot,binding,context}]` 两项（context 是 full3latent/2tokens **模型上下文，不是物理状态**）及 prepared/prepare_sha256/source_manifest_sha256。Hegel 的独占 `agent_p3_fixed_execution_physical_20261028_v1` 负责自己的 physical prepare/recipe/root/goal、全部7维状态及计量；本 worker 不写该 stem。consumer 的 model-prefix 验证不能代替 MAIN 的物理调度授权。

只有 `run --authorization MAIN_APPROVED_HOST_CUDA` 能调用 loader；loader 自身也先验令牌。此字符串是操作防误用，不是安全认证系统，实际审读/主机调度权属 MAIN。无网络、CPU neural fallback、训练或 physics。原 float32、完整3latent/2tokens ABI、原 action normalizer、identity Q、damping/生成规则全部继承 frozen panel。预封装 TF32=false/SDP=MATH、禁 autocast。按实际调用 hook 收 attempt/return，原始数组即时 blob、SHA、链式 fsync 保存；不把 RAM 测试当真实调用证据。

独占结果目录只能在 results/ 且本 stem 前缀、不覆盖；固定 binding.json/events.jsonl/report.json。先保存完整 candidate+prefix census、source manifest、weight/config hash，再模型加载。模型 state_dict 与包括 nonpersistent 的所有 buffers 前后 hashes；所有实际 model class 文件须预封存；最终重新 frozen verify，source/weight/config hashes 不变。资源/输出/存储/wall 失败保留全部分母、null 未返回值、已付 attempt、不重试、不换动作。若磁盘无法写最后报告，工程失败不能保证终态落盘，已有 fsync journal 保留；consume 关闭拒绝。

cap 继承合同：output256MiB、final reserve16MiB、wall900s（包括首次 saved verify）、floor200MiB、CUDA allocated3GiB/reserved4GiB。墙钟/内存 cap 是工程停止条件，不是科学阈值。full report 的 wall 在最后序列化/写盘之前采样，最终写盘开销不当作精确完整进程 wall；MAIN 可另记进程 wall。native 每次 IO CPU 持久复制会同步，ledger wall 为嵌套 inclusive 时间，不能累加当总 wall。

consumer：外部真实 report SHA → 全量 verify → exact artifact/binding/journalchain → 前后 source/weight/model closure → 完整分母 → 所有候选生成算术、原 Q 成本与稳定 tie-break → 资源/IO 双射及真实嵌套时间顺序 → 首包 raw/normalization/AE/predict/predictor/proj/端点闭包 → 完整33每臂。为复用旧 search validator，仅做 search-only **只读验证投影**（保留原 operation index/time，去除后发生首包账）；不生成新研究资产、不假装重新运行、不虚构第二个 prefix segmented replay。全部100真实调用仍以完整 ledger 再验证。unknown/额外 outcome 字段、孤儿数组/阶段、漏掉任意臂、未付 J、伪 child 均 failclosed。

Saved verify 本身只做 numpy 算术：两上游 consumers 每次合计8 array SVD；run 前后 verify 合计16；本 consumer 上游8+4生成器重算=12，和新 native ledger 的4 SVD 分开。权重仅 hash；CPU 允许 checkpoint 反序列化但不做 neural forward。上游已封存负结果只 hash/继承，不读取 outcome 设计此对照。

## MAIN 命令（worker 不执行 CUDA）

从 `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/agent_p3_fixed_execution_cuda_20261028_v1_test.py

.venv/bin/python -B -m experiments.spectral_reachability.agent_p3_fixed_execution_cuda_20261028_v1 prepare --contract-prepared results/agent_p3_fixed_execution_contract_20261028_v1_prepare_MAIN01.json --contract-sha256 428f043c731098c5f6620b29d3bf180fc6912fd69e90b29a2d6f6166454650d6 --output results/agent_p3_fixed_execution_cuda_20261028_v1_prepare_MAIN01.json

# PREPARE_SHA 替换为上一命令实际输出，不是预期/未来SHA。
.venv/bin/python -B -m experiments.spectral_reachability.agent_p3_fixed_execution_cuda_20261028_v1 verify --prepared results/agent_p3_fixed_execution_cuda_20261028_v1_prepare_MAIN01.json --sha256 PREPARE_SHA
.venv/bin/python -B -m experiments.spectral_reachability.agent_p3_fixed_execution_cuda_20261028_v1 queue --prepared results/agent_p3_fixed_execution_cuda_20261028_v1_prepare_MAIN01.json --sha256 PREPARE_SHA

# MAIN 审核后在 HOST CUDA 执行：
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p3_fixed_execution_cuda_20261028_v1 run --prepared results/agent_p3_fixed_execution_cuda_20261028_v1_prepare_MAIN01.json --sha256 PREPARE_SHA --output results/agent_p3_fixed_execution_cuda_20261028_v1_cuda_MAIN01 --authorization MAIN_APPROVED_HOST_CUDA

# REPORT_SHA 替换为上述实际落盘SHA。
.venv/bin/python -B -m experiments.spectral_reachability.agent_p3_fixed_execution_cuda_20261028_v1 consume --output results/agent_p3_fixed_execution_cuda_20261028_v1_cuda_MAIN01 --sha256 REPORT_SHA
```

联合回归：上面 pytest 命令追加 frozen contract/common_budget/native_program/local_signal_cuda/runtime_contract 的各自 `tests/<stem>_test.py` 路径。新测试冷导入在独立子进程，授权拒绝采用精准 import 拦截，不移除 sys.modules，可与已导入 torch 的其他测试同进程；没有模型实例/forward。

## 状态与未完成项

本交付仅代码+RAM/saved 验证，真实新 run/consume 由 MAIN 完成；未运行 CUDA/physics，没有收益或完整 C3 结论。仍缺固定五步物理 consumer/准确 root/recipe/goal 绑定、P1 独立机制准入、fresh cohort authority、任何真实 child 新观测与本地 J。所有 admission/release/full-C3/compute-advantage 标志 false；物理结果 null。不修改共享总账或上游封存。
