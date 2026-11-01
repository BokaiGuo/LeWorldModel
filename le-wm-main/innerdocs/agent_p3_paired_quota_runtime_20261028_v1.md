# P3 paired predict-quota runtime v1

新 schema / 新实验；不修改旧 contract、chooser、失败结果或任何封存文件。
本实现 worker 仅运行 RAM/已存数组算术测试；HOST CUDA 与科学验收由 MAIN 执行。

## 严格算法与费用

固定原24child、48method槽、2known roots及原 observed ancestry/context/task/goal。
每 observed child：fixed 独立支付 H1/H3 signal，adaptive 再独立支付同样 signal；
两份完整 signal 及原 chooser decision、source/context、完整 signal/账 SHA 先 durable pairseal，
任何 compiler 只能在此之后进入。不得借根 J、共享付费缓存或读取物理结果。

原 `decision` 原样保留；新的 `execution_quota` 独立字段：H1=12、H3=4。
真实生成 i/(N-1) 的全部 N 个 raw 候选；每个独立 whole H + segmented H replay，
全部原 Q terminal MSE 后 stable (cost,index) 选择，selected raw5 再独立 replay。
零候选及重复候选照算，不去重；原 bounded_ridge/float32 raw/normalizer 不变。
当 execution_count < 原 requested_count，或原 extension 非可定义 progress，整 pair paid-refuse。
例如 H3×8 不能缩为4；另一侧也不编译。拒绝不退款；not_reached 不伪造付费/输出。

每个准入 method：signal8 + search24 + selected-prefix1 = 33 predict API / samples。
每个 paid-refused method：8 predict、2J、384scalarVJP/grad/raw，保留原decision与拒绝原因。
H1×12 的 replay25/metric12，H3×4 的 replay9/metric4；signal各2SVD，compiler另1SVD/1solve。
两侧只匹配实际 predict API/sample；不宣称相等 FLOPs、完整资源向量或 wall。
原391 vs634属于旧实验，不用于宣称新实验匹配。若新付费decision仍为18可准入pair、
5拒绝pair、1未到达pair，预测的两侧各18×33+5×8=634仅条件演算，不是新实测。
共同 method硬上限24×33=792predict；其余轴取两个合法执行形状的逐轴最大值，
全部计数保留单位/nesting，gate是actual ledger镜像，不再叠加收费。

## 文件与消费闭包

同stem主入口 `.py`：prepare/verify/HOST生命周期/CLI；`_gate.py`：pair状态机、
predispatch钩子与Session；`_compiler.py`：真实12/4编译与完整原生IO校验；
`_consumer.py`：流式重建逐笔gate→native entry→native return→gate return、
两侧signal/barrier、全部候选、原Q、prefix、ledger及模型/source终态。
`tests/agent_p3_paired_quota_runtime_20261028_v1_test.py` 是RAM算术，不是CUDA或科学证据。

prepare必须完整调用冻结旧host.verify重建原全strict来源，仅继承tasks/context/model；
旧host.run/旧schema不执行新实验。所有新增分层源码/tests/doc均列入source closure。
NativeFactories/NativeChildSignal/NativeProgram直接调用冻结真实公共接口，不猴补旧globals。
consume不加载模型、不跑AD或physics，完整校验signal全数组/逐方向J与compiler全部原始IO，
重算SVD/ridge/Q仅saved算术；不把日志当GPU硬件证明。完整48终态/无错误才能签发strict。
失败保留所有槽、已attempt资源/partial arrays、pending ticket并fail-stop；不换策略重试。
consumer拒绝失败/partial；不将未执行或物理结果补0。

## 工程边界

输出512MiB，256MiB留给最终report（journal+binding≤256MiB），disk floor200MiB；
wall1800s从verify之前至同步/模型/source guards/journal close，最后report写入除外；
CUDA allocated3GiB/reserved4GiB；流式decoded journal≤2GiB、单行64MiB、单phase trace64MiB；
saved receipt≤256MiB。没有数学容量保证，真实cap不足保存工程失败，不调配额/删full arrays。
run 工程失败保留report/原始日志并返回非零退出码，stdout仅compact summary。
CUDA-only、TF32off、MATH SDP、autocast off、冻结eval/parameters/buffers前后hash、offline。
授权检查先于verify/import/allocation，NEW排他输出；不执行encoder/physics/新训练。
P1/P3/C3/physical release科学门始终false，仅已知根工程校准。

## MAIN精确入口

所有路径绝对；下列 `<PREPARE_SHA>` / `<REPORT_SHA>` 必须替换为对应命令真实返回SHA，
不是预填未来值。从项目根运行模块：

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
/home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python -m pytest -q /home/gbk/code/LeWorldModel/le-wm-main/tests/agent_p3_paired_quota_runtime_20261028_v1_test.py
/home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python -m experiments.spectral_reachability.agent_p3_paired_quota_runtime_20261028_v1 prepare --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_child_budget_host_cuda_20261028_v1_prepare_MAIN01.json --prepare-sha256 92f2edacae8e0f519e09d573ed1c78c61247e78b7b99042e27e29f39868146c8 --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_paired_quota_runtime_20261028_v1_prepare_MAIN01.json
/home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python -m experiments.spectral_reachability.agent_p3_paired_quota_runtime_20261028_v1 verify --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_paired_quota_runtime_20261028_v1_prepare_MAIN01.json --prepare-sha256 <PREPARE_SHA>
/home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python -m experiments.spectral_reachability.agent_p3_paired_quota_runtime_20261028_v1 run --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_paired_quota_runtime_20261028_v1_prepare_MAIN01.json --prepare-sha256 <PREPARE_SHA> --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_paired_quota_runtime_20261028_v1_cuda_MAIN01 --authorization MAIN_APPROVED_PAIRED_QUOTA_CUDA
/home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python -m experiments.spectral_reachability.agent_p3_paired_quota_runtime_20261028_v1 consume --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_paired_quota_runtime_20261028_v1_cuda_MAIN01 --report-sha256 <REPORT_SHA> --receipt /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_paired_quota_runtime_20261028_v1_consume_MAIN01.json
```

不运行神经/物理的worker不替MAIN执行上述prepare全strict重建或run；CLI与真实saved字段的
RAM边界测试不能称新actual CUDA通过。新实验是否有任务收益仍unknown；无fresh cohort、
完整树或native任务认证，本版不执行物理桥。
