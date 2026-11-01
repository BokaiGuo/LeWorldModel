# P2 runtime guard：真实 dispatch 接缝，known-root 工程专用

新增本stem源码/tests/本文，不改原frontier v2、next、封存结果或共享文档，不派生后代。
使用experiment-code技能固定预算、durable日志、失败回归；不训练、不拟合、不改原内核，
不做outcome筛选。worker仅运行RAM metadata／saved验证，没有GPU、CPU神经或physics。

## 真正接线位置

原frontier-v2 SHA `bf8ae74d7453cbf19f2420c8c83701f8e5c893f08f37293e7960cd84ac2ad061`。
新 `run_known_root` 直接调用其 `compile_cuda`、`suffix_cuda`、`candidate_bank`、
`run_arm`、`persist_array`、`guarded_entries`、entry/operation validators；只替换外层
orchestration以注入新sink，不修改数学、候选、目标、搜索或replay，不猴补模块globals。
native既有guard按原方法实例安装临时model.predict/action_encoder.forward钩子并finally恢复；
visual encoder仍被禁止，输入必须原CUDA batch1×3 context。

实际顺序：

1. shared/candidate/replay组reserve intent，落盘flush/fsync后才允许该组调用。
2. native entry_attempt或AD/SVD/solve request进入sink；primitive reserve + entry_boundary
   分别落盘fsync后，才返回原runner并继续原始调用。
3. native entry_terminal/operation terminal进入sink，独立写return_observed；此记录不是intent。
4. wrapper实际包住原target_metric调用，预留selection_evaluation后调用，再记录return；
   若后续cost标量检查失败，已付费metric不退款。

`entry_boundary`精确定义为**原host API/operation调用前的dispatch边界**，不是“CUDA kernel
已经启动/完成”。断电发生在fsync与函数入口之间时不能证明已调用，故不把entry累计为已完成
工作。成功host return才进入observed_returned_lower_bound；CUDA异步错误还须每根synchronize，
失败保持root不完整。数学Jacobian/VJP方向是原源码合同计数，不是backward-engine API/FLOPs。
root/suffix AD return事件在原数组持久化之后，因此AD已算完但转换/落盘失败时，guard仍记
pending而非猜测完成；原derivative_stages同时保留returned/converted/persisted独立状态。

## 收费与未知值

接next的 `allocation()`：base4 predict（并各臂归属全shared）、joint80×2、
conditional40×4、各臂replay4；standalone168/168，paired unique332/root、四根1328。
每candidate的primitive分配均不超过其原base/edge组上限；组预算只收一次，primitive
reservation是组内细分而非再次累加到总预算。AD包含的forward单独从真实模型入口计数，
不靠公式补记录。shared在paired unique只计一次，不能把standalone相加当unique。
原上限为suffix SVD/solve≤1，zero-radius实际0不抹掉原reservation。未知8轴保留null，
任何有限未知轴（含0）由既有supported guard拒绝。

输出分栏 `reservation_intent_standalone`、`observed_returned_lower_bound_*`、
`pending_operations`、`failed_returns`。pending或失败return使actual_complete_totals=null，
不把缺失完成量填0。返回错误不能证明内部完成量；已返回下界不代表全部实际费用。
physics/warmup/endpoint encoder根本不dispatch，保留合同中未来预算空间但不能声称这些
费用已执行或完整对照已完成。上游context/RGB/Q获取、FLOPs、backward engine始终null。
原native CPU SVD/有界solve属于已冻结规划器数值内核，不是训练、Q拟合或CPU神经。

## durable与失败

独占JSONL、每条带seq/前条SHA/本条SHA，flush+fsync后返回；初建文件parent也fsync。
短写/fsync/空间/日志大小/边界wall异常永久poison，不重试、不截断、不resume。
原Faults.journal保留计算primary与持久化secondary，不能用cleanup异常覆盖primary。
reserve成功而entry失败时保留reservation/pending；计算返回但return fsync失败时仍pending。
原AD nominal/J文件按原persist_array单独fsync封存，与intent日志分离。

每根日志上限16MiB、每根900s的**调用边界检查**，不是中断正在运行CUDA的watchdog。
磁盘floor200MiB；执行前要求400MiB可用（200MiB预留输出额度+200MiB floor），
native数组/行文件沿用原exclusive writer；不是对所有输出实现200MiB强制总量cap。
原native arrays/行写失败仍留日志与已有文件，不覆盖或删除。journal audit只验证链/字节
完整性，不是神经计算真实性证明，也不把seal自述当科学证据。

每根保留80+40候选，根失败后该根未启动槽为skipped；后续根保留skipped，report固定
4root/8arm/480候选。row写失败前已用账进入内存census，不能把已执行根重标成未运行。
无法写最终report时只能依赖已durable日志/文件，程序不保证磁盘故障下最终report必有。
无重试、补根、padding、跨臂转借；hook恢复失败必须终止并丢弃该运行model实例。

## prepare/verify/MAIN执行入口

prepare重验next全saved接缝与原native load_prepared/source seals，并精确join四根完整
identity（goal/history/model/action域），绑定本三文件和原闭包。不重封旧seal。
CLI prepare/verify/audit-journal不载模型；execute-known显式走原HOST CUDA loader，
同原模型冻结/eval，参数/buffer前后哈希、结束source再核。没有CPU fallback或下载分支。
prepare另检查本地绝对checkpoint目录内weights/config存在、encoder.pretrained严格false；
调用加载函数沿用原封存本地模型路径，不调用hub下载。report保存参数/buffer前后两组SHA。
真实CLI prepare曾发现旧writer仅接受旧resume前缀、拒绝新runtime_guard前缀；当时没有
prepare文件写出。现改用本stem自有exclusive canonical writer并补CLI实际写出/拒绝覆盖测试，
不是改旧writer。wrapper prepare文件SHA与canonical prepare SHA相同。

在inner repo：

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/agent_p2_runtime_guard_20261028_v1_tests.py tests/test_p2_composition_predict_frontier_cuda_20261027_v2.py
# NATIVE_SHA=1d79f5b17002334434454dfb5a94e6d918de1d603afdf84918e5ded6eb980d1b（canonical，不是protocol SHA）。
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p2_runtime_guard_20261028_v1 prepare --native-prepared results/p2_composition_predict_frontier_cuda_20261027_v2_prepared01 --native-sha NATIVE_SHA --output results/agent_p2_runtime_guard_20261028_v1_prepared01.json
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p2_runtime_guard_20261028_v1 verify --prepared results/agent_p2_runtime_guard_20261028_v1_prepared01.json --prepare-sha WRAPPER_SHA
# 下条仅MAIN审读、外部留存review文档SHA后调度；worker不执行。
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p2_runtime_guard_20261028_v1 execute-known --prepared results/agent_p2_runtime_guard_20261028_v1_prepared01.json --prepare-sha WRAPPER_SHA --main-review-sha MAIN_REVIEW_SHA --output results/agent_p2_runtime_guard_20261028_v1_known_MAIN01
```

main_review_sha仅记录MAIN外部审读引用，字符串不是密码或P1 authority；MAIN负责信任和调度。
可Python调用 `execute_known(prepared, sha, output, main_review_sha256=...)`，没有预载模型绕过。
confirmatory参数在读取/加载前永远拒绝；只有known-root engineering路径。原生输出source
provenance仍是原runner，wrapper prepare SHA另外绑定，不能冒充原v2独立执行结果。

## 交付验证与限制

实际测试/prepare/verify数值与三文件SHA见交付消息。RAM测试用metadata假callable验证原真实
method guard的664入口、组内收费、fsync拒绝在调用之前、return丢失、原primary、零半径、
完整失败分母；tmp只写metadata durability，未实例化模型/环境。saved prepare验证原资产布局，
不运行AD、模型或physics。测试不是科学证据。
另以原run_arm不变、compile/suffix替换为metadata回调，完整运行wrapper根级路径的80/40候选
和664入口及两臂独立replay；只证明callback接线，不能称实际CUDA或新物理证据。

独立P1/freshness仍未通过，本模块不解锁P2、不改科学门。MAIN仍须实际HOST CUDA运行，
独立读journal/arrays/模型hash、物理执行与exact恢复/goal7/成功合同、full-resource profiler
及fresh确认。当前工程进展是**实际dispatch sink接线**，不是完整神经/物理实验已完成。
