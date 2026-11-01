# P3 child paid signal v2：仅无损存储工程修复

独占三件套：`experiments/spectral_reachability/agent_p3_child_signal_cuda_20261028_v2.py`、`tests/agent_p3_child_signal_cuda_20261028_v2_test.py`、本文。旧 v1、contract、MAIN 文档及结果不改。worker 仅 RAM/saved 测试；真实 HOST CUDA 由 MAIN 审核授权执行。

## 不变边界

同 child 独立付费 H1/H3，8 API/8 samples/2 J/384 scalar VJP/384 grad/384 raw hooks/2 SVD；两个方法分别付费，不借 root J、不以缓存减少计算。不改 choose、policy、candidate 数、H、阈值、normalizer、模型、scope、full float32 arrays 或完整队列。24 child/48 method 分母；当前 23 observed/1 not reached，independent 模式全部 46 acquisitions：368 predict、92 J、17664 VJP、92 SVD、46 schedule。shared_equal 原合同仍支持，但必须 prepare 时明定，不得事后切换节省计算。全部 scientific gates 保持 false；不生成动作，不执行 physics。

512MiB 总输出 cap、256MiB final reserve、900s wall scope、200MiB disk floor 及 CUDA 内存上限不变。whole-job wall 从 verify 前至 journal close/最终源检查，最终 report write 不包含在该 wall 内，不宣称端到端 900s。压缩成本包含在原运行 scope，不增加 wall 配额。prepare/consume 的 saved 核验无神经调用。

## v1 失败与容量预测

真实失败 report：`/home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_child_signal_cuda_20261028_v1_cuda_MAIN01/report.json`，SHA `c0279c6a48d86bb46ed90b1d098bd436c76a9e85daa00893d34a119afa574641`。唯一工程错误 `ValueError: output engineering cap`。report 保留 19 returned/4 blocked/1 not reached；journal 仅 18 完整 child_return（第十九返回行已在 RAM，未成功写 journal）。不可把二者强称相同，更不可把 v1 当科学负结果或完整 release。

封存 journal SHA `41ee38987939097eff34f14bafc15dffecd488c7a689f0fb3be56a86f19a690e`，267806913 bytes/152326 events/38 groups。只读 RAM 测量：每 group 单独 level6 gzip，原行逐条 Z_SYNC_FLUSH、组末 Z_FINISH，最大 raw 5105623、gzip 564813 bytes；每 child_return 单独 gzip，最大 raw 4671684、gzip 1200584 bytes。其余 metadata gzip 合计 1123 bytes。完整 report 最大 row canonical bytes 4671473，去 rows 的 header 3235 bytes。

prepare 内封存 capacity_forecast，同时外部 SHA 绑定上述真实失败 report/journal。全 46 acquisitions+23 returns 预测：

- journal：`2*(46*564813+23*1200584+1123)+16MiB = 123969122 bytes`，低于 normal 268435456 bytes。
- 未压缩 report：`2*23*4671473+16MiB = 231664974 bytes`，低于 final reserve 268435456 bytes。
- 预测合计 355634096 bytes，低于总 536870912 bytes。16MiB allowance 各自涵盖 binding/header/source metadata；normal 实际计账包含 binding。

这是观测最大值两倍加余量的**保守经验预测，不是任意未来数据熵的数学上界**。实际无损压缩可能不符合经验比率：运行前/每事件仍以实际压缩字节检查 normal cap，最终 report 自身不超过原 256MiB reserve，全部实际文件之和不超过 512MiB；超限仍 fail closed 保留失败。不会增加 cap、删队列或减少分母。本次测量不是新的成功 CUDA 结果，也不把失败 report 冒充 v2 结果。

## 存储与严格消费

仅 `events.jsonl` 改为 `events.jsonl.gz`：单 gzip member level6，逐事件 sync-flush、flush、fsync。原 canonical JSON 行（含 index/previous/payload/sha256）逐字不变，全精度 blob 不量化、不丢弃重复项。每次写先复制 compressor，获得实际压缩长度，通过 cap 后才提交；拒绝事件不推进 hash/index/compressor。每次检查预留 32 bytes gzip finish，close 写完整 footer。中途停机可能留下无 footer 的 durable prefix；只用于失败追踪，strict consumer 不 release。report.json/binding.json 保持普通 canonical JSON，单独实际 SHA。

consumer 不展开到磁盘、不持有全 journal list；流式解压，单 child 完整 traces 原样交独立 validate_trace/SVD/choose/queue/ledger 审核后释放。全 report 本身仍一次读取，故非内存常数接口。decoded 总 cap 1GiB、单行 64MiB、输入压缩文件受输出 cap 限制；解压块至多 1MiB。拒绝 truncated footer、CRC/位翻转、额外 member/trailing bytes、非 canonical 行、链篡改、跨 child/group/重排/重复 guard、缺失返回和任何原 source/数组/付费/配额检查失败。外部 report SHA、完整 prepare 重建及全部文件末尾 SHA 再核，固定分母不变。新文件只使用独占 v2 schema，不 monkeypatch 冻结 globals。

## MAIN 精确入口

所有路径 absolute；以下命令从仓库根运行，使用已有环境。prepare 绑定已知真实 child contract SHA，不把 v1 signal prepare 当新 prepare。prepare 后须从实际打印值填入 `ACTUAL_V2_PREPARE_SHA`；run 后从实际打印值填入 `ACTUAL_V2_REPORT_SHA`，这里没有预造未来 SHA。

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
/home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python -B -m pytest -q /home/gbk/code/LeWorldModel/le-wm-main/tests/agent_p3_child_signal_cuda_20261028_v2_test.py
/home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python -B -m experiments.spectral_reachability.agent_p3_child_signal_cuda_20261028_v2 prepare --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_child_scheduler_contract_20261028_v1_prepare_MAIN01.json --sha256 8669fb219c6a17e05d5c1b8dc557b401651a2bf3fc759529d64231e5d317f7fd --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_child_signal_cuda_20261028_v2_prepare_MAIN01.json
/home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python -B -m experiments.spectral_reachability.agent_p3_child_signal_cuda_20261028_v2 verify --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_child_signal_cuda_20261028_v2_prepare_MAIN01.json --sha256 ACTUAL_V2_PREPARE_SHA
```

仅 MAIN 审核批准的 HOST CUDA 执行；默认缺 authorization 在 verify/import 模型之前拒绝：

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
/home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python -B -m experiments.spectral_reachability.agent_p3_child_signal_cuda_20261028_v2 run --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_child_signal_cuda_20261028_v2_prepare_MAIN01.json --sha256 ACTUAL_V2_PREPARE_SHA --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_child_signal_cuda_20261028_v2_cuda_MAIN01 --authorization MAIN_APPROVED_HOST_CHILD_CUDA
/home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python -B -m experiments.spectral_reachability.agent_p3_child_signal_cuda_20261028_v2 consume --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_child_signal_cuda_20261028_v2_cuda_MAIN01 --sha256 ACTUAL_V2_REPORT_SHA --receipt /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_child_signal_cuda_20261028_v2_consume_MAIN01.json
```

output/receipt 排他创建，拒绝重入/覆盖。run 仅 compact resources+report SHA；consume 带 receipt 才避免终端打印全部队列。worker 未运行真实 prepare/run/consume；RAM 测试验证编码与 ABI，不证明 GPU、AD 数学正确、控制收益或 C3 闭合。child program/reconfiguration compiler 与物理收益仍另待实现/验收。
