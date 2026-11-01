# Fixed four-update full-H2 line-search GN baseline

新独占模块，不修改 B8、FD、fixed-segment diagnostic 等封存源或 actual 输出。
这是更充分付费的 GN 基线修复，非 P1 准入、stationarity/theory 证明或物理改善。
固定完整原两根，不新增独立样本；不根据两根实际 .25 结果只保留 .25。

## 冻结算法

直接重用 B8 两个 observed task 的完整 native192 context、normalizer、独立 goal、原 Q、
模型身份与 packet logit cap，从 zero restart 独立开始（不是从 iter8 延续，也不复制旧 endpoint）。
每根先独立 nominal/whole/segmented fullH2，再固定四轮：

1. 当前 incumbent 重算完整 H2 logit J192x20，保留两个 ADforward packet 的上下文、全部192
   raw-control reverse derivative 行及 tanh chain 后 J。AD graph 联通两个 packet，不 detach prefix。
2. 原 Q 残差 `r=Q(z-goal)`、`A=QJ`，原 ridge 公式对 `A/sqrt(192)` 做 SVD，
   damping=.002。完整 GN endpoint 用原 screen.projected_logits 投影到原逐 packet logit 球。
3. 所有尺度 `[1,.5,.25,.125,.0625]` 均评估一次 nonlinear fullH2。
   float64 `x+s*(endpoint-x)`、一次 float32 rounding、NumPy float32 tanh 后 native10x2。
   中间尺度如因舍入越界直接 partial，不二次投影/clip、不换尺度。完整请求在 forward 前 durable。
4. 原 incumbent 和全部五候选同原 Q 评分后稳定取最小，平局 incumbent 优先、然后尺度顺序。
   无 Armijo/新阈值，无遇改善即停；每轮即使未改善或动作重复也支付全部候选。
5. 所选动作独立 nominal/whole/segmented 三次 fullH2，验证完整内容、history/layout/chronology、
   原 Q cost exact 一致，作为下一轮 incumbent。第四轮后仍付全重放，不自动延长更新。

调用 screen.NativeCuda：只有 MAIN explicit run 才加载冻结模型，CUDA:0、MATH SDPA、TF32 matmul
关闭，无 CPU neural fallback、新 image/projector、physics、训练或新 Q。
任务沿用原 task_id（科学输入不改）；新执行协议另由 prepare SHA 封存。
prepare 通过旧 diagnostic 的 saved.material 对原 B8 task/arm/raw gzip journal 逐项重建绑定，
不读取 diagnostic CUDA 的尺度收益、MAIN stagnation/selfchecker 结论作新候选选择。

## 全账与保存

每根 initial 三路 fullH2=6；每轮 ADforward2 +5候选×2 +三路重放6=18；
`6+4*18=78` predict 和78 actionencoder。两根总156/156、8Jac、1536数学 VJP directions、
8 SVD、8 linear_solve、8 proposal、8 selection。AD forward 已包含在156中，不能再重复加算；
反向 VJP 单列，不能称总 FLOPs/时间与 B8 或旧 fixed-step 等预算。
沿用 task/goal/Q 的旧编码与 source acquisition 属 inherited；prepare 保留旧 direct/image 账，
不是本次新执行成本，不虚称全 pipeline compute equivalence。

输出 `binding.json`、`events.jsonl.gz`、两个 `root_i.json.gz`、`report.json`。
全部原始事件无字段/精度删减，逐事件 canonical JSON 后 gzip level6 + Z_SYNC_FLUSH + flush/fsync。
完整J、raw行、每次 forward 上下文、评估请求/完成、五尺度 logits/成本/选优、逐轮 verified
prediction 均保留。事件带 root 身份、evaluation_id；最终 root 对象还保留 final prediction 和逐轮选择。
压缩不代表计算或新增研究证据，不是以旧 replay 冒充本轮。

硬上限2MiB，包括所有文件和失败写入已占用字节，预留128KiB给 gzip footer/最终报告。
启动至少202MiB free，每次写保持200MiB floor。失败不退款/重试；先保留首错，cleanup/source/hash
错误追加。模型未load时 model_unchanged=None，不能声称复核。文件 SHA 写进最终报告，prepare/input
运行后再次验证。错误可能令 report/receipt 本身不可 durable；日志不带 EOF 或缺 return 必须标 partial，
不能从 fsync 抛错后仍可 readback 推断成功持久化。recover 返回完整 JSON 行与 EOF，不接受尾随垃圾。

## 容量和验证界限

容量测试执行完整两根四轮生产 optimize 和相同 gzip/put/close 路径，仅 I/O transport 换成 RAM。
mock 使用全维 dense 非零 root/goal、192x20 float64 raw derivative、完整 context、1536 VJP 行事件，
无神经调用。解压 journal canonical 原始字节 SHA 必须 exact，root 对象逐字段 exact；完整三路
重放、全部五尺度、当前点 fresh J 的数量与78/root账精确核验。
这只认证该全尺寸 mock 的容量，不保证任意 actual 数组可压缩比；actual 遇硬 cap 则保留 partial，
不能抬 cap/删J/降精度/少算候选后声称完成。本轮完整 dense mock 原始 journal
7,675,316 bytes，gzip journal 1,359,831 bytes，全部输出1,382,266 bytes，
距2,097,152 bytes cap 尚余714,886 bytes。没有把未压缩数组或 footer 排除在外；
解压后还独立按 raw derivative 行与当前 commands 重建全部8份 chained J，核对全尺度与稳定选优。
其余测试覆盖完整尺度 tie、history corruption、sync/VJP partial、gzip truncation/CRC、
fsync 故障不伪造 footer、quota/floor。小 I/O tests 仅几十字节至KiB文件，不写大 fixture。

## MAIN 命令

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -s -p no:cacheprovider tests/test_p1_line_search_gn_cuda_20261027_v1.py
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_line_search_gn_cuda_20261027_v1 prepare --output results/p1_line_search_gn_cuda_20261027_v1_prepare01.json
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_line_search_gn_cuda_20261027_v1 verify --prepared results/p1_line_search_gn_cuda_20261027_v1_prepare01.json --prepare-sha PREPARE_SHA
# MAIN review +容量认可后独占 HOST CUDA；实现agent不执行：
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_line_search_gn_cuda_20261027_v1 run --prepared results/p1_line_search_gn_cuda_20261027_v1_prepare01.json --prepare-sha PREPARE_SHA --output results/p1_line_search_gn_cuda_20261027_v1_cuda_main01
```
