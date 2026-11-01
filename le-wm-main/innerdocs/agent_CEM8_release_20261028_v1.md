# Fixed8 CEM saved release：持久严格回执

独占新 source/tests/innerdoc；旧 producer、reference wrapper 和任何 shared docs 均未修改。MAIN 审核和实际运行，本 worker 不创建真实 release、不执行 prepare/run/neural/GPU/physics，无子代。按 experiment-code 与测试先行技能实现；用户三文件/RAM边界优先，不生成额外图或实验。

## ABI 与真实来源

默认且唯一 CLI provider 是实际 `agent_moving_CEM8_20261028_v1.consume(directory, report_sha)`，不是旧 observed4 consumer，不把 prepare/测试当 CUDA。调用前固定 producer source SHA `6dcb425c13ea8cade14bf23c97ecfe87b004b7804d155c5ef06282045e99c251`；禁止替代 source。测试的 consumer/io 注入仅存在于 Python 内部测试入口，CLI 没有跳过验证或替代 provider 开关。

MAIN 已报告真实 CUDA 与 ownconsume 完成。worker 获授权后只读核对了这些文件的 SHA/字段，没有再执行 consume：

- report：`results/agent_moving_CEM8_20261028_v1_cuda_MAIN01/report.json`，SHA `660751b5784d25cce652d0ecb06ffb4cf3810a24ed135755606c97c3f957da0a`；status `teacher8_CEM_commands_sealed_not_admitted`，3650 artifacts，predict/AE/pred_proj 各 attempted=completed=520。
- actual prepare：`results/agent_moving_CEM8_20261028_v1_prepare_MAIN01.json`，SHA `3b7dc90bbd1628e59b41426c16dcb87037e6f981720a68c330a47895bccd648a`；8 branches，118 direct bindings。
- MAIN 告知 ownconsume exit0，errors=[]，`teacher8_CEM_commands_verified_not_admitted`，plans=8。此消息本身不是本 wrapper 的已落盘回执；本 worker 没有制造替代 receipt。

## 完整绑定与持久化

调用者必须外部提供实际 report/prepare 的绝对路径与 SHA，二者严格等于 report/binding.json。先原子 mkdir NEW 输出目录，request.json fsync 后，校验完整 hash inventory，再调用真实 producer.consume，重新逐文件 hash 验证无漂移，最后 receipt.json fsync 和读取 SHA/census。两个进程竞争同一路径时只有一次 mkdir 成功；重入在重新 consume 前拒绝。错误也保留占用，不自动删除、重试或覆盖；要重新运行必须由 MAIN 选择新目录。

inventory 完整列入：producer 源码、实际 own prepare/report、全部 own prepare.bindings、固定 inherited factorial prepare 及其完整 bindings、全部 producer artifact_sha256 文件、wrapper 自己三文件。固定 inherited prepare SHA `330412c675844d7edff23ade45012cf4ef4920f64bd47a31a0c49e6872f673f5`。各表按真实绝对路径合并；同 path 不同 hash 直接 failclosed，不过滤或截断。原始 prepare/report/tensors/weights 不复制，文件 hash 流式读取；不递归扫描其他目录。producer 原输出须严格等于 report+列出的 artifacts，不能在其中写 receipt 或日志。

真正 consume 仍完整执行自己的 proposal/candidate/原模块数组/raw chronology/selection-before-replay/520完整计数/模型与输入闭包检查。wrapper 不复制弱化版 consumer，不调用 producer.prepare/run，也不手工拼8 plans。其**完整原始返回字典**嵌入 `producer_result`，包括所有 flags、errors、plans、model_Q、selection/hash、未来附加检查字段，不仅摘取 status/Q。另附 deterministic JSON SHA，不改 producer 计划的自身 semantic hash。

receipt schema `agent_CEM8_release_20261028_v1.receipt`，status `strict_CEM8_release_persisted_not_admitted`；包含 invocation、request SHA、producer source SHA、完整 input_sha256、输入数和完整 producer_result。result 必须为固定8 teacher anchors、同 report SHA、errors=[]、P1/P3/equal_total_compute=False、新 neural/physics=0。0 指这次 saved wrapper/consume 没有新神经/物理，**不抹去已发生的520模型调用**。

成功目录仅 `request.json`、`receipt.json`。失败时尽力保存有限 `error.json` 并原样抛出异常；不产生伪成功 receipt。若最终写入/磁盘/fsync/readback失败，可能保留 partial 文件；含 error 或非完整 census 的目录不可 verify。没有删除或原子回滚承诺。消费者只应在 CLI exit0 后，以独立 SHA 验证 receipt，不能仅凭文件存在判断通过。

`verify` 不写文件：校验外部 receipt SHA/request/census、全部当前输入 hash，重新调用真实 producer.consume，完整对比返回与重建 envelope，再次检查闭包和 receipt 未变。不把重新计算 digest 后伪造的新调用计数当有效 envelope。它不是只读一个成功字符串。

## 工程上限

整个新目录≤4MiB，正常 receipt+request 留16KiB error reserve；不足则失败，绝不删减 result/输入表。写前至少200MiB free+所写字节，claim 前保留200MiB+4MiB；文件/父目录均 fsync，所有文件 xb。输入 own prepare≤2MiB、parent prepare≤8MiB、report≤32MiB、hash inventory≤20000项。拒绝 symlink/别名/路径穿越；不解析任意 URL，不联网。

真实 own consume 会做保存数组/CEM proposal/父 strict SVD 的 CPU算术和文件哈希，这些不是 CPU神经实验，但有真实时间/I/O费用；不称“零总算力”。共享历史成本和 teacher8 subset 不相加重复计费。若上游任一 frozen 文件后续改动，verify 会拒绝；旧回执不会因此自动升级。

## MAIN 精确 CLI（尚未由 worker 执行）

下面报告/prepare SHA 均为本次真实值。输出是**全新目录**，必须在 producer 目录外；若 MAIN01 已存在，选择新后缀，不复用失败目录：

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
.venv/bin/python -B -m experiments.spectral_reachability.agent_CEM8_release_20261028_v1 release \
  --input /home/gbk/code/LeWorldModel/le-wm-main/results/agent_moving_CEM8_20261028_v1_cuda_MAIN01 \
  --report-sha 660751b5784d25cce652d0ecb06ffb4cf3810a24ed135755606c97c3f957da0a \
  --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_moving_CEM8_20261028_v1_prepare_MAIN01.json \
  --prepare-sha 3b7dc90bbd1628e59b41426c16dcb87037e6f981720a68c330a47895bccd648a \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_CEM8_release_20261028_v1_MAIN01
```

CLI 打印 receipt.json 的实际 SHA。MAIN 独立复算并填入 RECEIPT_SHA，不能预造未来 receipt SHA：

```bash
sha256sum /home/gbk/code/LeWorldModel/le-wm-main/results/agent_CEM8_release_20261028_v1_MAIN01/receipt.json
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
.venv/bin/python -B -m experiments.spectral_reachability.agent_CEM8_release_20261028_v1 verify \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_CEM8_release_20261028_v1_MAIN01 \
  --receipt-sha "${RECEIPT_SHA:?MAIN independently recomputed actual receipt SHA256}"
```

## 测试与限制

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_CEM8_release_20261028_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.agent_CEM8_release_20261028_v1 --help
sha256sum experiments/spectral_reachability/agent_CEM8_release_20261028_v1.py tests/test_agent_CEM8_release_20261028_v1.py innerdocs/agent_CEM8_release_20261028_v1.md
```

36 RAM tests：完整原结果保留、direct+inherited+artifact hash、调用顺序、并发式递归重入拒绝、失败占用、source/prepare/report/model/artifact篡改与执行中漂移、错误状态/分母/condition/flags拒绝、大小/floor、receipt/request/envelope重新封SHA篡改、真实 LocalIO 的 mkdir/xb/fsync RAM替身、help和真实producer.consume首个无效SHA ABI门。没有实际模型或模拟器，没有真实 release/verify invocation；注入 producer result 只测协议，不能充当CUDA证据。

已知两根8 anchors 的校准不是独立新根确认、官方CEM-MPC复现或P1/P3准入。model_Q 不等于实际任务收益；不创造训练normalizer来源、完整历史总成本、freshness、物理root等价或官方export，不清除既有阻断。真实持久回执仍待 MAIN 审核后执行此 CLI。
