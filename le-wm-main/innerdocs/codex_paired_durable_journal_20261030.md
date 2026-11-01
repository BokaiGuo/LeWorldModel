# Paired 接口 durable journal 与 saved consumer

本版本独占新增 journal、测试与本说明，绑定先前 RAM 接口的 SHA；不更改旧接口或
封存。journal 不 import 模型/环境、不提供真实 dispatch。Full D2P/SRA 的实际适配、
共同任务来源和科学前置仍未完成。

## 保存合同

`Journal(new_path, config)` 先按旧接口验证配置，再独占创建 JSONL，fsync 父目录及
header。每个 `call(op, ...)` 依次写并 fsync `begin`、接口需要的 selection 或
step intent `seal`、最终 `commit`；只有 commit fsync 成功才把返回值交给调用方。
每条记录有序号、上条摘要和自身内容摘要；超过 16 MiB 或任何事务异常即 poison，
不能继续或重试。硬中断、写失败或未结束事务没有完整 footer，consumer 拒绝；
保留原文件供审查，不能通过重启同路径重复 dispatch。`finish()` 要求两臂均为
complete/stopped，持久化固定分母 snapshot 并返回整个 journal 的字节 SHA。

消费 `consume(path, expected_sha256)` 必须由外部提供最终字节摘要，逐行拒绝
重复键、坏链、未知事件、源版本漂移、缺 footer、未完成事务及多余 seal；从
header 配置重建旧接口，逐事务重算返回值、选择/intent 确认、snapshot 和 footer。
终止后的 pending 资源或 step 保留 attempted、completed 下界和 null 未到达检查点；
重建本身不执行模型或物理。两个臂的预算各自保存，不将上限或调用数等同总成本。

CLI 只消费完成 journal，另写不可覆盖结果：

```bash
.venv/bin/python -B experiments/spectral_reachability/codex_paired_durable_journal_20261030.py \
  --journal NEW_COMPLETED_JOURNAL --sha256 EXTERNALLY_RECORDED_BYTE_SHA \
  --output NEW_CONSUMER_RESULT.json
```

该路径示例不是现有生产产物。consumer 输出同样独占、fsync，校验失败不创建输出。
不要用一个未可信固定的自算摘要把重新编造的整个链当执行真实性证明；hash chain
与重建只证明给定字节和状态机的一致性，不证明 caller 所报身份、guard、传感器、
GPU、物理或资源实际发生。固定旧接口源码与本 journal 源码，不核验其所有上游
transitive dependencies。保留 `actual_execution_verified=false`、
`dispatch_authorized=false` 和未知 FLOPs。

`Journal` 是单进程非并发原型，没有跨进程互斥、重启恢复、生产授权或环境驱动；
外部如果绕过包装器直接修改 RAM 状态，不受保护。真实执行还需可信 identity/source
closure、操作级真实计量及失败语义、稳定的存储配额/恢复策略、真实 journal 独立
消费、原生 guard 和审查授权。文件系统 fsync 的设备/断电保证取决于底层系统；
不能把摘要 ack 当外部硬件 attestation。

## 回归

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -m pytest -q -p no:cacheprovider \
  tests/test_codex_paired_durable_journal_20261030.py \
  tests/test_codex_paired_physical_interface_20261030.py \
  tests/test_codex_full_d2p_physical_preflight_20261030.py
```

临时 fixture 用于完成/早停、pending、I/O 故障、重封篡改和独占写入测试，不是
真实 physical 或神经执行结果。本任务不启动 GPU/物理、不训练，不声称等成本。
