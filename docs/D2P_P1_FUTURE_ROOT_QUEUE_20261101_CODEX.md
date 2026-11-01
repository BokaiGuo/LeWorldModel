# P1 未来开发隔离与确认队列：前置合同

本文件冻结的是**队列构造规则**，不是选定的确认样本，也不是执行许可。
没有提供预注册的有限队列规模、seed、producer/environment/模型与实验
protocol 字节身份及独立保管人，因此本次不擅自选择根或生成可执行确认队列。
本任务不接触 Hypatia 的 witness/payload；不执行 reset、神经模型或物理实验。

## 历史事实和限制

本次重跑已有只读验收器，四组明确固定的 sourcepack/journal 合计 44 条
reset 记录，另有 overlay 12 条，固定输入未出现读取失败；输出
`results/p1_future_root_history_inventory_20261101_codex_LOCAL01.json`，
SHA256 `70ab1a9b0a26ec16c9851c377cd275de4b4ada91c40ad6543e5382bd429ab8e4`。
它只覆盖这五组输入，不是全项目盘点。已保存 journal 可恢复部分尝试、
seed 和部分 state/goal，但不自动包含未保存失败、并发、重启和旧来源的
所有尝试；完整状态/速度/碰撞求解缓存及跨来源物理同一性也未完整认证。
seed 0 可表示显式 state 重建，不能当随机物理根身份。

另一个已封存 quarantine 将 A–H 的八来源、16 引用及条件种子包络保留，
其原运行与执行源码的绑定及实际尝试穷尽均未认证。历史全覆盖 unknown；
旧 checkpoint 到实际训练 run、不可变训练数据／split 和 normalizer 的链条
也未闭合，因此训练成员与训练外均为 unknown。源码哈希、非重合 seed、
状态哈希、签字或外部时间各自都不能单独弥补这些缺口。

## 队列输入与隔离

`p1_future_root_queue_20261101_codex.py freeze` 只接受严格 JSON：schema
`p1_future_root_queue_20261101_codex.spec.v1`，实验 ID，实际
`producer_sha256`、`environment_sha256`、`protocol_sha256`、`model_sha256`，
以及有序 `slots`。每槽仅有 `slot_id`、`role`（`development` 或
`confirmation`）和全部有序 `reset_seeds`。必须两种角色均存在、所有
slot ID 与 seed 全局不重复；拒绝额外 outcome 字段和非有限值。
工具拒绝命中上面有界台账实际 seed、seed 0 标签或旧条件包络的 seed。
这是保守隔离，不意味着未命中便已排除未知历史；显式 state reset 及非
seed 来源尚不在此接口范围内。

输入哈希是待外部独立核对的**声明**，工具不验证该 producer 真能捕获全部
reset，不验证模型或环境字节，不执行队列。输出 `xb` 独占创建并 fsync，
包含 spec 摘要与两个有界历史证据摘要，`execution_authorized=false`、
`P1_admission=false`、历史覆盖/训练成员/freshness 均 unknown。文件哈希
不证明冻结先于实验；需要可信独立时间戳或外部登记，以及独立保管的原件。
确认种子在队列文件明文中：必须由独立保管人进行文件级访问控制、冻结
协议和顺序，并保存不可被开发工作读取的确认任务。此工具本身不提供盲化、
访问控制、外部签名、数据治理或跨进程实验隔离。

## 前瞻账本

`append` 对同一个 ledger 用 flock 串行写入并 fsync；每事件包含 index、
前一事件哈希、冻结队列内容摘要和自哈希。每个 slot 只能按预定 seed 次序
记录 `reset_attempt`，然后相应 `reset_return` 或 `reset_failed`；所有失败
保留，不允许重试冒充原尝试。return 声明 state/goal/history/snapshot 的
SHA256；跨角色 state hash 精确相同时记录并报冲突，不丢弃回执；不同 hash 不证明物理状态、
位置容差或训练分布独立。异常结束的 pending 会在 audit 中保留。

调用方必须将 append 与实际环境调用及原始记录进行不可绕过的对接；当前
工具不截获环境 reset，也不能证明日志没有遗漏、哈希对应真实状态、
签发时间真实或 confirmation 保密。`all_attempts_accounted` 只覆盖冻结
表内尝试，不代表历史穷尽或资格。未来如果源头失败或域删失，不能删槽、
改 seed、补选或只报告成功；预先冻结所需的新尝试和处置规则。

例如在 `le-wm-main`，由负责人提供并单独封存真实 spec 后：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_future_root_queue_20261101_codex freeze --spec /absolute/path/to/reviewed-spec.json --output /absolute/path/to/new-frozen-queue.json
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_future_root_queue_20261101_codex append --queue /absolute/path/to/new-frozen-queue.json --ledger /absolute/path/to/new-ledger.jsonl --event /absolute/path/to/new-event.json
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_future_root_queue_20261101_codex audit --queue /absolute/path/to/new-frozen-queue.json --ledger /absolute/path/to/new-ledger.jsonl --output /absolute/path/to/new-audit.json
```

只有独立来源责任人完成历史执行与全尝试 authority、确认数据保管和
权限隔离、真实 producer 捕获对接及 state/goal/history 去重、必要时训练方
提供 checkpoint 对应数据和 normalizer 证据之后，才能另行评估哪一种
独立性主张可用；不修改现有 P1 gate。
