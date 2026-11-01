# 未来独立 holdout：采集设计与失败封闭合同

状态：**拟议／BLOCKED**。仅适用于未来新建采集事件，不接触 P4 保留槽，不认证
既有根，也不改变原 P1/P4 freshness 规则。没有已选保管者、真实信标轮次、已签名
承诺或新数据。本次只实现 metadata 检查，不调用 GPU、模型、模拟器或物理。

## 要证明的对象

目标是可审计的新采集事件和新来源命名空间，而不是在旧 reset ledger 不完整时
声称与所有历史物理状态严格不重叠。前者需要独立保管人、采集前承诺、不可预测
外部随机性、运行时原始日志和 no-copy 证据。后者仍需可证明完整的历史状态范围，
或另行审查的严格不交域证明；换 seed、UUID 或环境实例均不足以证明。
训练未见、事件独立、抽样独立、物理状态互斥分别记录，任何一项都不由其他项推出。
如果训练血缘未知，即使未来采集可证明发生于 checkpoint 生成之后，仍需证明
新采集没有复制旧数据且冻结模型在采集后没有变化。

## 未来采集流程（须另行批准）

1. 在任何新采集、信标结果或根状态可见之前，第三方持有并时间戳封存协议 SHA、
   本检查代码 SHA、冻结 checkpoint／normalizer、原生环境与 wrapper／renderer／
   physics 及依赖闭包、完整 P1 输入/动作/goal ABI、固定资源预算、预先选定的
   已知 seed 禁用清单及其范围说明。清单必须涵盖相关历史及保留 seed／半开区间，
   不得为使某个生成 seed 通过而事后改写；来源责任人单独提供和裁定，不从本工具
   的清单输入推断历史已穷尽。模型、实验和环境 SHA 都只绑定字节承诺，
   不自行证明来源真实性。
2. 在承诺中固定独立公开信标提供方、唯一未来轮次及发表时间窗口。承诺须先于
   该轮次；信标不可用、超时、签名或外部可信时间戳无法核实时整批阻断，
   不更换轮次、不挑值、不重抽。
3. 试点固定四个位置 `0..3`，每位置一个 native reset，无替补。
   `seed_i = int.from_bytes(HMAC-SHA256(key=R, msg=b"p1-future-holdout-v1/root/" ||
   uint32_be(i))[:4], "big")`，其中 R 是唯一已认证信标值的 32 原始字节。
   任何 batch 内 seed 重复或与批准的已知 seed／半开区间冲突，整批阻断，不换
   单个 seed、不重取信标。四是工程试点分母，不是统计功效证明。
4. 经正式权限批准后，独立保管者在指定的采集窗口内建立新实例，四个槽分别创建
   新环境、只按对应 seed reset 一次；不得从旧数据拷贝 state、goal、RGB、动作、
   物理接触缓存或重放结果。固定写入 attempt-start、reset-return 或失败、完整
   episode/source/instance 身份、原 state7／可见 body snapshot／RGB3／raw action
   history 的原字节 SHA 和有序不可回填日志。每次失败及未启动槽保留原分母；
   禁止看成败、难度、模型响应或 outcome 后替换或扩容。
5. 若采集要保持同一 P1 任务，逐项核验已封存原生环境、goal7、时间、action、
   渲染、预处理及 model/normalizer ABI；不同语义须另立迁移实验。先封存所有
   预测再开放本批标签；保管与访问证据支持审计，但不证明心理盲态。后续分支
   必须重新核对起点和可见状态，不把 state7／snapshot 当成完整 solver 隐状态。
   任何所需权限或证据缺失均不 dispatch。

## 机器合同

代码 `le-wm-main/experiments/spectral_reachability/codex_future_holdout_isolation_20261030.py`，
SHA256 `b1ce0b1dccb3f8b8023cfe00f202c9d1a13e90ac13db0b2a71b8c9a1a5dad735`。
只能读取 `results/codex_future_holdout_input_*.json` 显式绝对路径；输入文件 bytes
须匹配 CLI `--input-sha256`，上限 4 MiB，拒绝重复键、非有限数和非白名单字段。
不自动跟随任何路径，也不搜索历史或保留槽。输出只到 stdout，不写采集资产。

输入顶层精确为 `protocol / exclusions / receipt`：

- `protocol` 精确字段为 `schema, batch_uuid, source_uuid, experiment_sha256,
  contract_code_sha256, environment_sha256, checkpoint_sha256, normalizer_sha256,
  exclusion_sha256, beacon_provider, beacon_round, beacon_scheduled_utc, not_before_utc,
  deadline_utc, slot_count`。schema 为
  `codex_future_holdout_isolation_20261030.protocol.v1`；四槽固定，时间须为秒精度
  `...Z`，代码 SHA 必须等于当前脚本。`experiment_sha256` 要绑定完整冻结实验
  计划及其 ABI／预算／失败规则，不是任意文字替代审查。
- `exclusions` 精确字段为 `schema, scope_sha256, seeds, spans`，schema 为
  `codex_future_holdout_isolation_20261030.known_exclusions.v1`；seed 为排序去重的
  uint32，span 为排序、不重叠的 `[start,end)`，允许 end=`2^32`。protocol 的
  `exclusion_sha256` 是该对象按 `json.dumps(sort_keys=True,separators=(',',':'),
  allow_nan=False).encode()` 得到的 SHA256。`scope_sha256` 只声明对应范围证据，
  程序不读取或认证该证据。
- `prepare` 的 `receipt` 必须为 null，且实际本机 UTC 时钟须早于固定信标计划时间；
  输出协议 semantic SHA，**始终 blocked**。需要真正独立的第三方将完整协议及此
  SHA 在信标前封存，不是 stdout 自行构成可信时间戳。
- `review` 的 receipt 字段严格为 `schema, protocol_sha256, commit_utc,
  commit_attestation_sha256, beacon_provider, beacon_round, beacon_value_hex,
  beacon_published_utc, beacon_receipt_sha256, source_uuid, batch_uuid, instance_uuid,
  instance_created_utc, custody_attestation_sha256, no_copy_attestation_sha256`；
  schema 为 `codex_future_holdout_isolation_20261030.receipt.v1`。检查协议和 source/
  batch 绑定、instance 不同、承诺 < 计划信标 <= 发表 < 实例创建、创建在协议窗口
  内且不晚于本机当前 UTC、固定四个 seed 与已知排除项。receipt 所有时间／哈希
  都是未证实声明；校验不了签名、不读取证据，时钟也不是可信时间戳。

review 若结构和算术通过会列出四个种子，但状态仍为
`consistent_declarations_only`，各槽为 `collection_unobserved`，实际采集权、
freshness、P1 全部为 false；外部签名、保管者、历史范围、ABI 和审查依然阻断。
不得使用该输出作为模拟 reset 或预留槽访问授权。没有 receipt 示例的真实值，
也未生成现实协议或信标。

## 验证

固定向量、四槽分母、重名 UUID、commit/beacon/instance 时间与轮次、未来收据、
SHA 篡改、非 hex、额外 outcome 字段、batch seed 冲突、已知点／区间冲突、
半开边界及伪完整声明不发放 authority：17 项测试通过。首次测试中的 HMAC
预期向量写错，独立重算后修正了测试；没有改协议来迎合结果。测试仅是合成
元数据，不是来源、随机性或新根证据。

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_codex_future_holdout_isolation_20261030.py
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -m experiments.spectral_reachability.codex_future_holdout_isolation_20261030 --help
```
