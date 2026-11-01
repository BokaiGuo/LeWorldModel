# P2 strong-baseline exact saved-control evidence resolver v1

只做保存证据投影；不是 matched 认证、不是代码审计、不是实验。三个输入严格取 frozen `agent_p2_strong_baseline_saved_summary_20261028_v3.ASSETS` 的路径与外部 SHA。输出科学准入始终 false，composition_benefit=null；不改 v3，不启动模型、模拟、网络或新根。

## 真实路径闭包和输出

kinodynamic: report 的 prepared + prepare_sha256 → prepare.slots 的 task_export + task_sha256 → 原 moving task。按 source_slot join，再精确核 task_id/blind_sha256，核 task_contract_sha256 和原合同 JSON 默认空格序列化 digest。独立读取显式 inherited_encoding_report/SHA 的 model hash receipts。

no_composition: 固定 consume.rows 每行 task_export/task_sha256 → 同样的 task 身份闭包。不得按 task_00 文件名推断 root。sampling: report 四个 root/arm 保留，但没有带路径的 prepare/task 入口；单独 prepare_sha256 不构成可遍历路径，拒绝扫描目录补配。

每行输出 task/checkpoint/action/observation/metric/horizon/goal/source 的逐组件 value + 原文件 path/SHA/JSON pointer + 显式边 chain。目标保存原192维数组，不仅名字或范数。action 保存球半径、维度、packet length 与完整 normalization（包括 training_provenance_verified=false）；observation 保存 representation、latent维度、实际观测步数/像素 SHA、history 和 with_target。来源 SHA 是声明身份，不升级全局 freshness。模型权重/配置 digest 与 tensor hash receipts 分栏；不把声明变成原始 loaded bytes 的独立证明。不加载大模型资产。

聚合 controls 只有在全部原行全部所需组件闭合时才非 null。固定分母2/2/4，不删除失败或缺证行。root_ids 为源局部标识，不代表跨来源可交换；每行另附原 root/arm pointer。observation 还保留完整 root_context 和真实 raw action history，action 同时保留 task action_mean/std，不仅引用 normalizer 名字。保留原 native resource receipts，不将 actual counts 擅自转为 v3 的 per_root_per_arm_including_signal_search_replay limit/inherited 预算。完整 budget 尚缺；parameters 也未代为编造。因此不能直接交完整 v3 prepare 请求。

## v3 evidence 交接

v3 仅支持普通 JSON。resolver 将 gzip task 中的所选证据无损投影到普通输出 JSON；原 gzip 外部 SHA、pointer、引用边全保留。CLI stdout 的 v3_evidence 提供精确 `{path,sha256,pointer}` 三键，指向 `/assets/<name>/controls/<field>`；v3.evidence_value 可直接读取这些值。先运行 resolver verify 闭合原始链，再消费投影。此机械可读性不证明字段语义可公平比较，不能用本文件的 evidence 替代缺失 budget 或加载证明。不同源 observation/metric/support 不自动相等。

## CLI（inner repo cwd）

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p2_control_evidence_resolver_20261028_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_control_evidence_resolver_20261028_v1 resolve --output results/agent_p2_control_evidence_resolver_20261028_v1_MAIN01.json
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_control_evidence_resolver_20261028_v1 verify --output results/agent_p2_control_evidence_resolver_20261028_v1_MAIN01.json --sha ACTUAL_OUTPUT_SHA
```

输出排他创建，不覆盖；verify 以外部输出 SHA + 全量重建逐值相等检查。单元测试 RAM mutations，加实际三资产 saved join；无神经/物理。文件与解压 metadata 各8MiB上限；只沿白名单链接，不递归全盘扫描。安全边界不声称防恶意文件系统并发替换。

## 最小剩余证据

1. sampling 原 report 的 prepare_sha256 对应的原始文件**明确路径与原引用凭据**；随后须有原 task/sourcepack 路径+SHA，而不是依据目录名猜身份。当前 source history 文字为 causal padding，不能与 moving observed history 偷换。
2. 三方法所有预算轴的原 scoped limit/inherited 声明（v3 RESOURCE_AXES），包括缺失轴的明确零声明与 scope，而非本工具填零。
3. 如要求原运行实际加载闭包，需原 loader 对 loaded checkpoint/config bytes 的可核验绑定；已有 model hash before/after 只按原 receipts 保存。

本工具已经恢复的2+2行 task 字段无需再次生成/编码。sampling缺项不改变四臂分母；P1/P2依旧关闭。
