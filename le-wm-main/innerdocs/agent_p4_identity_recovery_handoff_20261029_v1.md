# P4来源blocker：分类后的最小可执行交接

本页只补操作交接，不重复生成manifest/null草案。完整字段分类及原report SHA/pointer见已冻结的 `agent_p4_prerequisite_recoverability_20261029_v1.md`（SHA `c115eb9170e265e49b43c8eb73d9b326cd6b4acf5eb7e9663ac1befd17178385`）。所有既有源码/结果/gate不改；无NN、物理、网络或自授权。采用documentation-and-adrs技能，记录证据路线，不新增审批层。

## 一、已经完成的恢复，不重复做

直接复用 `results/agent_p4_historical_source_identity_20261029_v1_LOCAL01.json`，SHA `577580e536598d850cd449fda2a02c314197267bdc41bff9d85692432165e724`：`/sources`保留8来源/16引用/52行身份，`/dispatch_load_closure`列十次调用，`/byte_verified_source_candidates`列21份候选字节。原字段pointer在各行`fields`内，不能拿本页或派生文件名充作原episode。

|分类|已经确定的边界|下一动作|
|---|---|---|
|原件可直接恢复|报告path/SHA、声明root/history/goal hash、config、命令及部分进程状态已恢复|不重扫；MAIN可完整verify既有报告|
|源码可导出而非现场观测|零history数组hash、reset/collector公式及有限循环候选|需要原加载依据才可将代码族包络用于历史执行；不把源码推导写成actual attempt|
|指定原件未记录|A–C原`/calibration_roots/i`、D–H原`/rows/i`无state7/snapshot/episode/attempt映射；全部52行key集合已核|停止对同一report/hash反复投影；定向问是否另存原数组/原日志。没有就记“该原件不可追补”，不重新模拟补写旧证据|
|其他资产是否存在未知|原加载manifest、完整reset账本、原env/wrappers版本、相关cutoff后delta|由原来源方给明确原文件/pointer/SHA或确认未记录/遗失。未找到不等于从未记录|

原硬前置来自外层 `docs/D2P_MOVING_FRESH_ROOT_PROTOCOL_20261027.md` §2–3及预注册 `/exclude`、`/cutoff`、`/approval`：固定候选、不按outcome选根、有界相关历史排除及外部MAIN范围批准、witness前实际身份排重。`source_episode.episode_id`字面字段和`panel_scope:*` token是后续工具的接收合同，不是原协议新增科学必要条件；但旧工具因此blocked的输出仍原封保留，不在本页override。

“未保存完整访问史”可作为披露限制；不能声称“从未被任何人打开”。“无法排除相关旧物理根”仍是原fresh合同的真实缺口，不能只靠披露消除。相关项目/任务截至cutoff的覆盖不等于无限机器/无限时间历史证明。本文不为工程运行另加无限前置，也不降低fresh身份要求。

## 二、MAIN现在可执行的只读入口

在 `/home/gbk/code/LeWorldModel/le-wm-main`，直接运行已有接口，输出stdout，不写新草案：

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_freshroot_authority_20261028_v1 request
```

本轮已实际调用同一`requests`+SHA绑定Reader：8 sources、16 inherited refs、10 known dispatches。request的compact canonical语义SHA为 `9b304bc853f678046cae8dabd7adeda1866ada6cbe8933b7cd6539b16ff699a0`，不含CLI末尾换行，不能与stdout文件byte SHA混用。原catalog SHA为 `00a3fffeaaa8943e851d129b941a7f07feaa1db387117beb98110e4ec95f5eeb`。

原来源方只需按该request中的十个call_id交接原证据；B有三次，不能只交最终一次。每次走以下任一可替代路线，不要求两套同时具备：

1. 原完整reset ledger（含失败/拒绝/未选/重启及原来源依据）；或
2. 原运行加载依据及对应producer/P0/factory/env/wrappers/shell字节，加可检验的有限尝试包络。现有postrun/downstream seal/Git候选只可用来匹配附件，不可倒签。

另存原root/observed身份附件与上述attempt覆盖是不同问题：账本计数闭合并不自动恢复state/snapshot/RGB3，也不证明新四槽已排重。若原资产有可认证的另一episode/lineage表达，应先交MAIN判断表达能力；不得为了适配字段名合成`ep0`或拿selected ordinal代替episode。

收到真实response及附件以后，已有接收CLI如下；其中`REAL_RESPONSE_PATH`、`REAL_RESPONSE_SHA256`须由MAIN设为**实际收到**的canonical绝对路径和byte SHA。本轮无此实际response，故未执行validate，不生成模板伪装响应：

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_freshroot_authority_20261028_v1 validate --response "${REAL_RESPONSE_PATH:?需真实原来源响应路径}" --response-sha256 "${REAL_RESPONSE_SHA256:?需实际响应byteSHA}"
```

该命令检查schema、request/run/report/附件/pointer绑定；即使exit0仍固定 `origin_review_required=true`、`reset_blockers_cleared=0`、`P1_admitted=false`，不是原来源真实性证明或MAIN批准。下一步由MAIN审原证据真实性、有限包络及有界覆盖；worker不签批准、不自行启动fresh source。

若原来源方明确未记录/遗失，交付具体call/字段及适用范围说明即可，停止追加空wrapper。原fresh确认是否仍能依其他既存证据闭合由MAIN审定；若不能，保留unknown及原门槛，不以新脚本、seed不冲突、静态代码或新重放充当fresh证据。

## 三、本轮验证与未运行项

已实际复核request接口的8/16/10分母及语义SHA；复用原authority和历史identity的RAM回归。没有新增测试实现、原来源response、批准、实际root、模型或物理结果。准确回归命令：

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_freshroot_authority_20261028_v1.py tests/test_agent_p4_historical_source_identity_20261029_v1.py
```
