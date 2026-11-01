# A–H原来源身份及加载闭包的有界恢复

独占源 `experiments/spectral_reachability/agent_p4_historical_source_identity_20261029_v1.py`；对应tests同stem。只做saved metadata恢复，不运行历史命令、不导入其Python、不读权重、不读outcome值，不签MAIN批准或freshness。

## 固定读取边界

以已封存 `root_exclusion_authority_resolution_20261027_v1_MAIN01.json` 的8个 `historical_reset_attempt_bound_not_attested` 请求和 `agent_freshroots_resume_20261027_coverage.json` 的A–H目录为唯一入口。两者SHA在源码固定；复核两份原prepare压缩payload中的16条pointer及原report SHA，不扫描目录、会话日志或全库。原报告固定52条selected记录（D为24，其余各4），保留全部分母。

复用旧 `root_exclusion_authority_resolution.project` lexical projector，仅解码root_id/root_sha256、history/action/goal SHA及目录已列config字段。`/protocol/outcome`只允许已有目录的预留seed/count设置，不是实测outcome；`jacobian`、`compilation`、score、status、metrics全跳过。权重路径/hash只读字符串，不打开模型。hash按原单位保存，history stack SHA不能冒充三帧各自SHA；root_id是selected ordinal，不能推导episode/reset seed。

每个身份字段保留原report绝对path、byte SHA、JSON pointer、原值。每条继承ref保留原parent prepare path/SHA、解压payload pointer、原reason；新状态仅 `source_bytes_joined_reset_history_unknown`，不是清除reset阻塞。

额外来源字节仅限已列出的21项：E postrun三源码、H下游physical prerun七输入、后续sealed program七archive、原目录四个固定Git blob。manifest自身固定SHA，snapshots限定原seal目录。静态AST记录import声明、reset/collector/fingerprint调用与identity函数文本及行号，不执行这些代码。H snapshot中的report只hash不解码；Git仅 `git show` 固定object/path，无checkout或新扫描。缺文件独立列缺项，存在但hash不符硬失败。

## 原加载闭包与缺失项

10次dispatch逐角色列producer、P0依赖（适用时）、四factory、environment、wrappers及shell（适用时）。`available_byte_candidates`只证明某份现存快照字节匹配候选；不证明原进程曾加载它。所有 `original_loaded_path/SHA=null`、`original_load_attested=false`；postrun或另一个下游prerun不能追溯改签为原运行加载。角色表是旧工具明确要求的最低闭包；AST import声明不是完整动态/transitive dependency闭包，辅助模块同样需原运行来源认证。

已封存 `agent_freshroots_resume_20261027_next_delta.json` 只验证压缩包/semantic SHA并lexically读取其十次dispatch disposition，不重新扫描日志或1033项delta。`saved_process_disposition.envelope_pointer` 指解压semantic payload，非外层envelope JSON。进程exit/失败证明与reset/加载认证分离，旧目录的process_return=false不作为新结论。

精确尚缺：每个selected-row对应的episode/reset attempt、真实state7/snapshot/raw history与逐帧identity；包括失败/拒绝/未选/重启的完整尝试ledger，或原进程完整loaded closure加穷尽envelope；原producer/P0/四factory/安装env/wrappers/shell及实际transitive模块的run-bound字节；独立完整机器/worktree/时间/未保存失败历史及后续delta权威。新脚本、预留seed区间、源码reset表达式、相同root hash都不能补这些证明。

`declared_hash_groups`仅供排除索引，按原声明hash汇总全部occurrences，不称物理唯一root个数；`physical_unique_roots=null`。`original_attempt_count=null`、全历史complete=null。16阻塞不清除；freshness/P1/scientific/run authority全部false。此报告不能代替独立确认panel批准。

## 精确命令

在 `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p4_historical_source_identity_20261029_v1.py
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p4_historical_source_identity_20261029_v1 recover --output results/agent_p4_historical_source_identity_20261029_v1_LOCAL01.json
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p4_historical_source_identity_20261029_v1 verify --report results/agent_p4_historical_source_identity_20261029_v1_LOCAL01.json --report-sha256 ACTUAL_RECOVER_STDOUT_SHA256
```

输出exclusive `xb`，不覆盖。CLI的verify SHA必须填recover实际stdout，不预造hash；已有LOCAL01时改新同stem后缀。verify重新读取相同有界资产/实现三件套并完整byte复算报告，不复跑旧工具的目录/log census。原数据通过SHA严格绑定；report4MiB、结果8MiB、底层有界read512MiB、解压delta16MiB、输出磁盘floor200MiB。RAM mutation和实际saved regression是工程校验，不是新根或科学证据。无NN、physics、GPU、网络执行项。
