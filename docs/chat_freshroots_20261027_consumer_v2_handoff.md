# 确认清单消费者v2：历史前缀父源绑定及保守排重

当前完整freshness goal仍active。本轮是实际metadata进展，不是P1模型/控制、训练或代码审计。所有原文件及上一轮结果不改，只新建本前缀文件；无子智能体、删除或推送。神经/物理调用0。

## 新证据改变了什么

上一轮80个前缀身份尚缺显式父源。本轮在读取父报告prefix值前封存 `docs/chat_freshroots_20261027_parent_scope.json`，固定两个父报告及path/SHA、白名单、全部记录匹配规则，不用outcome筛选。

从父报告只投影 `rows[*].root_sha256`、`branches[*].candidate_id` 和分支/探针的 `prefix_state` / `prefix_rgb_sha256`。成本、成功、预测、扰动后`state`、comparisons、status等均词法跳过。取所有父前缀而非selected结果；80条父记录全部保留。

按**原state7 little-endian float64字节SHA + prefix单RGB SHA**匹配原80份补充记录：80/80均有唯一祖先/分支组，0 missing、0 ambiguity；形成8个组，每组10条父证据。祖先4个root-goal为：

```text
9173218f71898716eac8fe0f3946b8e2cf20527ac71d84555c20c4e1b7fbf333
23b8de7837c39fb4e0fcffe7678732a9d840c57de7a0c8376e22076770ef6050
4afefa3be02a327aa69c22f38c17376d82340ba352aaf4aa29f95b070f5503ca
cf1b9bb913bf7815c9184d5de649b77bf09cc268dd80905a0feca9bd2a2edbf2
```

每条父证据都保存实际父报告绝对path/SHA、JSON pointer、ancestor root-goal、branch ID。**祖先分支关联不等于prefix与祖先采样state相等。** 没有用目录root_N推seed或合并物理根。

进一步对封存原8来源的全部selected root-goal投影作精确相等连接，不选一个有利来源：8个组各与E/H两来源匹配，共16连接。8份原报告字节均重新hash。E=`pusht_d2p_p1_packet_contract_v4_calibration_20261026_cuda`；H=`pusht_d2p_p1_root_anchored_v5_calibration_20261026_cuda`。绝对path、SHA与原投影pointer在新 `ancestor_reset_links.json` 中完整保留。

这把原“prefix的父根是谁”缩小为明确的E/H祖先；**不认证E/H原执行producer/reset上界**。E仍只有postrun快照，H仍只有后续physical seal，二者不可倒灌原运行。禁止由selected index推出`reset_seed=base+index`。16条新关联也不是原16条reset阻断已消解，两者不是同一统计口径。

## 新消费者与实际清单

`chat_freshroots_20261027_consumer_v2.py`生成新的consumer manifest，不改旧manifest。历史排重index为原4条物理桥+80条报告前缀身份=84条记录，全部保留，不称84个独立根。原18个unique inherited reset区间及其他seed/conditional包络仍由精确SHA绑定的base manifest引用，不声称本index的pose覆盖替代seed覆盖。

纯函数 `assess(index, actuals, identity, same_pose)`只接受固定slot0..3及state/hash白名单；拒绝cost/success/selected/root_seed等未认可字段，防止从consumer输入偷偷重选seed。它检查：

- 历史pose数值别名、state/root-goal/snapshot/RGB3 hash重复；原XY1e-9pixel、wrapped angle1e-12rad不变；换goal/velocity/加2pi无法绕过pose。
- 批内所有slot两两检查，重复双方都保留并标excluded_duplicate，不补根。
- raw10 hash相同单独不认物理重复；缺失身份保持blocked，未知coverage时即便没有已知重复仍不能授权。

该版本没有coverage批准或授权开关，不能将传入的任意所谓approval转为true。它提供当前阻断态的可审计消费接口，不冒充未来MAIN批准后的完整producer。全部四候选实际state仍未提供，实际输出4条blocked_missing_actual_identity；source_run_authorized/freshness/coverage全false，certified_fresh_roots为空。候选顺序、seed pair、失败分母4未改。

## 测试与独立复核

新增20项RAM tests，联合前三轮及原metadata四套测试 **199 passed in 1.04s**。覆盖显式父层级、exact tuple匹配、歧义保留、不能从final_state补、四slot顺序/分母、历史/批内重复、angle/velocity别名、raw10例外、坏身份、未知coverage不能授权等。fixture不作真实新root证据。

另不调用新consumer函数，使用旧lexical projector重新从两父报告投影白名单，逐个重算state字节SHA并核RGB/root-goal/candidate ID及JSON pointer：80/80通过；另核84条index及四候选阻断状态。父值关联是保存报告的声明，不升级成独立重放或source执行认证。

## 剩余证据请求

1. 80个prefix的祖先/分支声明已显式关联，但还缺seed/reset、snapshot、RGB3/raw10及原采样state等价证明；不得因父源找到了就清614旧身份引用。新解析合同下可记录关联进展，旧resolver不改、旧unknown清除数仍0。
2. 原8来源/16 reset阻断仍完整保留，尤其E/H必须提供原生成run的source/collector/env执行绑定或完整reset ledger；条件区间不替代。
3. original_four物理身份的具体祖先已更明确，但完整family范围、失败/未选/预留范围及618之后delta仍须MAIN明确证据和外部批准，不擅改原family_hint。
4. 四候选在历史coverage获得批准之前不能运行；之后仍需实际四槽历史/批内排重。P1条件准入由MAIN独立处理，本轮不重复。

## 新文件 SHA256

下列路径以 `/home/gbk/code/LeWorldModel/` 为根：

|文件|SHA256|
|---|---|
|docs/chat_freshroots_20261027_parent_scope.json|beb4231eb06fcf33de151de4bbb853f5900d1be2621b72a688ecf04e230dcf61|
|le-wm-main/experiments/spectral_reachability/chat_freshroots_20261027_consumer_v2.py|8b6f9c00ddce88b3c902ed7037d4a127b480b45ac08db718a3601b7197b5af0f|
|le-wm-main/tests/chat_freshroots_20261027_test_consumer_v2.py|8b6449ead559584a7a02318a7582c728e96faf09fe6b5be05b4b21fb3b60207b|
|le-wm-main/results/chat_freshroots_20261027_consumer_manifest_v2.json|2b80412eb2efc7b721d6edb754ce5d9d5d513e17744706aa1e5bab56c785f7e7|
|le-wm-main/results/chat_freshroots_20261027_ancestor_reset_links.json|4d52a593773bf7dd5c323c61b2a1e68e43941a56ceef67609c7dbcb4075189ca|

结果均通过apply_patch新增；消费者只向stdout输出，不创建/覆盖文件。原manifest及prefix supplement的精确path/SHA在consumer `bindings`；ancestor链接文件引用两manifest精确SHA，各项原报告亦按原SHA重核。

复算从 `/home/gbk/code/LeWorldModel/le-wm-main` 执行：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.chat_freshroots_20261027_consumer_v2
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/chat_freshroots_20261027_test_consumer_v2.py tests/chat_freshroots_20261027_test_prefix_identity.py tests/chat_freshroots_20261027_test_closure.py tests/test_root_exclusion_authority_resolution_20261027_v1.py tests/test_moving_authority_identity_link_closure_20261027_v1.py tests/test_moving_historical_reset_quarantine_20261027_v1.py tests/test_moving_root_exclusion_manifest_20261027_v1.py
```

本轮仍使用experiment-code技能的固定输入、失败分母与可复算要求，不采用训练/按效果选优/画图模板。CPU只处理metadata，未降级执行神经实验。
