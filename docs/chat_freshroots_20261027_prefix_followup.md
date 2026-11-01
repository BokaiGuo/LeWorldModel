# 历史前缀身份补充：实际解析80项，而非重述unknown

这是上一轮manifest之外的新增证据补充；不改原清单、coverage、resolver、catalog、producer或任何结果。完整freshness目标仍未达成，不以本补充签发source许可。未调用子智能体、模型、物理、训练、网络或删除/推送。

## 固定范围及实际结果

先封存 `docs/chat_freshroots_20261027_prefix_scope.json`，再执行补充消费者。唯一选择规则是原capture `top_level_keys` 含 `prefix_state` 的全部文件；不是按status/success/cost、root目录编号或planner结果选择。固定80分母，失败不剔除。只解码 `/prefix_state/*`、`/prefix_rgb_sha256`；`state/final_state/status/official/cost/probes/actions` 等均词法跳过，不用它们评分或补身份。

原capture绝对路径 `/home/gbk/code/LeWorldModel/le-wm-main/results/root_exclusion_authority_resolution_20261027_v1_inventory618_MAIN01.json`，SHA `e2afed12a1d8f788ad16d0223482377b5f54d281e67739766475228b53ac2591`。80个精确path/SHA均来自该封存capture，不扩目录、不follow其他引用。

实际80/80解析、0失败，读取1566709字节；按原XY1e-9pixel / wrapped-angle1e-12rad固定容差形成8个pose别名连通分量。每个分量保留全部原记录，不称8个独立根。它们与原manifest四个reachable/moving物理桥没有已知numeric-pose匹配；**不匹配不是freshness证明**。80份源对应原160条继承引用，但这160条没有被清除：新增身份仍缺源采样根/episode、seed/reset、snapshot及RGB3/raw10映射。

补充记录保留state7及规范little-endian float64 SHA、wrapped pose/key、原prefix单RGB SHA、明确缺失项和path/SHA/pointer。单RGB SHA只是旧报告声明，未重新解码PNG，不冒称完整RGB3/history。seed/reset/snapshot/raw10均null，未从文件名root_N或同目录推断。只能用于未来更保守的历史排重，不能冒认已证明原采样root的完整物理身份。

## 验证与原阻断

新增16项RAM测试，覆盖只读前缀白名单、缺失字段不从final_state补、duplicate key/尾随内容、state7类型/形状/非有限数、RGB SHA、2pi/velocity别名及全记录分组。与原五套测试联合 **179 passed in 0.96s**。初次收集发现新源码dict关键字笔误；修正后完整运行通过，不修改任何封存文件。

保存后另用不导入消费者的stdlib检查，重新核80原文件SHA、80 state7字节SHA、8组全分母及所有false/null状态：PASS。原manifest、coverage和handoff SHA与上一轮相同。

本次没有消解原8来源/16历史reset阻断，614条原缺直接身份引用仍按原合同保留；没有把新增prefix身份改写进旧resolver。已知moving两根仍不能算fresh。四候选顺序、原seed pair、分母4全部沿用原manifest，无替换、重试或收益筛选。coverage/freshness/source-run授权全false。

仍需MAIN提供：逐原run的实际source/root-factory/env绑定或全reset ledger；这80份prefix到采样root/episode的原字节绑定映射（包括未选/失败来源），原snapshot/RGB3/raw10和seed/reset；完整历史范围批准及618快照后的delta inventory。后续仍需实际四槽批内/历史身份排重收据，不得以新目录或新seed代替。

## 新文件/SHA256

根路径 `/home/gbk/code/LeWorldModel/`：

|文件|SHA256|
|---|---|
|docs/chat_freshroots_20261027_prefix_scope.json|dc2d13ac58b0392aba24253739e151000408704b37d24fe5c77a8810722a0c3b|
|le-wm-main/experiments/spectral_reachability/chat_freshroots_20261027_prefix_identity.py|063daa37801f702a8a0d88659f81f0a40a09f3594ea1ffdde7d6f82d66cb874d|
|le-wm-main/tests/chat_freshroots_20261027_test_prefix_identity.py|e35d155aaabd052f66dba68a9b4edf3012f4a3df710486d1ece2ecea616ccc0e|
|le-wm-main/results/chat_freshroots_20261027_prefix_identity_supplement.json|2cd81f5741fbef9fea0eae62b913e9ad73b09fa8c9d02e525876d6854a91d7e9|

消费者仅stdout，不自行写文件；本次结果由apply_patch新增。重算从Python根执行：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.chat_freshroots_20261027_prefix_identity
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/chat_freshroots_20261027_test_prefix_identity.py tests/chat_freshroots_20261027_test_closure.py tests/test_root_exclusion_authority_resolution_20261027_v1.py tests/test_moving_authority_identity_link_closure_20261027_v1.py tests/test_moving_historical_reset_quarantine_20261027_v1.py tests/test_moving_root_exclusion_manifest_20261027_v1.py
```

采用experiment-code技能的固定范围、失败分母、实际执行与复算要求；不采用训练/效果调优模板。CPU仅用于metadata算术与字节验证，不是CPU神经降级。个人goal保持active，未将历史认证缺口视为已完成。
