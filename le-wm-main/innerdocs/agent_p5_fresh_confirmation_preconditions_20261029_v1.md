# P5 fresh confirmation 前置解析（独占 v1）

只读固定6份已存在的 metadata/code/protocol 文件，复用 `agent_freshroot_authority_20261028_v1.requests/Reader`，不重建其补证schema，不调用旧 fullstrict、identity collector 或 producer。使用 experiment-code 技能组织固定输入和失败回归，用户授权边界优先：不训练、不NN/physics、不下载、不采样、不打开reserved outcome、不创建子agent，不修改canonical/封存文件。

## 现有等价组件与本次最小增量

已有 `moving_root_exclusion_manifest_20261027_v1` 负责来源metadata投影、跨root/reset seed及half-open reserved spans、state7/pose/snapshot/RGB3去重和外部MAIN coverage approval；`chat_freshroots_20261027_consumer_v2.assess` 负责固定四槽历史/批内排重，但不能补coverage；`agent_freshroot_authority_20261028_v1` 负责原8来源/16引用/10次已知调用的补证request及response字节绑定，始终要求原来源审查，不能签freshness。本工具只把它们已封存的blocked状态与 P5 独立确认尚缺的明确上游材料连接起来；无另一套排重阈值/approval override。

输入精确path/SHA在source `PINS`；仅这些文件可被读取，不自动follow任何内部引用，尤其不读实际heldout/planner outcome。文件名清单只用于发现现有组件，不是全历史覆盖证明。resolver status固定为当前snapshot的 `blocked_no_certified_fresh_cohort`，组件ready仅指代码/元数据可复用。若未来有真正完成的approval/cohort，必须先由旧authority和MAIN审查、再明确新版本绑定，本snapshot解析器没有变ready开关。

原四槽为 `203027101..104` root与 `203027201..204` reset固定对应，分母4、无重试。保存其原ID与顺序，不称四fresh，也不把两known roots、四strata、164 branches或80pairs扩充成独立根。`P5_confirmation_denominator=null`：未来P5自己的N/seed尚未批准，旧四槽设计不能擅自移植。输出fresh认证0来自已绑定候选清单为空，不是填补未知候选数量。P1/P5 admission始终false。

## 最小上游请求（非重复16条包装）

1. 原8来源逐条携带原report path/SHA、原call_id/command/producer/log绝对path与行号；B保留全部3次调用。运行来源负责人提供完整failed/unselected/rejected/random/explicit/goal-render/replay reset ledger，或原loaded producer/factory/env/wrapper（P1含P0依赖）字节闭包与可证有限包络；可直接复用旧authority `request` / `validate`，该工具exit0仍非来源认证。原日志证明已发现调用，不证明无其他重启。
2. MAIN提供旧cutoff以后完整scope delta inventory及原身份桥（path/SHA），覆盖全部已接触及reserved prepares与未选/失败来源。不能用618路径census、当前source、mtime或最后root_count回推历史loaded bytes/reset上界。输出保留旧content SHA、unknown数量及原consumer精确缺项，不宣称旧列表是当前全范围。
3. 需要MAIN外部coverage approval文件，绑定**新完整**manifest content SHA/cutoff/scope；当前旧清单approved=false、有unresolved。只有approval文字或身份哈希不同不够，不读取reserved outcome核排重。
4. 需要MAIN事前P5确认协议文件及外部SHA：N、候选顺序/seed/采样分布、source/episode分割、冻结模型/task/history/动作/labelABI、所有角/预算、失败分母/停止规则。原LeWM CLS192 moving源协议不是JEPA P5 factorial确认协议；不能混称同模型/输入，不能按效果选根。
5. 随后才可在另授权source收集中提供真实全attempt/reset/load journal、各slot实际state7/pose/snapshot/RGB3/raw10与历史/批内排除receipt；最迟witness/outcome dispatch前排除。新producer路径与实际SHA尚未存在/未提供时明确缺失，不编造可运行采样命令。此工具只准备请求，不授权采集；不足N不得自动补根。

以上新文件均以`required_new_file_role`而非假定未来文件名/SHA表达。原已知文件仍给exact path+SHA/pointer来源。缺任何一层，ready的只是元数据接口而不是cohort；没有安全材料可本机再恢复原加载真实性时交原run custodian，不重复扫描。没有证明全项目不存在其它可能资产，只证明这份明确绑定集合未提供它们。

## CLI / 状态 / 写入

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
.venv/bin/python -m pytest -q tests/test_agent_p5_fresh_confirmation_preconditions_20261029_v1.py tests/test_agent_freshroot_authority_20261028_v1.py
.venv/bin/python -m experiments.spectral_reachability.agent_p5_fresh_confirmation_preconditions_20261029_v1 resolve
.venv/bin/python -m experiments.spectral_reachability.agent_p5_fresh_confirmation_preconditions_20261029_v1 resolve --output results/agent_p5_fresh_confirmation_preconditions_20261029_v1_MAIN01.json
.venv/bin/python -m experiments.spectral_reachability.agent_p5_fresh_confirmation_preconditions_20261029_v1 verify --resolution results/agent_p5_fresh_confirmation_preconditions_20261029_v1_MAIN01.json --resolution-sha ACTUAL_RESOLUTION_SHA
```

resolve无output只stdout；显式output只允许本stem新文件，排他创建、fsync file+directory，4MiB输出/200MiB floor。原Reader每metadata4MiB、总128MiB/128paths；本工具实际固定6paths，重复JSON键/NaN/SHA冲突/符号链接/未知输入变更拒绝，不把corruption伪装成missing或ready。verify确定性重做这个小resolver，不fullstrict。exit0只表示解析成功，可同时明确blocked。RAM fixture和真实writer回归不是科研认证。后续读取新outcome前应另外冻结真正P5协议；本工具没有根据实际结果调整采样或门槛。

## 实际交付验证

新增18项fixture与旧authority38项联合：56 passed / 0.33s。实际无output的只读resolve已exit0，得到blocked、认证0、原四设计槽位、P5分母null、8来源10调用，reserved_outcomes_opened=false。复用request semantic SHA仍为`9b304bc853f678046cae8dabd7adeda1866ada6cbe8933b7cd6539b16ff699a0`。未持久化真实resolution、未跑生产verify/fullstrict、未启动任何采样或NN/physics。MAIN可按上面NEW输出命令审读后持久化并复验；没有新的历史blocker被此聚合器解除。
