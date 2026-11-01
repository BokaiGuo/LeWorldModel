# 原8来源 / 16引用：最小 authority 补证请求与有界验收

日期：2026-10-28。独占新增文档及 `agent_freshroot_authority_20261028_v1.py` / 对应测试。旧源码、结果、共享总账不改；不扩大历史扫描，不运行神经/环境/物理，不采样、不派生agent。MAIN负责审读、原运行来源真实性裁定及后续调度。本交付不增加任何fresh根、不改变认证门槛。

## 为什么新增这个窄接口

已核查 `root_exclusion_authority_resolution_20261027_v1` 的源码/innerdoc，以及本人旧coverage/next交付：前者处理metadata投影/身份桥，后者恢复命令和返回链；两者均没有原运行方补证响应的严格schema验收。本实现不重做其扫描，不调用旧producer，不自动follow父绑定。

按experiment-code技能及其两份references固定输入/预算/失败处理；用户限定metadata/RAM任务优先，不执行训练、图表或性能优化。旧blocked个人goal仍占用，未覆盖或假标完成；本人本次完成条件仅是请求协议、验证器、RAM回归和固定catalog实际读取。

## 已取得什么，不必重复索取什么

前次封存结果：`le-wm-main/results/agent_freshroots_resume_20261027_next_delta.json`，SHA `5ef07ab7815fff0c392feef59309c0116596e95d541eb95413d1d966a8b2a29b`。固定1033文件全部可投影，464条literal path+SHA连接；这不是global coverage。13条dispatch/poll、15条patch返回无悬空配对。已消除5项进程状态未知：B前两次异常、A/E/H明确exit0。完整reset认证仍未解除，不能将5项状态恢复写成5个reset blocker解除。

当前request只读旧coverage：

`le-wm-main/results/agent_freshroots_resume_20261027_coverage.json`

SHA `00a3fffeaaa8943e851d129b941a7f07feaa1db387117beb98110e4ec95f5eeb`。

**16是8个来源各被reachable_prepare/moving_prepare引用两次，不是16个独立原run。** 同一来源补证可共享给两条引用，但不能遗漏B的早期调用。工具输出原report绝对path/SHA、命令、producer和原log/行号；不重新读取巨大原日志。

## 逐来源请求矩阵

每行对应两个request_id：`<ID>/reachable_prepare`、`<ID>/moving_prepare`。完整call_id见CLI，不用目录名推断run。

|来源|原目录（均在inner results）|已定位dispatch行 / 最小调用范围|补证接收方最少提供|
|---|---|---|---|
|A|pusht_d2p_p0_packet_contract_v2_20261026_cuda|58279，1次；exit0已恢复|该次加载闭包+完整有限reset包络，或穷尽reset ledger|
|B|pusht_d2p_p0_preflight_20261024_cuda|22791/22814/22850，3次；前两次异常已恢复|三次各自ledger/包络，含失败前尝试；shell版本和producer版本不得合并成最后一次|
|C|pusht_d2p_p0_preflight_packet_layout_repair_20261024_cuda|23599，1次|同A；不把最终root_count作为尝试上界|
|D|pusht_d2p_p1_outcome_admission_20261024_cuda|23773，1次|同A，额外P0依赖闭包|
|E|pusht_d2p_p1_packet_contract_v4_calibration_20261026_cuda|58307，1次；exit0已恢复|同D；postrun_implementation不是原加载收据|
|F|pusht_d2p_p1_reconfiguration_replay_bound_20261024_cuda|23658，1次|同D，原P0版本不能以后续版本替代|
|G|pusht_d2p_p1_rollout_alignment_repair_20261024_cuda|23730，1次|同D|
|H|pusht_d2p_p1_root_anchored_v5_calibration_20261026_cuda|58363，1次；exit0已恢复|同D；后来physical seal引用报告不证明当时加载字节|

行号均来自旧coverage固定的Oct23会话01a16903-c194…jsonl。每行范围仅为**已发现调用**，不是声称没有其他重启。B早期失败与最终report的SHA对应只是来源家族join，不主张失败时最终report已存在。

## 最小证据的两条可替代路线

### L：穷尽原reset ledger（可不再索取完整加载闭包）

每次call_id对应真实全部reset尝试，含随机采样、显式state重建、goal render所需reset及replay；保留selected/unselected/rejected/failed，而非只提供被选根。原现场失败未观测到identity可以null，不能填造state。连同原run范围、restart完整性和来源证明提交。无reset前即失败可提交0次，但“0次”的真实性仍须原记录支持。

### C：原加载闭包 + 可穷尽的有限包络

每次call_id提供producer、P1时P0依赖、四层factory、installed env、wrappers；B三次shell入口也须对应。字节附件和原run加载依据不可缺一。原factory候选：

- `audit_pusht_reachability_horizon.py`
- `run_pusht_mode_aware_shared_candidate.py`
- `run_pusht_signed_torque_audit.py`
- `run_causal_visual_pusht_world_model.py`

原env候选路径 `.venv/lib/python3.12/site-packages/stable_worldmodel/envs/pusht/env.py`，连同实际wrapper/import closure。以上是请求定位线索，不是认定旧run加载了当前文件。

有限包络须逐调用涵盖失败/未选/拒绝、随机reset、显式state/goal/replay和重启；文字推导必须可由绑定闭包逐处检验。若有数据依赖的隐藏while且没有可证上界，路线C不够，需L。验证器只校验声明的count不超bound及附件，不自动证明推导数学正确或停止性。

## 本机可恢复 / 需原来源权限

|证据|本机已明确取得或可做的有限操作|仍需的原来源权限/资产|
|---|---|---|
|命令、候选producer版本、异常/退出|旧coverage/next已保存，不重扫|无需再问已恢复的5项状态|
|原加载字节|已有历史Git/blob和patch重建候选，可在收到明确manifest后按path/SHA比对|原prerun/load manifest、可追溯容器/环境快照，或同强度的执行绑定；当前SHA/mtime/包版本不足|
|失败/未选reset|现有异常收据证明调用失败，但没记录此前所有尝试|每次原reset journal/进程stdout归档，或已绑定闭包可证明的有限包络；不存在的日志不能事后制造|
|历史边界|固定618清单外已补1033路径差集，不能推全覆盖|原运行负责人说明机器/工作树/时间边界、未保存失败及其他重启；需可追溯依据，不能只写complete=true|
|未来candidate排重|有身份算法与固定候选规则，尚无认证候选|仅在另行授权采集后取实际state7/物理pose/snapshot/RGB3作历史与批内比较；本工具不采集|

**没有新发现能够恢复旧加载字节真实性的本机文件。** 本工具不声称记录不存在，只明确已固定材料尚不能证明；若负责人确认原记录遗失，应记录不可恢复及具体run，维持未知，而不是从未找到推导fresh。

## 机器schema：request与response

工具只用Python标准库，不导入其他实验模块。`request`从上述SHA固定catalog生成，semantic SHA为：

`9b304bc853f678046cae8dabd7adeda1866ada6cbe8933b7cd6539b16ff699a0`

semantic SHA使用 `json.dumps(sort_keys=True,separators=(',',':'),allow_nan=False).encode()`；不含CLI末尾换行。request包含8个固定来源及16条继承request_id、10次dispatch。**request有效不意味着response已存在。**

response顶层恰为：

```json
{
  "schema": "agent_freshroot_authority_20261028_v1.response",
  "request_sha256": "9b304bc853f678046cae8dabd7adeda1866ada6cbe8933b7cd6539b16ff699a0",
  "sources": [],
  "historical_scope": null
}
```

这是真实可接受的**空/未完成响应schema示例**：返回8来源缺失、0解除，不是补证成功。允许部分响应，未提交来源必须保持missing_sources；同一来源漏调用需非空unresolved_reason，不允许以最终调用代替B前两次。

每个source恰含 `source_id / report_sha256 / runs / unresolved_reason`；来源和report须匹配request。每个run恰含 `assertion / origin_citation`。

assertion字段：

|字段|严格合同|
|---|---|
|call_id / report_sha256|匹配该来源已发现调用和来源家族原report，禁止猜行号join|
|route|`exhaustive_ledger` 或 `loaded_closure_and_exhaustive_envelope`|
|attempts|0..1e9整数，bool不是整数|
|attempts_complete / restarts_complete|必须为true才能提交该run；只是待核实声明，不是信任锚|
|reset_evidence|citation，指向下面对应ledger/envelope对象|
|loaded|L路线为空；C路线完整精确role/path/sha256列表，字节附件逐个核SHA|

L路线reset对象恰含call_id和attempts数组；行恰含 `index/reset_kind/disposition/identity_sha256`。index从0连续且总数等于声明attempts；reset_kind四类见L；identity_sha256只允许failed行null。identity哈希本身不能替代原state/物理pose排重证明，MAIN需核其原算法及实际载荷；不把不同goal的hash当不同物理根。

C路线reset对象恰含 `call_id/upper_bound/derivation`，非负整数上界及非空推导。loaded的role必须精确等于producer、factory_1..4、environment、wrappers，另按真实命令增加shell_launcher、按P1增加p0_dependency。wrappers附件可为有原始依据的完整wrapper/import manifest；工具核字节，不擅自解释运行语义。

所有citation恰含 `path/sha256/pointer`；literal绝对无symlink路径、完整文件SHA、RFC6901 pointer。origin_citation指向与assertion语义逐字一致的**来源方提交记录**。该记录可以是现在整理的有原始证据支撑的导出，并非要求旧日志当时已有新schema；它必须由MAIN核验其原始证据链，不能因自签JSON匹配就认定为当时加载收据。原生JSONL/二进制容器不能直接用本JSON-pointer接口，需来源方提供显式可追溯metadata导出；本工具不自动执行解包器、不把导出时间倒灌为旧运行时间。

historical_scope可null；非null时恰含 `machines/worktrees/time_interval/unsaved_failures/origin_citation`。前两项非空字符串列表，后二项非空文本；citation精确绑定除citation外四项。机器只验证声明绑定，不证明未保存失败不存在或范围穷尽。最终origin review必须独立于响应中的自述字段。

## 接受与拒绝示例 / 安全边界

- 接受进入origin review：RAM测试中，完整两次attempt（失败identity=null、显式重建selected有hash）、其ledger和exact assertion均有外部SHA附件。B必须全部3次；C路线须全部角色及字节。返回 `syntax_and_bound_bytes_valid_not_authority`。
- 接受但明确未完：上面的空response；或B仅最终调用且unresolved_reason写明前两次缺失。固定缺项仍输出。
- 拒绝：report/call/request错配、重复source/run、遗漏字段、额外`externally_reviewed_complete`、已选root空identity、attempt index断裂、bool count、count超bound、缺factory角色、origin与声明不符、任何附件SHA变化、路径逃逸、重复JSON键/NaN。

输出**始终** `origin_review_required=true, reset_blockers_cleared=0, certified_fresh_candidates=0, P1_admitted=false`。没有自动approval模式、自动全局发现、自动选根或下游resolver override。若MAIN完成原来源核验，应另封存明确裁定/范围，不能把本程序exit0当裁定；本版本刻意不消费或伪造该裁定。

读取预算：输入response/catalog/JSON citation每份4MiB，其他附件每份32MiB，累计128MiB、最多128个唯一路径；输出4MiB。缓存路径不同SHA拒绝。无递归follow；超限直接失败，无重试、无补造；不支持的原始格式需明确补接口，不能静默忽略。运行只输出stdout、不写结果，因此不会覆盖seal；完整外部文件SHA包括原文件换行。provider仅供RAM测试，生产CLI不可替换provider。

## 实际验证及下一条命令

本次实际只读上述固定catalog（不读全盘/旧巨大log），request成功生成：16refs/8sources/10dispatch，stdout10757 bytes。实际空response纯内存验证返回missing A–H、0认证。RAM回归 **38 passed (0.12s)**：覆盖两条路线、B三次/partial、SHA与指针变异、scope声明、不合法输入和字节cap。例中RAM内容不是科研证据，未生成假prepare/report。

在 `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_freshroot_authority_20261028_v1 request
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_freshroot_authority_20261028_v1.py
```

原来源提交实际response后，MAIN调用：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_freshroot_authority_20261028_v1 validate --response /ABSOLUTE/ORIGIN_RESPONSE.json --response-sha256 ACTUAL_EXTERNAL_FILE_SHA256
```

最后一条只是有前置条件的接口命令，当前没有那份原来源response，不可声称实际补证验证成功。最小下一动作是将矩阵交原运行负责人，优先B三次attempt ledger/已加载快照，逐项提供现存附件或明确不可恢复；不需再次让本worker遍历目录。

## 事前新协议能解决的边界

可以独立封存随机新采集的有限池、顺序、失败分母及全尝试journal；但当前历史缺口下，仍不能由“新seed/随机概率极小/未找到旧记录”证明全历史物理根不同。若无完整旧身份或严格历史域不交证明，global freshness仍unknown。任何缩小确认范围或更换证明标准需MAIN另行明示协议决策，不能在本schema中暗改。known-root原生成功1/8不是freshness证据；本工具不读取outcome做来源排除。PushT任务位置同时含agent/object，不用object-only投影替代完整任务判据。
