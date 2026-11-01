# 固定8来源的历史 reset bound 追溯与保守处置

日期2026-10-27。只读证据追溯；仅新增本文，不改sealed source/report或总账，不运行模型、模拟、物理、重生成roots或下载。不是科学负结果，也不签发freshness。当前8文件的**历史执行完整reset bound仍不能全部认证**；已找到Git历史代码、postrun快照、后续prerun绑定，明确给出其能证明与不能证明的范围。

## 1. 固定分母和原字节校验

全部相对路径以 `/home/gbk/code/LeWorldModel/le-wm-main` 为根。

唯一入口 `results/root_exclusion_authority_resolution_20261027_v1_MAIN01.json`，实测SHA256：

`032ea722f53da7c93db90377d775b75f7ff7a40771a43fdfa38dfd7761e27d00`。

只取 `history_file_requests` 中 `inherited_refs[].original_reason == historical_reset_attempt_bound_not_attested` 的项。实际8files/16refs，每文件分别被reachable_prepare和moving_prepare引用两次。8份当前report字节SHA全部与该入口历史SHA相同；这是旧报告未变的证据，**不是其producer源码被历史绑定的证据**。

|ID|`results/<目录>/report.json`|report SHA256|
|---|---|---|
|A|pusht_d2p_p0_packet_contract_v2_20261026_cuda|`2f29a195f291d040a666cc3db6673f9be12c315c96162b126ddaef9bcdb5e135`|
|B|pusht_d2p_p0_preflight_20261024_cuda|`b1b0fbc29ceb050939ae1403c7644d0ca704b133e47a2411c079dbd626ee0eaf`|
|C|pusht_d2p_p0_preflight_packet_layout_repair_20261024_cuda|`c499ececab67135a2f4d4d698359e6f62b557be66f9132480ef95b14a0dce080`|
|D|pusht_d2p_p1_outcome_admission_20261024_cuda|`a9266fb3cd3e68ee197ec187e8b7916cf3ca8c11966778df3d4c92083b7d6b54`|
|E|pusht_d2p_p1_packet_contract_v4_calibration_20261026_cuda|`a0c9c53bd0161946ec4466878de49795abdf6db102dbb97b174035ed4028a006`|
|F|pusht_d2p_p1_reconfiguration_replay_bound_20261024_cuda|`4911ae7e945abfb042b332f71d7cda0163e14c9727ee8131ff7881bc67257a14`|
|G|pusht_d2p_p1_rollout_alignment_repair_20261024_cuda|`1c5fdd414a5104b6487c2c6e5cb38d05943f902a9d7a695b891ae73202ef7abb`|
|H|pusht_d2p_p1_root_anchored_v5_calibration_20261026_cuda|`99cbb6b6dc4653eb629427353f6f6d05022c219333f3f7362cf4ca43a3076bc2`|

固定过滤复核命令（只输出8行，不打印巨型单行原JSON）：

```bash
jq -r '.history_file_requests[] | select(any(.inherited_refs[]; .original_reason=="historical_reset_attempt_bound_not_attested")) | [.path,.sha256,(.inherited_refs|length)] | @tsv' results/root_exclusion_authority_resolution_20261027_v1_MAIN01.json
```

## 2. 追到的真实历史代码，不拿当前版本冒充历史

### 2.1 Git可恢复旧版本，但没有run→commit绑定

外层Git中可实际读取：

* commit `675cbb2d5d743ab5da8abd53b73d153c17bbdcff`（2026-10-24 18:43:33 +0800）的P0代码，SHA256 `6170c69352462b8755b729b0e8dd32cdb873364944a60be9756f6d8935ad1962`。
* commit `d26e4bff84f31138adf5d1bda809bccafd9be10a`（2026-10-24 19:31:01 +0800）的P0代码，SHA256 `567bcbe1b4edc9241806f1b60882de5d971cab694620153d992199d984b5f3ac`；P1代码SHA256 `24171e996addc6910d30f800f8ad7c03817f9296b95ae4c675cf9dd3168006f3`。P1的SCHEMA为v3，与D/G报告匹配，main显式调用collector(count=args.roots,seed=args.seed,oversample_factor=4)。

可复核例：`git show d26e4bf:le-wm-main/experiments/spectral_reachability/preflight_pusht_d2p_p1_reconfiguration.py`。这些是原Git对象字节，不是当前dirty producer。B/C同用P0 v1 schema不能区分layout修复版本；F的P1 v2并未由这个v3 Git blob恢复。schema相符、提交日期邻近或文档叙述均不能证明实际执行使用了该blob。

这两个commit中以下完整collector调用链字节相同，且当前文件未见差异；列SHA是为了把数学推导绑定到可恢复历史代码，不声称旧run已绑定它们：

|Git中的`le-wm-main/`路径|SHA256|
|---|---|
|experiments/audit_pusht_reachability_horizon.py|`73871055f10b09aea15f03acf87b04e79e30d546668d44bb8160c7be3c439a67`|
|experiments/run_pusht_mode_aware_shared_candidate.py|`1b027be83024ea1d4b868c86f2c1340032596520256e5eb5d7a9be869ad203e6`|
|experiments/run_pusht_signed_torque_audit.py|`f35fe78b70237fcb7f10cabc950b30c163318c5eb5121cbaff8a5081e5ad3493`|
|experiments/run_causal_visual_pusht_world_model.py|`261f609e91da9110f8ba5254033529527314d6bcf38e058c3a237c920d3c4190`|

### 2.2 E有postrun源码，缺原执行前绑定和collector闭包

实际目录 `results/pusht_d2p_p1_packet_contract_v4_calibration_20261026_cuda/postrun_implementation/` 有三份源码，读取hash如下：

* preflight_pusht_d2p_p0.py：`5215f0b49b772d8def14997996774bfd49db614b52b9f615c164271591f7b83b`
* preflight_pusht_d2p_p1_reconfiguration.py：`80b9674063360a2f51d2e146c830c29a5399bb53b9086b177f3b3f2dc3b0d5e4`
* run_pusht_d2p_p1_physical_calibration.py：`7aa9f156484ec3ab483ce3f8308466f60b09a512232f21cd2bcd02f25944f114`

P1快照210行确实collector oversample4；P0快照112行有seed0重建。目录没有collector及其三个下游模块的原执行前seal，也没有reset-attempt journal。名称明确postrun，不能升级为prerun证明。

### 2.3 H有后续物理prerun seal，但不可倒灌到上游model run

实际 `results/pusht_d2p_p1_root_anchored_v3_physical_calibration_20261026_cuda/prerun_seal.json` SHA：

`4e024c0ded185e0bb58c6ceaeb33a0ab1d784165a962995c28f6216d439ef81e`。

7/7 snapshot字节与seal SHA实际核验相符，其中H报告作为`prerun_inputs/06_report.json`绑定SHA99cbb6...；P0 snapshot SHA `745a13d47934f1862a2b43a62b20bc07fc7308075b7b6fea26c3236d9ffa4c98`，P1 snapshot SHA `6dc263e6f69840479daedd92a064bead928b1b1b80e1628f46e4c44964525806`。

后续physical report引用这个seal且`prerun_implementation_and_upstream_unchanged=true`，证明后续执行自己的局部source/upstream绑定；它不证明H生成前已经使用相同source。seal也未包含collector及三个下游模块。物理report即使status=fail，source绑定事实仍可保留；不能因为某gate结果删证据，亦不能把source gate当完整reset ledger。

另找到 `results/pusht_d2p_sealed_program_execution4_v1_20261026_cuda/prerun_source_seal/manifest.json`，SHA `0f0f5b8b8a525990dcffd2393ada1a49270dd50a3d4865800343cfd9b8a39aef`。7/7 archive实际hash匹配，`sources/006.source`确实保存上述collector73871055...。这是后续run的有绑定archive，不是A–H原生成run的closure；下游三模块仍未在该7项manifest内。

### 2.4 D的实际后续report确认被使用，仍没有producer seal

`results/pusht_d2p_p1_outcome_physical_20261024_cuda/report.json` SHA `f67d22b56e8f8bc3f41f4890ccea40ddf0c24f8735f1ea7713dd6d6d5badfe5e`，upstream_p1精确绑定D的path/SHA，protocol outcome_roots_opened=true。但它没有prerun_seal，不包含D的源码SHA或原reset尝试清单。其24根selected root hash、reconstruction/replay声明不能恢复全部未选候选。

外层研究文档 `docs/SPECTRAL_REACHABILITY_ATLAS_RESEARCH_PLAN_20261023.md` 约1860–1920行记录旧layout/alignment修复及D/G执行经过，是历史叙述而非source seal。只读查询user journal 2026-10-24至27、匹配精确producer/报告名无条目；项目文件清单及可读/tmp中未找到这8run各自的执行命令/source-hash/reset日志。此为明确搜索范围内未取得，不声称所有机器/会话日志不存在。没有执行历史shell、源码或任何重建来补证。

## 3. 逐层reset循环：为什么root_count×4还不等于完整bound

下列推导仅对第2节已恢复的**代码族**成立，未绑定的历史执行不自动获认证。

1. `audit_pusht_reachability_horizon.py:84` 的collector先调用make_boundary_contact_roots(env, count*oversample_factor, seed)，**完整返回候选后**才去重/去terminal并截取count。selected root数不能当生成候选数。
2. `run_pusht_mode_aware_shared_candidate.py:387` 默认max_initial_angle_error=inf，因此base_count=count，无finite-angle额外8倍；调用make_terminal_roots(...,root_protocol='terminal_pose')。64次边界几何尝试是NumPy/shape运算，不执行reset，不能把它误算成64个reset seed。seed+70000是NumPy RNG，不是env.reset。
3. `run_pusht_signed_torque_audit.py:189` 的terminal_pose分支调用terminal_pose_cases(...,support_gap_max=0,position_jitter=16)。其另一个radial_contact分支确有seed+50000/+60000和while循环，但**本调用路径没有走它**，不应无依据加到本代码族包络。
4. `run_causal_visual_pusht_world_model.py:443` 中attempt_budget=max(roots,1)*(20 if support_gap_max>0 else 2)，for index依次reset(seed=base+index)，达到roots后break。默认support_gap_max=0，当前分支不做support拒绝，每次append，成功路径实际恰好4n个seed；仅按显式for硬上限取更保守包络则为8n个seed。有限support/angle若被历史代码另开，不能套此8n。
5. 每次terminal case还执行reset(seed=0,options={state,goal_state})并render。旧Git P0 `_collect_causal_rgb_context` 115/127行分别对source和goal reset(seed=0)；后来的packet版本也有seed0。因此必须把 **seed0的显式state重建**另登记，不把它当seed0随机根。seed0复用不意味着所有state同根；需要state/goal/history身份排重。
6. P1 compile中的seed=base+root_index、projection的+10000是Torch/RNG proposal/projection种子，不因变量叫seed就当env.reset。与它同名的physical执行reset必须按physical source另追，不由model-side协议代替。

env安装源也只读查看：当前`stable_worldmodel/envs/pusht/env.py`构造器不调用reset，reset入口转交super seed；但A–H没有绑定当时installed env/wrapper版本，不能把当前环境实现当旧包闭包证据。

## 4. 可采用的候选保守seed spans（闭区间，尚非历史认证）

表中n来自原report，oversample=4。紧路径为base..base+4n−1；保守for-envelope为base..base+8n−1，外加{0 explicit-state resets}。**表不是实际尝试ledger，不应写成observed/exhaustive_attested。**

|ID|n/base|默认成功路径随机reset spans|保守for-envelope|
|---|---|---|---|
|A|4 / 202753000|202753000–202753015|202753000–202753031|
|B|4 / 202610241|202610241–202610256|202610241–202610272|
|C|4 / 202710241|202710241–202710256|202710241–202710272|
|D|24 / 202710281|202710281–202710376|202710281–202710472|
|E|4 / 202753010|202753010–202753025|202753010–202753041|
|F|4 / 202710251|202710251–202710266|202710251–202710282|
|G|4 / 202710251|202710251–202710266|202710251–202710282|
|H|4 / 202753010|202753010–202753025|202753010–202753041|

A/B/C的protocol.outcome是**reserved_not_opened**，不把保留计划写成实际reset。排除资产仍须保留其预留命名空间：A base202753100、B base202610281、C base202710281，各24根/oversample4；在同代码族保守预留包络分别为202753100–202753291、202610281–202610472、202710281–202710472。D明确打开C预留队列，这不说明B/A预留队列也打开。

E/H与A存在保守span交叠，C/F/G也交叠；D与F/G保守for-envelope尾部也可能交叠，但默认成功路径未必交叠。**区间交叠不是实际同根证据**；不同seed也不是独立性证据。这里只作更保守隔离，不用它推翻或重写旧disjoint gates。future source仍须对实际physical state/goal/history执行身份排重。

## 5. 逐项处置及最小missing，不能只凭当前代码清除16条

|来源|历史版本实际找到什么|严格处置／还缺什么|
|---|---|---|
|A|report v2及packet-contract audit外部SHA；后续postrun/prerun中有P0 packet代码|可保守隔离A候选及预留span；缺A run→P0确切blob、collector到terminal_pose_cases闭包、完整reset/failure尝试清单或其等价source-bound上限认证|
|B|675cbb2旧P0代码确实存在，report v1；同目录仅report|缺B run→675cbb2/blob的绑定及root factory闭包；不能以旧v1 schema确认是哪次实现；保留旧16refs中的B两项|
|C|d26e4bf旧P0 layout修复代码及report v1；D/F/G显式upstream指向C SHA|upstream边证明report被用，不证明C使用该代码；缺C执行前source/依赖绑定与attempt bound；保留C两项|
|D|Git可恢复P1 v3，与report匹配；actual后续physical精确引用D|**MAIN所指出问题确认存在**：root_seed/count/oversample和同schema Git blob不能建立D run→source。缺执行命令/运行source-hash/seal或原reset ledger；采用202710281–472只能标条件quarantine，不能标已认证历史coverage|
|E|有三份postrun_implementation真字节和明确oversample4调用|缺postrun快照与E实际执行版本的前置绑定，也缺其collector闭包；mtime/目录名不能补；保留E两项|
|F|report为P1 v2，现已恢复的Git P1为v3|确切v2 producer原blob未取得；不能以v3数学相同猜v2；须恢复v2及collector/terminal链或原reset ledger，保留F两项|
|G|Git P1 v3匹配schema、后续physical报告及历史叙述存在|仍缺G run→Git blob/依赖绑定、完整尝试上界；不得以D/G同schema互认证，保留G两项|
|H|后续physical seal同时绑定H报告和P1 v5 snapshot，7/7文件hash实核|只认证后续run；缺H自身执行前source seal及root factory闭包/attempt ledger，不倒灌；保留H两项|

目前**0/8获得足以自动清除historical_reset_attempt_bound_not_attested的完整新证据**，不等于8run没执行、没代码或没selected identity。实质推进是恢复可检查的旧代码族、给出具体seed包络、识别postrun与后续prerun的真实绑定边，并把缺项缩小到执行版本绑定与root生成闭包，而非笼统unknown。

MAIN可立即采纳两类非追认处置：

1. 将上述for-envelope、预留队列和seed0显式state重建身份加入新的**保守隔离候选集**，标签为`conditional_code_family_quarantine_not_execution_attested`，绝不能借此将freshness_certified改true。无完整闭包时这个有限集合也不能证明覆盖所有历史版本。
2. 若MAIN持有原执行日志/source seal，精确补reportSHA→执行源码blob→collector→三个下游模块/实际env的绑定；或提供绑定report的全attempt/reset清单。可以另封authority文件，声明逐项采纳的保守上界、依赖版本及失败run覆盖，旧报告保持不动。认证内容须涵盖未选/失败候选及预留来源，不只selected root hash。若确实没有这些材料，应保留阻断或由MAIN明确裁定更保守的历史scope；本文不自签豁免。

仅依据现有代码和report无法证明未记录的进程重启、先前失败run、替代collector/安装环境没有调用别的seed。用一次新模拟复现selected roots也无法弥补这个历史事实，因此本任务没有这样做。

## 6. 验证与未执行项

只读标准库脚本实际核验：8/8报告SHA、8files/16refs固定分母、后续physical prerun 7/7 snapshots、后续source archive 7/7文件、两Git commit的原blob SHA；纯整数计算n=4/24的4n/8n spans。未调用producer、gym、torch、numpy模型、模拟或CUDA，也未写prepare/result。唯一新产物为本文；不运行大tests，不修改共享账或旧authority结论。
