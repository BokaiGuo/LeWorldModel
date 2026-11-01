# MAIN-only 原生 contact source + strict saved consumer

仅新增本 stem source/tests/doc。旧 contact contract 三文件与旧结果冻结，不修改原 env、globals、
回调、physics、阈值、模型或 normalizer。worker 只RAM，不prepare/probe/physics/neural/network。
这是刻意构造的OOD接触/动作响应校准源，不fresh、不构成D2P贡献或任务成功。

## 固定ABI与执行顺序

prepare接受**MAIN实际保存**的 `agent_p5_contact_source_contract_20261028_v1` JSON与实际FILE SHA。
精确重建原contract.prepare后相等才接受；没有未来hash默认值。新prepare再封存自身三文件、
旧helpers/runtime/native source、原goal资产及已安装Pymunk观察接口源码，绝不运行runtime.probe。

完整复用冻结contract.plans()：seed2026102801、reset_to_state
`[180,271,256,256,0,0,0]`；除此一项保持原生constructor参数。relative raw64，内部尺度100。
原goal `[256,256,pi/4]`，with_target=True，不调goal setter，不补goalRGB或额外render。
原reset含一个.01s Space.step；post-reset实测agent/block位置与零速度、角度、goal及真实圆/T gap=1
必须与预声明相等，否则停止，不能修改初态修过。

实际执行为两个独立env，但**分两阶段**，不先把第一臂后缀跑完：

1. fresh push_x constructor→显式seed→原reset→原10个raw-zero prefix控制步，保持live env。
2. fresh raw_zero constructor→同seed/reset→同prefix10，保持另一个live env。
3. 两个prefix frame0..10，每帧五文件RGB/state/proprio/public state/transition逐字节比较；
   同核显式seed后RNG和十个command文件字节。全部55文件门通过且durable gate入journal，
   才准许任何suffix dispatch。不使用savedpoststate restore或旧prefix替换。
4. push_x十个raw64[.1,0]；raw_zero十个raw64[0,0]。零控制仍运行physics。
5. finally close两环境、finally释放本模块owned profiler；不覆盖已有profiler。

不筛臂、换seed、按结果挑动作、重试、补步或增预算。全部2分母始终保留；任何工程/身份失败
停止后续dispatch，保留实际返回与partial。原done恒False；异常done保留早停资产并使strict不通过。
native_goal/coverage/reward/done分开，不按coverage>.95构造done或本研究task success。

## 真实callback观测（不是静态接触推断）

`make_meter`扩展封存 `base.NativeCalls` 的Python sys.setprofile观察器；未替换
`_handle_collision`，未包装/patch Space.step，未新增solver调用。
在原post_solve callback的Python call时只读arbiter：

- 当前arm/operation/frame、全局Space.step index及operation内substep序号；
- arbiter两shape的真实body identity映射agent/block/static，非center-distance猜测；
- 原contact points的point_a/point_b/distance、normal、post-solve total_impulse；
- 原 `n_contact_points` before，callback return时的after及returned标志。

`total_impulse`按已安装Pymunk文档只在post-solve内部读取，不作力/效率/成功阈值。
读接口不推进physics。观察器不在profile中抛异常（CPython会自动关闭失败profile），
而是保留observer_errors并在下个dispatch/capture拒绝继续；strict必须errors=[]且pending=0。
callback上限8192、每callback点数上限16为工程护栏，不是接触科学门。

每帧独立保存callbacks_00..20.json，report亦保留全原始callback meter。
strict逐callback核原计数增量=len(points)、call-entry数量、return、上下文/Space.step索引；
逐控制步核callback累计points恰等于实际public n_contact_points，
原info.n_contacts==ceil(points/10)。两者不是唯一接触对数量，不混用。
只有实际**agent-block** callback正点数才是本源的观测contact，墙/其他body回调不冒充。
静态first-target gap=-9只属于冻结候选几何，从不作为observed contact。

## 完整物理数据及严格消费

实际frame0..20共42帧，每帧五文件均沿用原panel.capture，不另调render/get_obs；
public state保留完整native block position/angle/velocity/angular velocity、agent全部状态、
RNG/shape/space/env设置。native7末两维是agent_vx/vy，不是block速度。
消费者逐帧交叉核state/proprio与原精度public body、原transition/info、原goal/coverage/reward、
raw command/target/prestate chain，严验byte gates先于第一suffix、全journal链/终端/report绑定、
真实intent-return顺序/计数、完整inventory和来源closure。

完整trace验收与source witness分开：

- all2_saved_source_traces_verified：两臂完整、清理/来源/计量/原生合同均通过，不等于接触成功。
- source_contact_observed：push后缀出现实际agent-block callback正点数。
- push_block_pose_changed_from_prefix：任一后缀帧block原生position/angle与其frame10不同。
- zero_block_pose_stationary：零臂全部suffix block原生pose与其frame10相同。
- paired_block_pose_difference：某同帧push/zero block原生pose不同。

source_witness_observed为上述四个**事先声明的描述性布尔量**合取，不拟合阈值。
exact native pose比较不等于统计显著/抗噪或非交换机制证明，也不把agent惯性作为block响应。
若实际无contact/响应，仍可strict通过完整trace，但witness=false且publication关闭。
partial只返回partial_source_retained_not_verified，witness=null，不将缺失填0。
scientific_admission/action_witness_established永远false，task_success永远null；
单场景两个参考臂不是独立评估样本或planner胜利。

原contract预计总资源保持：construct/close2、seed4、reset2、control40、Space.step402、
render/get_obs42、setup/set_state/Space constructor2、goal-body82、Body constructor88。
新增native_collision_callback是额外**实际观测**轴，次数未知直到运行；不是预填100等RAM值。
每轴分开，不加成总算力/FLOPs。复用sealed Writer的128MiB accounted/96MiB payload/
8MiB final reserve、2048nodes×8192B、200MiB free floor、900s wall+SIGALRM。
前置intent与写前guard，失败不退款。close/terminal用final reserve，不因普通guard耗尽而放弃close。
原异常保留，close失败另记。C代码可能延迟Python signal；最终超wall不能strict通过。
磁盘满等不可持久化故障可能只留下partial文件/intent，缺终端不能取得strict回执。

## source私域与public部署分离

run/consume结果、完整pose/contact/参考suffix、私域prepare/contract都不传planner。
publish须提供MAIN**实际持久strict回执路径/SHA**，并重算consume精确相等；
只在witness_observed且blind_publication_allowed为true后发布。

public目录只允许原contract的七个NPY：真实prefix RGB0/5/10、原生proprio4×3、
prefix raw history10、原统计normalized history2×10、已有独立goalRGB字节复制。
不得发布参考suffix、block pose、contact labels、seed/constructor/geometry/source receipt路径。
prefix来源固定push_x，不按结果选择。原stat normalize复用封存函数，无新拟合；raw-zero历史
不得误写normalized-zero。raw proprio未归一化，后续模型必须采用原冻结native预处理，不能refit。

另写public目录**外**的 `<public_dir>_payload.json`，只含公共inputs与数值加载标志；
`<public_dir>_publication_private.json`保留私域receipt/manifest/report lineage，不交planner。
发布不改旧结果/contract.project_blind逻辑，其unreleased返回值仍保留；本模块只在实际新consumer
通过后发本模块public_context。ready_for_numeric_context_loading不等于科学/P1/P5放行。
这不是OS沙箱：MAIN部署时只挂载七NPY目录+public payload，不能把私域目录/lineage暴露给planner。
发布中断留partial，不自动清理/覆盖，不得拿不完整目录作为ready。

## MAIN调用（worker未执行prepare/probe/run/consume/publish）

cwd `/home/gbk/code/LeWorldModel/le-wm-main`。先完成source审读与RAM：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_agent_p5_contact_source_native_20261028_v1.py
```

下列实际contract文件应由MAIN运行旧已冻结contract.prepare得到；所有占位SHA由MAIN真实sha256sum
填入，不接受未来hash。manifest/report/receipt及public输出均exclusive create，不覆盖/恢复重试。

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/spectral_reachability/agent_p5_contact_source_native_20261028_v1.py prepare --contract results/agent_p5_contact_source_contract_20261028_v1_MAIN01.json --contract-sha256 <ACTUAL_CONTRACT_SHA> --manifest results/agent_p5_contact_source_native_20261028_v1_prepare_MAIN01.json
sha256sum results/agent_p5_contact_source_native_20261028_v1_prepare_MAIN01.json
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/spectral_reachability/agent_p5_contact_source_native_20261028_v1.py verify --manifest results/agent_p5_contact_source_native_20261028_v1_prepare_MAIN01.json --manifest-sha256 <ACTUAL_NATIVE_PREPARE_SHA>
```

MAIN确认后唯一真实物理命令（无GPU/神经）：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH=/home/gbk/code/LeWorldModel/le-wm-main/.jepa-physics-site .venv/bin/python experiments/spectral_reachability/agent_p5_contact_source_native_20261028_v1.py run --manifest results/agent_p5_contact_source_native_20261028_v1_prepare_MAIN01.json --manifest-sha256 <ACTUAL_NATIVE_PREPARE_SHA> --confirm-physics --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p5_contact_source_native_20261028_v1_physics_MAIN01
sha256sum results/agent_p5_contact_source_native_20261028_v1_physics_MAIN01/report.json
```

Saved strict消费和（只有witness=true才允许的）公共发布：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/spectral_reachability/agent_p5_contact_source_native_20261028_v1.py consume --manifest results/agent_p5_contact_source_native_20261028_v1_prepare_MAIN01.json --manifest-sha256 <ACTUAL_NATIVE_PREPARE_SHA> --report results/agent_p5_contact_source_native_20261028_v1_physics_MAIN01/report.json --report-sha256 <ACTUAL_NATIVE_REPORT_SHA> --output results/agent_p5_contact_source_native_20261028_v1_consume_MAIN01.json
sha256sum results/agent_p5_contact_source_native_20261028_v1_consume_MAIN01.json
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/spectral_reachability/agent_p5_contact_source_native_20261028_v1.py publish --manifest results/agent_p5_contact_source_native_20261028_v1_prepare_MAIN01.json --manifest-sha256 <ACTUAL_NATIVE_PREPARE_SHA> --report results/agent_p5_contact_source_native_20261028_v1_physics_MAIN01/report.json --report-sha256 <ACTUAL_NATIVE_REPORT_SHA> --receipt results/agent_p5_contact_source_native_20261028_v1_consume_MAIN01.json --receipt-sha256 <ACTUAL_NATIVE_CONSUME_SHA> --output results/agent_p5_contact_source_native_20261028_v1_public_MAIN01
```

RAM fixtures是显式token recorder（fake Space不积分），用于验证callback/计量/cleanup/字节门/
不发生接触时关闭publication及私域隔离。fixture计数不是实际native结果，不替代MAIN执行。
