# 固定 contact-source 合同：OOD 候选，尚无实际接触认证

只新增本 stem source/tests/doc，旧 v12、12word 源码、旧几何/goal-scene、所有结果保持封存。
本模块实现 saved prepare/verify 和严格数值 allowlist projection，无 run、无native import、
无模型/物理/网络、无teacher/readout/adapter fitting、无训练。worker只RAM，不产生科学证据。

## 已有入口与最小缺口

1. `p5_jepa_factual_pilot_20261027_v1`：原 CTOR、runtime metadata、native捕获、journal/profile。
2. `p5_native_contact_geometry_main_20261027_v1`：已保存实际T多边形局部顶点、圆半径15、
   body local origin→world转换及outside-circle gap；不重新实现另一套几何或PD模拟器。
3. `p5_native_goal_scene_20261027_v1`：已有独立 goalRGB/source generation资产，无需再生成goal。
   原 reset_to_state 合法入口已有使用，但并不因此继承新source接触/任务认证。

旧12word所有120suffix contacts0/object固定是已完成负证据，不重复运行、不替换旧prefix。
新合同只补静态接触候选前置。硬缺口仍是**另一个新独占 native generator/strict consumer**：
其实际reset几何、两条独立完整prefix、参考动作接触/物体响应、资源/清理和来源回执尚未产生。
本文件不含实际physics runner；不是以prepare成功冒充source生成完成。

## 固定新recipe，必须明确OOD

唯一 canonical scene：seed2026102801，原生state7 reset参数
`[180,271,256,256,0,0,0]`，其中末两维为agent velocity，不是block velocity。
T原local origin(256,256)、angle0，agent圆中心(180,271)、radius15，零初速。
使用支持的原生 constructor.reset_to_state 参数；只改这一项，其余原CTOR逐项保持。
`legacy=False, relative=True, action_scale=100, with_velocity=True, with_target=True,
render_action=False, render_size224, shape=T`，原damping/CoG/color等不变。
这改变旧v12 random-reset合同，属于**刻意构造的OOD校准源**；不是fresh根/新独立架构/
任务成功/D2P算法贡献。不在旧adapter里patch，不将旧状态/旧RGB当新场景。

实际新generator须构造→显式seed→原reset一次（原_set_state内有一次.01s physics）→
十步raw64零prefix→十步参考suffix→finally close。raw zero仍是真实PD控制与physics，不是暂停。
两臂固定顺序：push_x（十个[.1,0]）与raw_zero（十个[0,0]），两臂各独立fresh constructor，
全部2分母，无筛选/替换/重试/额外动作/按outcome微调。源文件给出可执行constructor()/plans()/
reference_queues()以供未来新adapter直接消费，不需要从描述猜控制值。

源几何元数据复用实际sealed frame00的local shapes，仅用于核相同native形状与collision support，
完全不复用旧agent/block位姿。新静态计算：T左横梁边界x196，中心y271；初始净间隙1。
首个push控制绝对target(190,271)，静态circle-polygon gap=-9。
这只说明目标进入接触候选区域，**不保证PD轨迹、碰撞回调、物体运动或有效推力**，
不产生连续swept-path证书、不在worker积分/运行新物理。原body CoG不重复扣除。

## source witness 与部署盲payload分开

prepare产物是**私域**source合同，包含STATE/seed、参考队列、几何、witness需求和来源closure，
绝不传planner。原goal是[256,256,pi/4]且RGB含原nativegoal；不插goalsetter或改goal协议。
已存在独立goalRGB可直接复制到新公共目录，按原文件SHA核对，不重跑goal-scene。

实际未来source生成后，独立publish到只含下列七个NPY的public目录：

- context_rgb_00/05/10：uint8[224,224,3]，实际新prefix帧；
- context_proprio：float32[3,4]，实际原生返回，仅agent xy/vxy；
- history_raw_u：float64[10,2]，仅真实prefix，不含参考suffix；
- history_normalized：float32[2,10]，精确原native normalize；
- goal_rgb：既有独立goalRGB原字节复制。

`project_blind`拒绝任何多余目录/文件（含witness/contacts/source合同）、缺项、symlink、
object/pickle array、错误shape/dtype、raw-zero替换normalized-zero或修改goalRGB。
归一化复用已封存producer.normalize与真实原生ACTION_MEAN/STD，不重新估计统计。
source witness/state/geometry/reference动作不得通过JSON或NPY额外字段进入planner。

即使数值文件格式全部通过，也只返回`unreleased_numeric_projection, ready=false`。
没有任何本模块路径可以签发source-ready/scientific-ready。真正native生成与严格consumer
的实际SHA仍缺，不能预造未来hash，也不能由RAM fixtures冒充。
投影只限制数据ABI，不是OS权限沙箱；MAIN后续部署进程只能挂载公共目录/投影JSON，
不能将私域source目录、witness或本prepare合同暴露给planner。新adapter须审查这条边界。

## 后续真实验收所需（当前不宣称满足）

两独立arm frame0..10的五文件逐字节gate之后才发suffix；原nativeRGB/state/proprio/
full-public-state/transition和commands帧0..20全部保存，prefix历史不替代。
逐suffix10观测callback contact points与ceil(points/10) returned_n_contacts分开；
记录block位置/角度/速度、agent位置/速度、原coverage/reward/done，不把agent惯性当物体能力。
检验push相对prefix及零控制的实际block响应；如果没有接触或运动，保留失败，不改初态/动作凑通过。
本合同不设新成功阈值，静态间隙也不是科学任务成功门。
原native done恒False，coverage>.95只是原注释而非done条件；任何task成功仍unknown/null。

预计两臂资源分别计：constructor/close2、seed4、reset2、public40、Space.step402、
render/get_obs42、setup/set_state/Space constructor2、goal-body82、Body constructor88。
预算128MiB accounted/96MiB payload/8MiB final reserve/2048nodes×8192B、900s、200MiB free floor。
这些是新generator将来必须落实的合同，不是本saved模块已观测meter。
须native intent-before/returned-after，失败不退款、partial不补步，close与hook cleanupfinally，
完整2分母及earlystop/engineeringfailure全保留；任何后续额外replay须单列新增合同，不偷增预算。

## MAIN命令与状态

cwd `/home/gbk/code/LeWorldModel/le-wm-main`，先读码/RAM，再纯saved prepare：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_agent_p5_contact_source_contract_20261028_v1.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/spectral_reachability/agent_p5_contact_source_contract_20261028_v1.py prepare --contract results/agent_p5_contact_source_contract_20261028_v1_MAIN01.json
sha256sum results/agent_p5_contact_source_contract_20261028_v1_MAIN01.json
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/spectral_reachability/agent_p5_contact_source_contract_20261028_v1.py verify --contract results/agent_p5_contact_source_contract_20261028_v1_MAIN01.json --contract-sha256 <ACTUAL_CONTRACT_SHA>
```

未来真的独立发布了公共NPY之后，MAIN可核数值投影（仍不release）：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/spectral_reachability/agent_p5_contact_source_contract_20261028_v1.py project-blind --contract results/agent_p5_contact_source_contract_20261028_v1_MAIN01.json --contract-sha256 <ACTUAL_CONTRACT_SHA> --public-dir <ACTUAL_PUBLIC_DIRECTORY> --output results/agent_p5_contact_source_contract_20261028_v1_projection_MAIN01.json
```

worker未实际prepare、project-blind或运行物理/模型；RAM只验证静态几何、数组ABI和failclosed。
旧全部封存。source generator与实际strict source witness是精确下一硬前置，不重复泛manifest。
