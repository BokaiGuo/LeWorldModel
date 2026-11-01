# Moving-history：四个新候选根的最小 source producer 协议

日期：2026-10-27。状态：**设计交付，未实现 producer、未 prepare、未运行物理或神经模型**。
本文件独立新增，不修改任何既有 sealed 源码、输出或总进度文档。MAIN 审查与批准后才可实现、
封存、实际执行。个人 goal 仅覆盖本协议，不是科学 P1 完成。

## 1. 要回答的问题与不变合同

从此前已打开的 calibration 来源中分离出固定四个新候选根，检查同一冻结模型、原始度量和
真实非零 history 的既有规划算法能否迁移；本轮 source 只构造合法任务，不运行或筛选 planner。
四个候选并不预先等于四个合格根。全部成功后才可称“四个新 source roots 已构造”；
有失败则如实报告 `k/4`，不追加候选补齐。新 seed、本协议或 witness 可达均不单独证明独立性、
direct difficulty、非贪心机制、强基线优势或 P1/P2 准入。

原 moving 三臂物理的 0/6 success 是 MAIN 已打开的校准结果，**不用于本协议的 seed、几何、
witness、history、目标、成功阈值或算法参数选择**。本协议不根据 planner modelcost/contact/
success 排根，不查看新 planner outcome 才决定保留谁；既有失败保留，不回填为新根结果。

固定环境 `swm/PushT-v1`，原 `relative=True, action_scale=100, dt=.01, control_hz=10`，
`render_mode=rgb_array, resolution=224, with_target=False, render_action=False`。
这不是 overlay 条件实验：不切 on、不改 goal_pose、相机、颜色或 renderer 变量。
goal 是真实 witness endpoint，不是默认绿色轮廓。后续只准入既有 frozen LeWM、native projected
CLS192、identity192 terminal **mean-square** Q；不造新 Q、readout、adapter 或训练。

模型身份继承原 moving encoder，未来 prepare 须重核并绑定原文件，不在 source 阶段加载模型：

|项|固定身份|
|---|---|
|checkpoint directory|`/home/gbk/.stable_worldmodel/checkpoints/pusht/lewm`|
|config.json SHA256|`2564086e961e7b5c7c04dffc451091115b389a590645ff19653c64fd0bc16e09`|
|weights.pt SHA256|`48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607`|
|frozen tensor SHA256|`4b12c01f5859f4b46bea478f3a8d5e2d3f516df456a622a3feec31bde7e33546`|
|原 raw normalizer 来源 SHA256|`0a4eb4592951139e596b35becede17d175df7b48a2fde16679ab0aac7c4dce34`|

normalizer 仍为原 float32 `mean=[-.0087,.0068], std=[.2019,.2002]`；不重复除100，
原训练 provenance 未因此获得新认证。native physical success 使用同一环境 `eval_state`，
另外保留 wrapped-angle 与 object/agent 分量诊断；不改变原生角公式或把 latent Q 当物理成功。

## 2. 一次封存的 seed、抽样分母与几何

slot 顺序和符号固定如下；不设置候补池、oversample、retry 或按难度筛选：

|source_slot|root_seed|reset_seed|witness sign|
|---|---:|---:|---:|
|0|203027101|203027201|+1|
|1|203027102|203027202|-1|
|2|203027103|203027203|+1|
|3|203027104|203027204|-1|

**抽样分母始终4**。每 slot 只有一个几何 proposal，`proposal=0, attempt=0`；
随机源精确为旧 `sample_state` 的 `default_rng(SeedSequence([root_seed,0,0]))`，
NumPy 版本与源代码 SHA 一起封存。每 replay 用同一 seed 重建同一请求，不是新增抽样机会。
不要求先找到四个“好根”再对外报告4。

保持原 recipe：object XY 均匀 jitter `[-16,16]` 各维、angle offset `[-pi,pi]`；
按旧 polygon 顺序均匀选 shape/edge，edge fraction `[.15,.85]`，pusher surface offset15；
其余状态初始化沿旧数学函数，state[5:]=0。禁止换形状、扩 jitter、缩角度、靠近目标或重抽接触边。
原始 recipe authority 是
`le-wm-main/results/p1_constructive_reachable_source_20261027_v1_prepare01.json`，
SHA `98ffb806018a270f5ef27ffa76be929393a74355c213bc274bf158c72d80b086`。
新 producer 只能显式取其中冻结的 recipe/witness/action 语义，**不能继承其旧 slots 或 freshness**。

第一次 reset 使用上表 reset_seed 得到 native original_goal；第二次保持原 seed0 placement reset，
`options.state=sample_state(original_goal, native_shapes, root_seed,0,0,recipe)`，goal_state 为
刚得到的 original_goal。seed0 是所有样本共有、显式设置 state 的原 placement 实现细节，
不是新根随机族；必须封存此例外且核对 setter 最终实际状态，不能用它绕过 root_seed/reset_seed
的排除。两 reset 都实际计账，不使用后验 poststate setter 复制 root。

如固定八个 seed 与已用/保留 seed 或相关 reserved span 冲突，准备阶段拒绝该协议的执行；
不自动换 seed、递增或扫描空闲号。另选 seed 必须 MAIN 事前批准另一版本，旧候选记录保留。
这里尚未证明八个 seed 全局未用，不能用“文档中未搜到”替代历史覆盖证据。

## 3. 历史排除：必须覆盖全部校准根，不只旧 moving 两根

未来 prepare 接收 MAIN 外部批准的 `exclusion_manifest_path/SHA256`，只含来源身份、
seed/span、原 root/observed-history 身份及证据文件 hash；不读 planner costs、选中动作、
物理 future outcome 或 witness 命令来决定源分布。清单需覆盖截至 prepare 的全部 PushT
校准/探索/已预留来源，包括失败或未被选中的候选，以及旧 zero、原四根、moving、其复制/overlay
和 feedback 派生根；继承分支应归并 lineage，但它们实际不同的 observed root 不能遗漏。

可复用 `p1_conditional_source_metadata_20261027_v1.resolve_history/as_exclusions` 的元数据投影
作为**基础索引**，不能修改旧源。其当前 `as_exclusions` 硬编码 `coverage.status=unknown`，
源码 SHA `13e5b65c149f56d7a0285d262f264dfe933b55a8366de0223e6724aedb3dacde`。
其白名单不完整覆盖新 moving prepared envelope、嵌套/压缩来源和全部 reset seed 字段，
不能把它的输出自签为“历史全部排除”。MAIN 须显式补充对应 metadata-only 投影、所有 prepare
预留、实际 sourcepack 和 observed 根来源，再给独立的范围明确 approval；未知项不得默认为新。
文件名目录 census 与已知范围/未知项一并封存。**未完成覆盖则 source run 不获 fresh-root 授权**，
不是先跑再把其结果倒签为独立根。

至少封存以下逻辑字段（新 schema，不冒用旧 resolver schema）：

```text
schema = moving_fresh_source_exclusions.v1
cutoff_utc, project_root, evidence_files[{path,sha256,metadata_scope}]
covered_source_families, used_or_reserved_root_seeds, used_or_reserved_reset_seeds
reserved_reset_spans             # 明确 half-open [lo,hi)
root_goal_sha256s                # 继承旧 root_hash 原算法
observed_roots[{source_id,root_state_sha256,physical_pose_key,
                root_snapshot_sha256,history_pixels_sha256,raw_history_sha256}]
unresolved_sources
approval{authority:"MAIN",coverage_scope,approved:true}
```

MAIN 的批准是外部 SHA 封存文件，不允许 producer 自置 `approved:true`。未知来源未解释或
缺少相应 observed state 身份时拒绝独立根授权。此文只定义接口，未生成该清单或批准。
不要求历史所有 planner 结果具有相同 renderer/Q；历史物理 root 一样就必须排除。

旧 `root_hash(state,goal)` 是二者拼接 float64 hash，**换 goal 不能把旧 state 变新根**。
所以除 root-goal hash 外还须离线对比：

1. 单独 root state7 的规范 little-endian float64 SHA（明确 signed-zero 原始 bits）；
2. 物理 pose key：state[:4] float64 原值，angle 规范到 `[0,2pi)`，零规范为+0；
   使用冻结 NumPy 余数实现。另外固定数值别名拒绝规则：四维 XY 每维差≤`1e-9` pixel，
   且 `abs(atan2(sin(angle_delta),cos(angle_delta)))≤1e-12` rad 即按同 pose 排除；
   这只保守处理2pi/浮点别名，不是任务难度/性能筛选，不允许按四根结果调整；
3. 实际 root exported body snapshot、RGB3 pixel SHA 和 raw10 SHA 的重复/lineage 记录。

相同物理 pose key 或符合上述数值别名规则即拒绝，不能靠 velocity、signed zero、角度加2pi、
渲染 flag 或 goal 标签规避重复；完整 snapshot/history 相同也拒绝。
哈希不同不证明随机统计独立或训练数据不重叠。
历史无法提供足够 root 身份时列具体缺失，不声称全局排除已完成。
四个新 slot 之间也按同规则去重；碰撞 slot 原地标 `excluded_duplicate`，不补根、不删行。
warmup10 后取得实际 root 时先检查 state/pose/history 身份，最迟在 witness dispatch 前拒绝
重复；construction 取得 goal 后再检查旧 root-goal hash。预检不拿未来成功/困难程度打分。

## 4. 每个 slot 的真实 source 时序与 witness 防泄露

每 slot 固定 `construction → replay0 → replay1`，每 phase 都做上述两次 reset 和独立真实
warmup10；不从 saved root restore，不复制 endpoint，不跨 phase 复用 physics state。
warmup 原始命令为 float32 前5 `[.1,0]`、后5 `[0,.1]`，世界坐标 raw，禁止 tanh/归一化后
dispatch/zero padding。保存实际 latest_action bits 与全部10次返回；RGB/state/snapshot
逐步 durable，部署历史严格来自新 replay0 的真实 observed0/5/10。

construction 的新 root 后，按旧 witness 的确定性几何规则生成一次未来10命令：
radial=(objectXY-agentXY)/norm，tangent 为其逆时针垂直；前5 logits `.5*radial`，
后5 `.25*radial + sign*.5*tangent`，先 float32 后 tanh；原每 packet logit L2 cap
`.75*sqrt(5)`。完整 logits/commands 在首次 dispatch 前 durable；不改振幅、角度或 horizon。
只有此离线构造器可用几何 state，不把它当 model planner。

construction 正常完整10步的真实终态定义唯一 derived goal，实际 terminal RGB 定义 goalRGB。
不向默认 goal_pose 注入 derived goal，不贴图，不从未来 H5 state 生成目标。若 construction
原生提前 stopped、越界或 guard 拒绝，则保存其 partial，不把提前 endpoint 改成替代目标。
goal 已在新 root 原生成功范围内则保留 `degenerate_goal`，不缩成功阈值或重造 goal。

replay0/1 各自从 reset 真实重建 warmup；所有 observed0/5/10 的 raw/RGB/state/exported
body snapshot（含 bits、force/torque 等）与 construction 一致后才 attach 同一个 derived
goal/goalRGB。attachment 不改 state、goal_pose、不多 render/physics；实际 setter 单独计。
各自执行同一预声明 witness，逐步与 construction 同步比较 actual commands/state/snapshot/RGB。
合法 native success 可在≤10步结束，保存其真实步数/缺测 checkpoint；禁止为了full10强行继续。
truncation或非成功终止也保留，不能宣称两 replay 可达。未导出的 solver cache 不作精确同一声明。

只有 construction 与两 replay 合同通过，且历史/批内排除通过，才发布该 slot 的 blind NPZ。
部署白名单文件严格只有：

```text
rgb_history:uint8[3,224,224,3]       # 新 replay0 observed0/5/10
raw_action_history:float32[10,2]    # 新 replay0 实际动作 bits
goal_rgb:uint8[224,224,3]           # 新 construction 实际 terminal20
```

不含 state、root native pose/hash、witness、certificate、候选动作、future images 或 success。
证书与 replay provenance 留在 evaluator 单独目录/清单；planner manifest 不给证书路径。
MAIN 另审 source 后显式审批 blind paths/SHA，encoder 部署只获该白名单，不能自行枚举 sourcepack。
task_id 在实际 frozen CUDA 编码后按完整 task 合同生成，此时不伪造 pending task 或 native-root ID。

## 5. 资源上限、失败分母与持久化

预算按四个 slot 都完整 construction20+两个 replay20 的**最大值**；replay 合法早停减少
实际资源，不补齐预算，也不减少抽样分母。root sampling/数组比较不是神经调用，但计入host时间。

|source-only资源|每完整slot上限|4slots上限|
|---|---:|---:|
|独立几何proposal|1|4|
|sample_state同请求重建 / witness重算 各|3|12|
|phase|3|12|
|实际reset / reset goal-render 各|6|24|
|reset-only state setters|12|48|
|warmup public / witness public 各|30|120|
|全部 public steps|60|240|
|实际internal space.step|612|2448|
|离线 goal_bind|2|8|
|native goal setter（reset+bind）|8|32|
|explicit renders / PNG（每phase0..20）|63|252|
|blind NPZ发布|1|4|
|model encoder/projector/predict/actionencoder/AD/训练|0|0|

internal physics 包含每 reset 两次 setter 各一步，以及每 public 十次积分：
`physics=2*actual_reset+10*actual_public`；`set_state=2*actual_reset`；
`goal_render=actual_reset`；`goal_setter=actual_reset+actual_goal_bind`。
全部实际 module/operation attempt 在调用前 durable、return 后 completed，不退款或合并重复。
前向/重放不能省账；无意外 poststate setter，第一次physics/render异常后不可自动重试。

固定 source prepare≤256KiB（可用有界 metadata gzip envelope、文件仅path/SHA引用），
run总cap16MiB含所有partial/PNG/NPZ/journal/final，final reserve512KiB；启动至少216MiB free，
每写/每 dispatch 保留200MiB。wall1200s全局硬cap，MAIN可另设外部watchdog，不在此启动。
完整 RAM mock 必须覆盖四slot全量252PNG/12phase、serialized journal、失败/partial/fsync；
若实际完整receipt无法容纳，需运行前新版本明确调预算，不能删图/receipt或压薄证据凑cap。

固定 report 四行：`source_slot,seed_pair,status,first_error,secondary_errors,phases,
counts,root_identity,missing_checkpoints,blind_export_or_null`；phase状态分别可见。
失败/重复/guard/native-stop/未开始都留在4分母；不能仅报告获批 blind 数量作为抽样分母。
每次新 frame 及原始 state/snapshot/commands 先保存，再作图像/接触/精确性 guard；原合法
native早停与工程异常分开。源失败终止当前 slot，可继续后续固定 slot；全局seal/资源/写盘
失败停止后续，仍保留其 not_started。close/restore/finalhash/fsync的次错不覆盖首错。

不要求 partial 达fullcap：无工程异常的实际 returns 用上述等式核对，完整12phase才要求完整
上限；对抛异常的操作保留attempt>completed及最后成功观测，不伪造失败瞬间状态。
只发布合同完整的 slot；若不足4，不把其当“四根全齐”，后续 planner 表必须显式列缺失原因。
可报告 source合格k/4；planner success只在未来真实执行后给出，source失败不得伪称planner失败
或从整体方法分母无声删除。无代码或运行产物时不填写实际k、SHA或成功数。

## 6. 后续最小接点与当前具体缺口

未来新 producer 建议独占 stem `p1_constructive_moving_fresh_source_20261027_v1`，自有
source/test/innerdoc/prepare/schema，显式四slots；不能调用旧 `build/execute` 冒用其旧两根。
旧 moving source SHA `9fb76ecfa054f878ab9778fc29c59bc708728b240dd2cbc3060b166e52622f55`，
只复用 frozen recipe数学、warmup/witness、snapshot和原生绑定低层；新 execute独立计账/发布。
所有import闭包、环境源码、模型/normalizer身份、本协议、exclusion manifest均prepare绑定，
run前后重hash，MAIN独占准备/实际运行。

旧 moving encoder/planner/physical 的 slot0/1、task path及sourcepack白名单均已冻结，**不会
自动兼容本四新根**。后续需新adapter明确source_slot0..3、四blind及外部SHA；不patch旧globals、
不伪装旧source/task SHA。若四slot均获批，最小encoder账是16encoder+16projector（每slot
RGB3+goal各实际一次，不因相同像素去重），0predict/AD/physics；这只是未来成本，不是已付。
标准原normalizer将raw10变成两time-major 5x2 tokens，history3无padding，原identityQ/H2不变。
之后算法/arm列表与预算需独立事前封存，保留所有新task，不用source witness作候选或初始化。

当前具体缺口：MAIN全校准根排除清单及approval未提供；八个新seed尚未全局冲突核验；
新producer/RAM tests/prepare/真实source尚未实现或执行；四slot encoder/planner/physical
adapters尚未准入。本文件可直接指导下一实现，但不解除这些前置，不声称已有新实验依据。
采用 experiment-design 技能的固定抽样/预算预声明；不采用调参、训练、按获胜结果扩样的通用阶段。
