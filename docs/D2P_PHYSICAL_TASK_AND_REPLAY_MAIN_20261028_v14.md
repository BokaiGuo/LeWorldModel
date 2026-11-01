# D2P 后续 P：真实任务复算与重放数值故障

承接 v13；本页记录 MAIN 实际执行/复核，不缩减原 C1–C3、三实验族、多环境范围。
工程通过、局部改善、科学准入分别记录；完整研究尚未闭环。

## P2：新四分支真实执行与完整严格消费已完成

输入是事前同144预测规则产生的全部四个 winner（两已知 moving roots × beam/spectral）。
不是 Full D2P 比较，也不是新鲜根确认。MAIN 审读 adapter、联合95项测试、实际 prepare/verify 后，
主机新执行四分支；旧标签未换绑。物理报告：
`le-wm-main/results/agent_p2_equal144_physical_handoff_20261028_v1_physics_MAIN01/report.json`，
SHA `50240e6f5e9bca90a5d8734cb5c41a5e06d61337b20dcd3d7fb9693fea45ebcb`。
状态 equal144_physical_recorded_not_admitted，errors=[]，7.52505秒，4完整 horizon、0成功、0早停。
资源：construct/close4、reset8、set_state16、goal_render8、goal_bind4、public80、Space.step816、explicit_render88。

MAIN 新独立算术封装实际 analyze→verify 均 exit0，7项新测试通过；持久化完整 native_strict_receipt：
`le-wm-main/results/main_p2_equal144_physical_metrics_20261028_v1_MAIN01.json`，
SHA `6b2e0301614d163cffba3d8eb2f1e522ebf6304c86d8c589eda3fe0469bfd413`。
全部40 plan 返回的当前 goal、raw command、native7 reward、native termination 直接从 state/goal 复算一致。
warmup40次发生在新 task goal 绑定之前，明确不拿新 goal 复算旧 reward。

| 已知根 | spectral−beam jointXY | objectXY | native7距离 | velocity误差 |
| --- | ---: | ---: | ---: | ---: |
| 0 | −18.996306 | −6.864757 | −11.001805 | −1.802160 |
| 1 | −7.795297 | −6.539378 | +52.751074 | +93.244246 |

两根位置误差均下降，但四分支都未达到原生成功。第二根7维距离恶化不能直接等同位置控制恶化；
反过来 objectXY 改善也不能替代 jointXY+angle 原生成功。
native7 的原始角差/目标速度保持原合同，没有更改阈值或拟合 Q。
每方法每根144预测匹配，导数成本不匹配；禁止宣称等 FLOPs、等总资源或 P2组合增量。

## P3：完整配对 MAIN verify 已通过，任务与混合状态分开

`le-wm-main/results/agent_p3_budget_paired_analysis_20261028_v1_saved01.json`，
SHA `55d5ab8c49d31b7d0af3ddcb803f114d272c220c6ab9524076e32df035223288`，
MAIN 完整源码审读、29项相关测试及实际 verify exit0。
48槽/24配对全部保留；18配对两臂有真实结果，6配对缺结果；41精确复用物理槽，5付费拒绝、2未到达。
只有两个已知根，不是18或48个独立根。

18已知配对的 adaptive−fixed 均值：native7距离−103.16424747，而jointXY +33.76931658；
objectXY +0.67957691，velocity −144.39652295，成功差−0.05555556。
不能将混合7维改善称作任务胜出。新分项分析正在沿原state7/goal7精确拆解，
latent Q没有已认证 native7 decoder，不能把其预测冒称关节/速度预测。
fixed/adaptive实际391/634预测；各46J/8832VJP，不是等实际资源比较，C3未成立。

## P1：有限公共动作库支持已完成 MAIN 验证，不等于机制闭环

完整40槽机制面板 MAIN 已审读、40项测试和实际 verify 通过。
公共动作库分析 MAIN 已审读、36项测试及实际 verify exit0：
`le-wm-main/results/agent_p1_common_library_support_20261028_v1_analysis_NASH01.json`，
SHA `89e218f59fdabd5b7a3568ca72ffcdd28e627c3328599eacb9642b64b87ec77d`。
保留40槽、160投影、所有5词与正负动作原始float32位；缺latent槽12不填补。
有限库上界只是 evaluation upper envelope，不能作为实际部署选择器或连续能力集扩张证明。
仅4/40槽观察到联合位置前缀变差而后缀变好，不能据此宣称非贪心因果或P1通过。

## P5：原生重放新失败已定位到覆盖率末位，尚未进入颜色/CUDA

MAIN完整审读原生 replay renderer 和新10图 encoder，58项联合测试通过。
新 prepare/verify 均实际完成；MAIN在主机执行原constructor/seed/reset和固定四词。
原五状态全部先过 baseline 才允许10颜色；完整预预算805次Space.step，不作零物理声明。

实际报告：
`le-wm-main/results/agent_p5_replay_target_renderer_20261028_v1_render_MAIN01/report.json`，
SHA `1b02ed4a2ca136b4dc7f8549073ff030620df8587d654eb2a07ca86036cad0c6`。
状态 partial_engineering_failure，wall1.695709秒。
四prefix的44帧五文件均逐byte通过；xy suffix到frame13时精确门拒绝。
此帧RGB/state/proprio仍逐byte一致，差异仅：

- coverage_arr[12]：新0.35743524812972466，原0.35743524812972455；
- transition reward：新0.3762476296102365，原0.3762476296102364；
- info max/final coverage 为同一个末位差。

实际control43/Space.step434/render47/construct4/close4；其他状态保持未完成，10颜色全未运行。
不能冒称85帧重放成功，也不能给新 encoder 伪造 strict_saved_complete。

MAIN查原生源码：pymunk_to_shapely 直接迭代 shapes 构造 MultiPolygon；step分别读取
self.block.shapes形成goal/block几何再算intersection.area/goal.area。
Pymunk Body.shapes返回set(self._shapes)，顺序不是合同保证。
这是有源码支持的数值顺序假设，尚需精确枚举重算才能判定根因；
失败保持封存，没有调容差、重试挑运行、修改环境或删除coverage字段来绕门。
新10图CUDA计划虽有代码/测试，但没有合法重放回执，尚不能执行；无CPU神经回退。

## P4 / E1 / 强基线后续前置

P4原reset ledger、loaded-byte及session终态归属缺口仍未解除，不打开保留根或伪签freshness。
Hi-LeWM指定compressed checkpoint member已从官方URL完整传输113,104,429字节，
仍只是raw-DEFLATE，不是可加载checkpoint；需bounded解码、EOF/CRC/decoded SHA、可信loader、
embedded-low/预处理/非零history适配。whole-archive作者MD5未认证，不能称官方逐byte等价。
Hi high-target实际adapter与独立资源账目继续推进；作者报告值与本地结果继续分栏。

后续P正在真实推进，不以准备文件代替完成：P2物理已严格验收；P3需要任务收益与同资源证据；
P5先解数值/重放问题；P1机制、新鲜独立确认、FullD2P增量、完整强基线/E1/E2/E3仍未闭环。
本轮未训练冻结模型、未删文件、未push GitHub。goal保持active。

## 同轮追加：Hi-LeWM 数据字节已实际解码验收

MAIN完整审读有界decoder源码/doc，59项测试通过、三件套SHA冻结后实际执行。
22.603077秒，errors=[]；decoded122,424,517字节，CRC32 `d59f865b`，EOF完整、无尾随数据；
现场decoded SHA `b87805747d40037841877ce7b99b7dda3ebe7a52202c0ba46bf0006ab5d6f008`。
新文件 `le-wm-main/results/agent_hilewm_bounded_deflate_decode_20261028_v1_MAIN01/decoded.unverified.ckpt`。
报告SHA `f399b23e07f37960127433d10e14c8c7b7eae6c75ff329accb75a52f701edcc9`，
实际consume也exit0，重新读取原compressed SHA及decoded SHA/CRC。
合并asset+decoder写入349,296,564字节，低于3GiB预算，未删partial/chunks。
这是标准库数据解压，不是CPU神经实验；torch/pickle/外来模块均未加载或执行。
来源许可、对象结构和可信加载、embedded-low与normalizer配对未因解压而认证，
baseline_runnable仍null。下一步只做有界zip/pickle opcode静态清单，不反序列化。

## 同轮追加：分项与几何顺序已完成 MAIN 严格验证

P3分项报告 `le-wm-main/results/agent_p3_task_resource_decomposition_20261028_v1_saved01.json`，
SHA `f1530e79023932eb287a5ae431edc503ec2510519b35354aacf64b7e9dce0acf`，
MAIN完整源码审读、26项测试、全量verify exit0。
18已知pair的adaptive−fixed均值分项：jointXY² +8805.147482、raw-angle² +0.004384、
velocity² −84401.097778，合计native7² −75595.945912。
9个已知pair表现为native7下降而jointXY上升，均可由速度项抵消解释。
这是同一已知集合的代数分解，不是因果改善、等预算收益或native预测解码。

P5单frame13几何顺序诊断 MAIN审读、17项测试及实际verify exit0：
`le-wm-main/results/agent_p5_saved_geometry_order_20261028_v1_saved_worker01.json`，
SHA `9d46896fc85ce2fb3e0c6a1f6e07c7cfd970631366e3a3fe0cdbb06bbfb3cd63`。
在按绑定二进制乘加顺序和当前libm重建系数的条件下，目标polygon顺序[0,1]得到
新coverage/reward，[1,0]得到原coverage/reward，均逐bit命中；block两种顺序均保留。
但历史cached matrix/world vertices/libm loaded bytes未存，legal_native_permutation_member保持null，
没有声称唯一根因或释放v1渲染结果。后续新观察器将直接捕获原生geometry返回，
不修改renderer函数、物理状态或旧字节门。
