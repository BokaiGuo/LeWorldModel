# D2P：静态渲染隔离与同预测预算对照

本页承接 v12 的真实执行证据。原 C1–C3、P0–P5、三组实验及多环境范围不变。
完整研究仍未闭环；工程验收不能代替非贪心能力改变、物理任务收益或独立确认。

## 已核实的输入

P3 新预算主机 CUDA 及完整交错预算/48槽 fullarray 消费均完成：
report SHA `066b2f7f6e37fc9c4ed2795a07387c473da7e2e4418414e05ce0e6080ddc2978`，
strict SHA `14c87cd5c69562a53cacc7d3082a4938fcb776b814c9fc2b00dacd99df20bd0d`。
41 compile/5 paid refusal/2 not reached；fixed/adaptive 实际391/634 predict，各46 Jacobian/8832 VJP。
保持 P1/P3 admission=false；物理交接正在基于精确同源/同状态/同命令身份实现，旧轨迹不能无条件换绑。

P1 全40槽两路主机 CUDA 的对应终态、原生buffer均逐bit相同。旧诊断误将三行native预测与
shifted context比较，已由新saved语义消费者更正；不能用误比字段声称旧预测漂移。
12/76已知组内反转尚在，4未知pair保留。下一归因将真实latent和原生物理指标明确分层。

## P4 原运行证据：已实际恢复，但未解除认证缺口

MAIN完整读码、19项测试及真实extract→consume均通过。报告：
`le-wm-main/results/agent_p4_original_evidence_extract_20261028_v1_report_MAIN01.json`，
SHA `32f166efb9d66c2c6e5e3467a0973d147e816bffe615f3370bd00049a663fb73`。
10个dispatch保留；B前两次direct return恢复13+9=22个traceback frame，各自有绑定日志前缀、
record SHA/字节区间/JSON pointer/文本区间。不是22个独立根或reset，也不是原loaded bytes哈希。
reset ledger/count仍null，reset_blockers_cleared=0，不打开保留根、不签approval。
后续仅沿明确session_id和原call_id继续恢复异步返回，不按邻近文字猜归属。

## P5 静态渲染诊断：事前合同

固定四词终态与原goal，全部五个状态先复现原色baseline，再做target LightGreen/White十条件。
只允许原生constructor→_setup→公开静态render-view属性恢复→原生renderer；
reset/_set_state/Space.step/控制步/神经调用均0。五个baseline的原始NPY/RGB须与封存字节完全相同，
任一失败则保留失败，不进入新因子图、不补图、不改容差。

原snapshot的current_time_step=0.01，而新零步Space该只读值=0；该差异须显式报告，
不声称完整动力学恢复或private contact cache恢复。实际渲染前后公开状态/RNG必须完全相同。
agent隐藏无已认证原生开关，仍拒绝；本轮只隔离合法target颜色因子。
原色推断错误已纠正为with_target=True→LightGreen，但最终仍以baseline真实复现为准。
这不是planner新评分、重新校准或任务收益实验。

## 新强基线执行设计：同144预测，而非等总计算

此决定在新CUDA/选择结果之前固定；旧144 vs 88配额结果和源码保持不动。
使用现有完整两moving任务，全部四root/method槽，不按效果挑根；仍是known-root诊断而非独立确认。

| 方法 | 固定候选/搜索规则 | 每根predict | 额外导数成本 |
| --- | --- | ---: | --- |
| Kinodynamic | 原20个首包、beam6、120个后包；原全局选择及4次replay调用 | 144 | 0 J/VJP |
| 无组合谱 | zero1 + DLS3 + top16×2原半径×2符号，共68个候选，stable全库nonlinear Q选择 | 144 | 1 J/192 scalar VJP/2 SVD/1 bounded solve |

谱方法预测数为nominal2+AD forward2+候选136+独立winner whole/segmented4=144。
top16在20维动作空间中合法；不追加padding、不重复胜者凑预算、不事后选模式。
goal/Q、动作域、normalizer、真实历史及模型不变；本协议不包含训练、Q拟合或阈值修改。
原编码成本单列继承，所有新模块attempt/entry/return/sample/AD真实计账。
新loader必须绑定实际反序列化的同一config/weights缓冲区与哈希，而非只检查路径前后哈希。
神经执行仅主机CUDA，模型冻结。每job900秒属于预先声明的工程守卫，不是性能优势。

本设计只匹配world-model预测次数，不匹配导数成本、FLOPs或wall时间。
不得把两基线互相比的结果冒称Full D2P胜出；完整强基线集、Full D2P、闭环物理及独立确认仍需完成。
新runner/tests/consumer由独占实现线准备，MAIN审读后才执行。

## MAIN 后续实际终态（覆盖上文“正在/待执行”快照）

### P2 同144预测：已完成主机CUDA与完整消费

MAIN完整审读新源码与文档，21项runner回归通过；真实prepare/verify后执行HOST CUDA，
24.955642秒、4/4槽全部plan_sealed、576 predict API及576 predictor samples，errors=[]。
模型哈希前后一致；谱方法另付2J/384VJP/4SVD/2bounded solve，不能称等总计算。
实际报告 `le-wm-main/results/agent_p2_equal144_baseline_cuda_20261028_v1_cuda_MAIN01/report.json`，
SHA `7eeaa059d58895545340458455ddcaabceda4a813a41051e1a2c60f22bc10228`。

MAIN完整consume检查全部候选、原beam树、逐packet模块输入输出、192行J、全bank稳定选择、
独立双回放、实际加载同buffer及完整源闭包；持久回执/描述分析
`le-wm-main/results/main_p2_equal144_saved_20261028_v1_MAIN01.json`，
SHA `d8ded78d152938a3c6e5c244729f120d040c42666b44d9ed89bf2e0beca029dc`。
另3项saved封装回归通过。

| 已知任务 | beam模型Q | 谱模型Q | 谱减beam |
| --- | ---: | ---: | ---: |
| slot0 | 0.57598809 | 0.67832599 | +0.10233790 |
| slot1 | 0.66564188 | 0.53187054 | −0.13377134 |

两根一好一坏；这只是模型目标，不是物理任务成功。两臂所选动作各不相同，后续新物理adapter正在实现，
不能换绑旧物理标签。FullD2P对比、正式组合增量和独立确认仍未完成。

### P3 新预算到物理：41槽精确复用，不重复支付

MAIN真实strict_saved prepare与完整verify均完成，
`le-wm-main/results/agent_p3_budget_physical_handoff_20261028_v1_prepare_MAIN01.json`，
SHA `5c39919c632fd8b7feb8b22ad310900e424d5bcb226f1afd99a6a5b89b01fc9a`。
全48槽：reused_saved41、paid_refused5、not_reached2、pending_new_physics0。
只有同槽/同祖先/完整native状态/goal/context及float32 raw5逐位相同才复用。
没有新增物理调用；历史执行费不算新支付。复用面板仍仅2个已知根，不能冒称48独立样本。
后续任务是固定/自适应全槽配对结果与真实资源分析；compute_advantage与C3准入仍false。

### P4 异步原日志：已实际extract与consume，未解除认证缺口

MAIN完整审读v2源码/doc，27项回归通过；真实extract→consume完成。
报告 `le-wm-main/results/agent_p4_original_evidence_extract_20261028_v2_report_MAIN01.json`，
SHA `0155c1dbfbab61830baddae73a26c85c3d0d166cef9286c2a7e32046ac4ee77c`。
3个父dispatch、8个session全部存在literal poll输入；10条batch terminal返回未带可唯一归属的session，
故逐session terminal仍unknown，不按返回顺序猜。reset_blockers_cleared=0，未打开保留根。

### P5 静态渲染：两层工程失败均保留，下一步改原生重放

v2实际失败在错误地要求pygame也位于overlay，尚未construct/render/physics；
已在新v3绑定封存base_site pygame准确路径、版本、实际import SHA，未修改旧封存源码。
v3的13项回归、真实prepare/verify通过后，MAIN实际静态运行7.411620秒。
依赖来源已通过，但第二状态原baseline逐byte门拒绝，未进入任何颜色因子：
`le-wm-main/results/agent_p5_renderer_factorial_20261028_v3_render_MAIN01/report.json`，
SHA `76f8488c852572960730f8a11082e2636636f0e0b27884766f7b7ffbd7328c93`。

MAIN独立saved差异分析及verify/5项测试完成：
`le-wm-main/results/main_p5_static_baseline_diff_20261028_v1_MAIN01.json`，
SHA `0dfe56b6c34d94dd1986fe9181bfd5c8e09611473716293bdbf5f26a0c869711`。
xy原图逐byte一致；yx仅3像素不同，bbox=[88,120,88,122]，原图含红色接触点。
两状态全部公开属性/RNG在渲染前后相同；其他三状态未到达，保持unknown。
实际reset/control/Space.step/神经均0，10颜色条件全部not_run。

原生DrawOptions默认开启DRAW_COLLISION_POINTS，颜色(231,76,60)，原native draw_dot会画圆。
公开snapshot不保存arbiter contact cache，故这是有源码支持的差异假设，尚非完整因果证明。
下一新协议用原constructor/seed/reset+固定20动作真实重放，保留live cache，再要求全部五baseline
逐byte一致后才做颜色因子；预先新增物理预算5reset step+80×10control substeps=805。
不修补像素、不隐藏碰撞点、不放宽baseline门；真实重放不能继续称零physics。
该renderer和后续P2物理adapter分属独占实现位，MAIN继续负责源码审计与实际执行。

本轮未训练模型、未CPU神经回退、未删除文件或推送GitHub。原C1–C3/三实验族范围与科学门不变。
