# P2 full30 四槽物理 adapter：MAIN交接

## 范围及证据边界

实现批准协议 `agent_p2_two_program_full30_physical_protocol_20261029_v1.md`，SHA
`dcfc98c43431d09a74227ffc5d1ebaaf9b3edab144ac3387eef46d6170da35dd`。
仅本stem source/tests/本文新文件；没有修改任何旧source、四个full30 planner文件或结果。
本轮worker只有RAM/临时磁盘夹具及实际已存模型plan metadata ABI检查，无NN/physics/生产prepare/fullstrict。
fixture consume是合成环境回调产物的消费，不是生产fullstrict或模型parity证据。
experiment-code技能用于组织代码/测试；用户限定范围优先，没有训练、绘图或实验搜索。

新物理schema是 `agent_p2_two_program_full30_physical_20261029_v1.{prepare,execution_plan,report,consume,completion}`。
不伪造full15/旧模型receipt；不patch任何旧globals。复用原place/bind_goal/native_hooks/dispatch/capture和metric原语；
自有raw30、H30 episode、七cp、完整saved validator、Store预算及CLI。
固定root0 transfer/joint、root1 transfer/joint，共4rows、known2、fresh0。
`P1_admission/P2_admission/P2_mechanism/scientific_admission/physical_release=False`，即使工程complete也不改变。

## 实际三权威，不重做完整模型审计

source.PINS默认且严格只接受以下实际path+SHA，不能输入另一份“差不多”的模型结果：

|资产（路径相对inner根）|实际SHA256|
|---|---|
|`results/agent_p2_two_program_postcontext_20261029_v1_prepare_MAIN01.json`|`da1803815f14a1e3ee254e7915dc781d66367c7f96b011547c10f0736ba3f9c1`|
|`results/agent_p2_two_program_postcontext_20261029_v1_cuda_MAIN01/report.json`|`99f2b5edceae839bc3b0387ac93ef1d866a3fb74ec869ec7a30ceeaf4211ae00`|
|`results/agent_p2_two_program_postcontext_20261029_v1_consume_MAIN01.json`|`af2890ae4d330f69e8b01c7a144786ee454b08091e6d79ef63fa1b42e0bf6c5e`|

明确 `full_upstream_rebuilt_here=False`：继承MAIN完整full30 receipt的神经保存IO验收，
不是本adapter重新执行producer.consume。所有原prepare.bindings_sha256、report artifact_sha256及其完整目录inventory
均重新hash，包含源/模型/原任务与本次返回数组；不只保留指针。source/kernel字面SHA和批准协议另严格pin。
原native source_material保存文件重建原放置/warm/RGB任务身份；它不是NN或物理生产。
本adapter own三文件、native原语源闭包及原env源也封存，prepare后漂移导致verify重构拒绝。
prepare/run/consume重新执行自身material与全部binding核验；不会再次调用旧fullstrict。

check_planner_plan独立核每个witness digest、task/root/goal/model、source_programs、normalizer、
合法logits60、NumPy tanh64→float32 raw30bits、全部6个context的history shift和实际归一化动作token接缝、
最终192维identity-Q目标均方误差。原模型forward数值认证继承MAIN，不冒称本次再次神经验证。
完整4plan必须存在；丢失/损坏sealed export在prepare即拒绝，不替补。运行失败则仍保留4rows/null。

## 执行与预算

授权字面值 `MAIN_TWO_PROGRAM_FULL30_PHYSICS_ONLY`；无此授权，run在读取/建目录/导入gym前拒绝。
factory仅MAIN run路径导入gym并构造原PushT；prepare/verify/consume不导入或执行环境/torch。
每槽新env、两次原reset/place、warm10、原warm0/5/10 RGB与原native root/goal哈希守卫。
原task绑定后连续执行literal raw30，不额外tanh/clip/scale，control15不reset、不换状态、不重编译。
首个plan动作intent前，显式持久化native_history_guard：实际warm0/5/10 RGB像素SHA、raw10bits、
一次归一化history bits、原root native7/goal/snapshot、task与planner witness/root_context SHA。
消费者要求此guard位于三个真实warm观测及goal绑定之后、所有plan观测/动作intent之前。
这不声称新编码：latent root身份继承MAIN模型证据，实际原生RGB/raw历史在此重新验收。
原environment内部relative/action_scale100保留。goal pose/RGB/renderer/状态守卫逐step执行。
每warm0..10、plan0..实际终步采off/on/off，控制始终off；cp固定0/5/10/15/20/25/30。
时间：warm0/5/10为absolute0/5/10与control−10/−5/0；plan c为absolute10+c。
native7、每步reward/t/tr、actual/requested rawbits、snapshot、全部RGB和返回计量完整保留。

满额硬cap：constructor/close4、reset8、public160（warm40+plan120）、physics1616、
set_state16、goal_setter12、goal_bind4、goal_render8、explicit_render/PNG504。
physics=2×reset+10×public；PNG=4×3×[(10+1)+(30+1)]。
512总render=504显式+8reset；public/physics是嵌套轴，不相加冒称FLOPs。
新model encode/project/predict/J/VJP/backward/solve/SVD/train全部0。
原228predict/4J是已付继承费用，notmatched、非等FLOPs、非纯组合因果。

硬wall900s从verify之前起，到report flush/fsync并计算实际report SHA。
verify耗尽预算时不建输出目录、不dispatch。每次普通原生资源dispatch再检查wall/fault/cap。
final reserve路径不走普通wall拒绝，允许落盘partial和finally close；不能据此签超时complete。
report保存后计算SHA，再写completion.json（绑定report SHA/journal SHA和post-report wall）。
completion fsync后再次检查deadline；若越界/失败写terminal_failure.json，consumer因缺completion或额外文件拒签。
report.wall是report前采样；验收最终边界使用completion.wall，不能把前者冒充全阶段。
completion自身最后运输亦检查，超界只partial。若底层磁盘已彻底不可写，不承诺凭空生成report，保留已有文件前缀。

输出256MiB、final reserve8MiB、floor200MiB，不静默扩cap。
每PNG≤256KiB且≤504个=126MiB；journal≤64MiB；其他metadata/invalid arrays合计≤16MiB；
126+64+16+8=214MiB≤256MiB，留42MiB未分配，不能给单项自动借用。
504裸RGB共75,866,112bytes；不能沿旧64MiB宣称全图保证。
这是强制可拒绝上界，不保证任意PNG/异常数组能装入。超限保留partial，禁止缩图、删帧或少跑槽。
Store增量维护size账；journal流式hash，不每event重读全journal或扫描整个目录。
write在with内部flush/fsync，随后目录fsync；原native moving.raw的单LF和source.digest ABI保持。
consume独立磁盘census再核各组字节，所有未知/符号链接/空journal记录拒绝。

## 早停、失败和consume输出

原native t/tr立即停止，不padding、不拿缺cp后续邻帧补位。全部4rows及missing_checkpoints保留。
warm earlystop不能冒称原root已建立；original history不全则initial_exact=False。
每根两臂只能比较common time0..min(actual_plan_steps)，事件终态另标eventtime，不能将异时终态当same-time。
失败intent/failed/return、嵌套reset/set_state/physics和close均计账；失败不退款、不替根。
consumer重建全部动作/label/PNG像素/链hash/计量，未知event不filter；source前后hash守卫。
工程partial不是负科学结果，也不签all4完整成功；损坏守卫数据即使记录partial也可能拒绝认证。

consume公开字段：physical_report_sha256、prepare_sha256、plans[4]（包括commands及完整planner_plan）、
rows[4]、counts、metric_breakdown、completion、byte_census、bindings、saved_PNG_verified、
all_four_physically_executed（仅4槽各30步时true）、own_native_saved_reconstructed_here=True。
下游新RGB桥可直接逐槽按plan与cp引用join；本实现无RGB encoder。
本协议建议的28encoder/28projector需MAIN另授权新桥，不能据这里0neural当作已经编码。

## MAIN精确CLI

以下只交接，不由worker执行。工作目录 `/home/gbk/code/LeWorldModel/le-wm-main`。
prepare未传planner参数时使用上述严格实际PINS；六个参数亦可显式提供，但值必须完全一致。

```bash
.venv/bin/python -m pytest -q tests/test_agent_p2_two_program_full30_physical_20261029_v1.py

.venv/bin/python -m experiments.spectral_reachability.agent_p2_two_program_full30_physical_20261029_v1 prepare \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_two_program_full30_physical_20261029_v1_prepare_MAIN01.json
```

从prepare stdout取实际sha256填入下列PREP_SHA；不可填未来/示例hash。

```bash
PREP_SHA=MAIN实际prepare返回的sha256
.venv/bin/python -m experiments.spectral_reachability.agent_p2_two_program_full30_physical_20261029_v1 verify \
  --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_two_program_full30_physical_20261029_v1_prepare_MAIN01.json \
  --prepare-sha "$PREP_SHA"

.venv/bin/python -m experiments.spectral_reachability.agent_p2_two_program_full30_physical_20261029_v1 run \
  --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_two_program_full30_physical_20261029_v1_prepare_MAIN01.json \
  --prepare-sha "$PREP_SHA" \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_two_program_full30_physical_20261029_v1_MAIN01 \
  --authorization MAIN_TWO_PROGRAM_FULL30_PHYSICS_ONLY
```

仅MAIN后续明确授权物理后执行run。之后取实际report SHA：

```bash
REPORT_SHA=MAIN实际report_sha256
.venv/bin/python -m experiments.spectral_reachability.agent_p2_two_program_full30_physical_20261029_v1 consume \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_two_program_full30_physical_20261029_v1_MAIN01 \
  --report-sha "$REPORT_SHA" \
  --receipt /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_two_program_full30_physical_20261029_v1_consume_MAIN01.json
```

所有输出exclusive，新名字已存在即拒绝；禁止覆盖旧prepare/results。prepare三件套SHA冻结后MAIN再运行，防止窗口漂移。
当前未完成：MAIN真实prepare/verify/physics/consume、未来独立28槽RGB桥及实际误差分解；不填未来receipt SHA。

## 冻结协议实际权威补记

原协议dcfc98c4…35dd不原地修改。其第1节称prepare文件名为“拟用MAIN01接点”的句子已被MAIN实际交付纠正：
`/home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_two_program_postcontext_20261029_v1_prepare_MAIN01.json`
确为实际文件，SHA `da1803815f14a1e3ee254e7915dc781d66367c7f96b011547c10f0736ba3f9c1`。
本adapter严格pin该绝对path及SHA，不将它视为未来文件或可替换默认值。
