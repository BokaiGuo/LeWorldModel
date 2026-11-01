# Constructive GN/D2P：模型计划—实际物理saved诊断

仅新增此stem的独立分析器、tests、本文及小型诊断报告。不修改任何sealed producer、模型、规划器、
physical源或既有结果；不导入producer/core/环境/模型，不调用任何verify/episode，不运行神经或物理，
不拟合Q/readout/能力模型，不新设科学阈值。experiment-code用于小型持久化报告和失败路径测试。

## 显式输入、访问边界与接口

三个根输入必须由调用者提供明确path及完整SHA；本次实际输入为：

| 输入 | 项目内路径 | SHA256 |
| --- | --- | --- |
| source report | results/p1_constructive_reachable_source_20261027_v1_main01/sourcepack.json | 9ad3a414fd2148258524d4a7e9d4f070f59532ec2e5943cfa61f9145b3a41bd9 |
| planner actual report | results/p1_constructive_blind_planners_cuda_20261027_v1_cuda_main01/report.json | 4dd5823e9a273b820d6b7b3f6a36a4a5a19b193383eaa9cbf94442082e79d98d |
| physical actual report | results/p1_constructive_plan_physical_20261027_v1_main01/report.json | d205850e95f639b649fba9b5c6ee8059f2507431753a538b1ce30eccbd234300 |
| 可选selected RGB actual | results/p1_constructive_selected_rgb_cuda_20261027_v1_cuda_main01/report.json | c77a5d208acb813ad968afe24ef0c547f08a83ff9968a984b29502baf0dda770 |

source JSON只投影两根root SHA和blind文件SHA，不打开其中certificate/blind路径。
实际root/goal state7来自已SHA绑定的physical report，并与journal goal_attached逐项核对，
再由root+goal float64 bytes复算身份，和source report的root SHA一致。
planner后续输入仅为其report hash manifest绑定的四plan、四result及plan显式指定SHA的task export。
physical后续输入仅为report绑定journal、journal中selection显式指定SHA的handoff bridge以及fresh事件
指定SHA的PNG。没有目录发现，没有读取source construction/replay未来帧；通用Reader在读取前拒绝所有NPZ/NPY、
certificate/witness文件名。source JSON中证书路径字符串不等于打开证书，更不把witness交给任何selector。
唯一NPY例外是显式selected receipt授权的8个`image_XX.native192.npy`：专用入口同时绑定固定index、
manifest与row SHA、float32/192维、≤4096bytes、finite、无pickle／尾随数据；不放开通用NPY读取。
selected的prepare只比对images配方，不遍历其其他bindings路径；binding/prepare/events都核对SHA。
48项ordered image/module journal核对8encoder+8projector，逐图绑定task、root/goal/predicted数组、
physical selection digest及原fresh checkpoint/PNG；并未重新证明CUDA内核或重跑模型。

`run_analysis(source_path,source_sha,planner_path,planner_sha,physical_path=None,physical_sha=None,reader=None)`
另有尾部可选参数`selected_path=None,selected_sha=None`（必须同时提供且有physical）。
是纯saved入口。未传physical时先完成source身份与模型计划检查，返回
`awaiting_explicit_physical_report_and_SHA`，没有physical_rows，不制造零值物理结果。
实际分析须同时传physical path+完整SHA；错误为`unverified_or_inconsistent`，移除可误用的物理比较。
部分物理现场可返回`saved_partial_physical_diagnostics`，完整固定四分母不丢失；不能拿最后一帧padding
成未执行的step5/10。提前terminal同理，终态不冒充固定时域比较。

文件逐项hash后读取、结束再hash；单文件≤8MiB、gzip单流展开≤32MiB、总读取≤64MiB。
CLI小报告≤200KB，exclusive新建/fsync，写前保留200MiB。没有模型大文件复制／加载、tmp或下载。

## 独立复算与验证范围

模型侧按task真实latent独立复算initial、prefix5、terminal10的identity192 Q-MSE：
GN用final_prediction的两段context；D2P用完整10candidate中的冻结beta1、stable-index winner。
逐候选重算stage、nonlinear terminal及stage+own-V算术，核对winner和plan命令。
own-V只核对已有保存值与分数的关系，不重新拟合ridge，不冒称本分析重新认证其全套优化算法或AD。
模型调用resources从实际report继承，非本次新执行或完整独立CUDA审计。

同task/goal/model-contract/blind-source必须对齐；核对identity Q、H2、5×2 packet、float32 tanh动作、
每包logit球、3states/2between-action token，十条raw zero按原float32 mean/std归一化，不重复÷100或
改成零padding。动作球1e-7及cost复算1e-12容差仅用于saved浮点舍入核验，不是机制epsilon或科学margin。

物理侧核验完整2314事件hashchain、run_started/finished、branch_started/closed、嵌套API
attempt/completion及账本，selection seal／bridge／planner own commands逐条相等。
所有88张PNG均重hash bytes/pixels、224RGB、状态snapshot、guard身份；每条实际plan返回的命令、
goal/goalRGB身份、native success/distance/reward/termination独立复算；guard拒绝或terminal后不准续走。
report的plan0/5/10 checkpoints必须是journal里对应accepted观测，不能只信汇总字段。

四臂共80public（含每臂10warmup）、8reset、816physics、88explicit render、8goalrender、16setter、
4goalbind，attempted=completed；四个horizon_completed，native successes=0，errors=[]。
PNG88、事件2314和全四分母是本次saved核验，不增加真实样本或神经计数。

每个step5/10输出：物体相对root的净位移、signed/absolute wrapped净转角、原生distance/success、
object position error、wrapped object angle error及其相对initial变化。
rotation是首尾wrapped净变化，不是累计路径角度；object_success=null，不擅定物体成功阈值。
原生success仍为agent+object前四维位置norm<20且原生角项<π/9；native distance/reward另含state7速度。
保留负原生角项bug与正确wrapped诊断，不修改success；本批未遇到负原生角项。

## 实际四分支结果

root0 initial物体位置误差94.668030px、角误差0.640151rad、native distance270.693092；
root1对应149.181105px、0.810623rad、295.033599。

| root/arm | step | 物体净位移px | wrapped净转角rad | 物体位置误差px | 物体角误差rad | native distance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0/GN | 5 | 46.367136 | 0.347847 | 49.292891 | 0.292305 | 282.029582 |
| 0/GN | 10 | 48.294484 | 0.383101 | 47.976995 | 0.257051 | 152.662136 |
| 0/D2P | 5 | 31.556453 | 0.261309 | 63.653480 | 0.378842 | 201.833340 |
| 0/D2P | 10 | 30.704421 | 0.163836 | 64.251279 | 0.476316 | 181.747486 |
| 1/GN | 5 | 0.000000 | 0.000000 | 149.181105 | 0.810623 | 448.181599 |
| 1/GN | 10 | 24.027057 | 0.203353 | 130.929152 | 0.607269 | 492.696370 |
| 1/D2P | 5 | 93.558004 | 0.433081 | 68.241341 | 0.377542 | 179.382488 |
| 1/D2P | 10 | 132.970057 | 0.316205 | 45.398509 | 0.494417 | 453.385088 |

root0 terminal GN物体／native均优于D2P；root1 D2P物体与native均优于GN，但两臂native distance
相对initial反而上升。所有八个step5/10的官方success均false，不把连续误差下降升级为成功。

### 首段短期变化与非贪心边界

GN step5 native变化分别+11.336490、+153.148000；D2P分别−68.859752、−115.651111。
D2P两根物体位置误差也都在首段下降。模型prefix Q变化：GN为−0.338452、+0.037223；
D2P为−0.484289、−0.903743。
更关键的是：D2P两根winner分别index2、index7，**均等于自身同库immediate greedy**，stage差与
own-reach gain相对greedy均0。因此没有把这两条D2P选择称为严格非贪心重配置的依据。
GN的短期native变差也不自动变成D2P机制证据；它可能只是agent/速度和物体进度分离。
分析仅输出非进步的描述性符号，没有新epsilon；能力改变明确`not_measured`，不是false或成功。

### 模型—物理比较：分开目标，不制造latent误差

模型terminal Q-MSE为root0 GN0.667756 vs D2P0.682555，root1 GN1.291969 vs D2P0.176435。
两根terminal模型相对排序与native/object位置误差相对排序均同向；这不证明预测数值准确。
root0 prefix5模型Q和native均偏向D2P，但object位置误差偏向GN，属于不同评价对象排序分离。
不能将Q-MSE减物理px/rad，或把native distance上升称模型latent预测error。
未传selected接口时`latent_prediction_error`明确不可计算。本次已接MAIN真实编码receipt，
从保存数组独立计算以下同一identity192 Q及预测误差，不调用encoder：

| root/arm | 真实Q5 | 真实Q10 | 预测对真实MSE5 | 预测对真实MSE10 |
| --- | ---: | ---: | ---: | ---: |
| 0/GN | 1.082647 | 0.825542 | 0.106013 | 0.147491 |
| 0/D2P | 0.912043 | 0.856466 | 0.069927 | 0.118847 |
| 1/GN | 1.412312 | 1.537921 | 0.057846 | 0.191100 |
| 1/D2P | 0.489668 | 0.385148 | 0.339586 | 0.258235 |

两根在两个边界的GN/D2P同Q排序**均未反转**。root1 D2P真实Q从1.461798降到0.385148，
仍native success=false：不是“模型排序反转”的充分证据，代理Q改善不等于官方位置／角条件达标。
root1 GN的prefix预测Q略恶化但实际略改善（相对initial的方向反转），这和两臂相对排序是不同问题。
实际latent误差非零可量化，但不能单凭它确定接触／context偏差是唯一原因。

## 优先排查的可证伪解释，不跳到理论失败

1. **错根／错goal／错臂／动作被替换**：先检查root+goal bytes、blind/task/selection/bridge及每条返回命令。
   本次这些saved绑定未见不一致；不复用构造witness充当模型动作。若将来任一不符，归协议错误而非方法负例。
2. **动作单位、归一化、time-axis或history错位**：task数组已核对原mean/std、时间顺序、float32 tanh和
   states3/actions2；没有发现这一保存数组路径的错误。运行时实际送入模型的tensor仍依赖封存代码／receipt，
   本分析不以JSON一致性声称穷尽所有实现bug。
3. **评价对象混用，可直接数值证伪**：native distance包含agent速度误差，success不含速度。
   分析保存distance平方的agent_xy/object_xy/raw_angle/agent_velocity四项分解，及allxy位置norm。
   root1 terminal GN速度误差447.410723，D2P450.145352；因此物体更近仍可伴随state7 distance上升。
   但0success不能仅归咎速度：D2P root1 allxy仍54.101344>20且角误差0.494417>π/9；
   root0 GN allxy134.548334、D2P109.636820，同样不满足官方位置条件。没有重写success。
4. **原生角差bug／fresh guard／渲染失配**：并列原生与wrapped、核验PNG和逐步guard；本批无负角项。
   保存记录支持的部分可排查，未另重渲染或重新积分，不称环境实现全排除。
5. **imagined-prefix与实际边界不一致**：当前planner是open-loop，suffix编译基于预测prefix，physical
   实际step5并未送回重编译。可证伪检查是另获批准的实际step5 RGB编码，对照预测prefix与own history；
   本次已量化实际step5/10 latent误差，但未做observed-context重编译干预，不能把误差当作因果能力证据或宣称复用了错J。
6. **Q代理与任务目标差异、接触响应误差、优化覆盖不同**：都是尚需干预区分的解释，不是已证唯一根因。
   使用共同真实same-prefix probe、独立fidelity测量和强预算对照，不能调Q/阈值或按actual挑winner。

GN与D2P不同计算预算，四次GN更新不是plateau证据；本批2known roots、4branches不是4个独立根。
见证构造成功属于数据构造器，模型两臂均未实现官方成功，不据此宣布冻结模型不具备整个任务族或P1理论失败。
所有输出始终P1_admission=false、independent_confirmation=false、new_fits/neural/physics=0。

## 测试与MAIN复跑命令

30项测试：纯算术／Reader firewall与actual saved回归／内存故障注入；包括重hash后改actual action、
native label、goal、API账本、PNG、全分母、partial到step4不padding和缺physical等待接口。
fixture全部在RAM，无tmp目录、无模型/物理导入。actual不存在时saved专属项明确skip，不冒称通过。
新增selected完整8分母、实际Q/MSE、SHA、缺图、模型身份、数组字节、metrics、PNG/checkpoint、
重封journal API篡改及专用array路径拒绝。测试通过不替代独立确认、非贪心和真实能力改变门槛。

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
.venv/bin/python -B -m unittest discover -s tests -p 'test_p1_constructive_model_physical_diagnostics_20261027_v1.py' -v
.venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_model_physical_diagnostics_20261027_v1 \
  --source-report results/p1_constructive_reachable_source_20261027_v1_main01/sourcepack.json \
  --source-sha 9ad3a414fd2148258524d4a7e9d4f070f59532ec2e5943cfa61f9145b3a41bd9 \
  --planner-report results/p1_constructive_blind_planners_cuda_20261027_v1_cuda_main01/report.json \
  --planner-sha 4dd5823e9a273b820d6b7b3f6a36a4a5a19b193383eaa9cbf94442082e79d98d \
  --physical-report results/p1_constructive_plan_physical_20261027_v1_main01/report.json \
  --physical-sha d205850e95f639b649fba9b5c6ee8059f2507431753a538b1ce30eccbd234300 \
  --selected-report results/p1_constructive_selected_rgb_cuda_20261027_v1_cuda_main01/report.json \
  --selected-sha c77a5d208acb813ad968afe24ef0c547f08a83ff9968a984b29502baf0dda770 \
  --output results/p1_constructive_model_physical_diagnostics_20261027_v1_audit01.json
```

重复复核请省略output读stdout，或用独立NEW别名；不覆盖audit01或MAIN结果。
CLI完整诊断exit0；等待physical、partial或证据不一致exit2，状态与局限必须连同数值一起解释。
