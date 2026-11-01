# D2P 研究完整闭环审计

2026-10-29 P3三层最新实测：五同context U/Z2均预测/真实编码收益同向；根1两对原七维负收益来自编码→物理指标断点，位置改善而速度恶化，不能归结为本批预测排序反转。完整10分支/5配对/4logicalC/8比较/3no_witness已MAIN全审、43测试、actual build/wholeverify，SHA `36b31985a27a16156afb3dea8c852b84199ea1814cfcba986c5b889db39bb01c`。不同于P2另批已证预测反转，不作统一归因。C双对照与原成功条件仍未通过；六组相关联合192测试通过不等于科学闭环。C3最终冻结25项独立全套正在MAIN运行，HOST接线尚未完成，不提前签通过。完整结果与并行分工见[最新两节](D2P_DOWNSTREAM_PARALLEL_MAIN_20261029_v2.md)。

2026-10-29 并行续推最新终态：CEM完整8位置三层分析已MAIN全审/28测试、实际analyze与wholeverify（SHA `6ed50b44e306004360281bde781acb4d0207d25426dc7cfcd94b8a2dff14a729`）：4个模型可比根内时间对均与实际编码同向，实际编码与native7的9个可比对仍有2个反转、与jointXY全部同向。不是同时间方法或独立泛化比较。E3输入resolver已MAIN全审/63联合测试、实际prepare/wholeverify，known2×两goal×两horizon×两domain完整16ready/0blocked（SHA `b44699e6a10bb4c257ed8872dc86b99a37508d7022850f8b4fcd257bcd232b26`）。输入ready不代表GD数值runner或Full D2P实验完成；production授权和所有科学门仍false。五独占线程继续P3在线运输、完整三层分析、公平预算、E3数值runner和E1可执行差距；MAIN验收与实际执行，神经实验无CPU降级。详见[续推实际记录末节](D2P_DOWNSTREAM_PARALLEL_MAIN_20261029_v2.md)。

2026-10-29 最新实测：P3十分支原生全轴summary已MAIN71联合测试、actual build/wholeverify（SHA `e59eeadc6e6de711e60e50709162c4310ce0e69e9b3fdb8a5db838fd74797b19`）；根1两组U位置改善但速度误差增大，导致七维G负，不能称所有轴均无效或改指标放行。P3十张真实cp15也已MAIN46测试、主机CUDA10encoder/10projector及完整consume（SHA `946ca6cc017fc66591920eca7b7d1d42bc07e8d9ebf169c0b67ecfb0b8c30cc9`）。生产前捕获并修复底层只允许8图而实际需10图的计数接线bug，没有把它记成模型负结果。CEM七张实际图也已MAIN33测试、主机CUDA/完整consume（SHA `0e5d8736e07ffd07417b39c84471ebd68682e4bfb448e4459517d9a5fa3234fa`），保留8槽/1早停缺项。完整三层独立分析与在线C3、公平对照、E3输入继续并行，科学门仍不变。证据见[并行记录末两节](D2P_DOWNSTREAM_PARALLEL_MAIN_20261029_v2.md)。

2026-10-29 P3阶段C最新终态（覆盖下方“十条未部署”快照）：MAIN全文审读/39测试、实际prepare/主机原生十条U/Z2及完整saved consume均完成，receipt SHA `4e7d35ec52dba109c9e2c181ad46d754f44bc0c0ef41a362c30a68776c66f3b1`。十条full15、51/66守卫全通过，250 public/2540 physics/810 PNG、无工程错误，但原native success均false。原reward距离收益G：r0 C/Z/P为+68.2938/+70.1048/+52.1646，r1 Z/P为−132.0210/−275.2491；C未比双对照更好，原即时退步合取仍false。不将该诊断外推成理论否定；真实cp15编码及独立全轴saved分析继续推进。E3 known2描述性16格输入准备已批，不等于Full/fresh资格。完整证据见[并行记录末节](D2P_DOWNSTREAM_PARALLEL_MAIN_20261029_v2.md)。

2026-10-29 强基线新增终态：bounded CEM H3 原封存 raw15 已 MAIN 全审、51测试、主机原生部署及完整 consume（receipt SHA `1e69a2fe082af3c5da32ec28ac8ba10b2375b5a593fa0bf69a60367d7cbc9b86`）。root0 在control10原生成功早停、cp15保留缺失；root1满15未成功。45 public/458 physics/147 PNG，errors=[]，0新神经/训练。只有known2、不是等总计算比较，不能外推科学胜负或P1成立；真实RGB桥待执行。P3十分支/在线C3与独立泛化仍未完成，主线程及五独占任务继续并行。详见[实际证据记录](D2P_DOWNSTREAM_PARALLEL_MAIN_20261029_v2.md)。

2026-10-29 P3独立能力算术终态：七signal/五U的完整J、70奇异值、原bounded解与全k前缀贡献已MAIN47测试、实际build/wholeverify，SHA `88587a2d9f610df775c321eda6497bf3602c6ccd993b8779d7f36c6ad4744a4d`。C−Z同时间R仅−.00025633559208626533、C−P为+.2408989815722094；局部R/预测收益不构成真实能力证据。原C双对照即时退步合取仍false，actualcp15 U/Z2与G待阶段C真实部署。未重新训练、未CPU神经降级、未放宽科学门。

2026-10-29 三层分析最新终态：P2 12策略/84位置/84配对已MAIN30测试、实际analyze和wholeverify（SHA `83483d13bae02bbdf8a8affecfc0c31c9a695b7db447a6cdffe0a5ba80359c8b`）。12个同输入O-I后缀配对为9tie/2同向/1反转；base0/cp25预测→实际编码反转，cp30预测与编码同向改善但native7恶化。两个断点不能合为统一offset，亦未证明代码全无问题或理论失败。实际五端点full66物理见证也已MAIN41测试、build/wholeverify完成（SHA `2351ba5b05c91fb192ae4542e6839ab58b56bcd974170b94994246f61981f346`）；阶段C真实U/Z2部署及C3完整在线生产仍未完成。完整归因与证据见[并行记录末节](D2P_DOWNSTREAM_PARALLEL_MAIN_20261029_v2.md)。

2026-10-29 当前最新终态（覆盖下方“阶段B测量尚未运行”等快照）：P2 新24张I/F cp20/25/30实际图像已MAIN43测试、主机CUDA24encoder/24projector与完整consume完成，receipt SHA `972688100781340072a62fa5cfae353005a746ec24d713df4af293f0846834ba`。P3阶段B亦已MAIN29测试、主机CUDA7J/1344VJP/24predict/7solve/5U/10replay/22Q与完整consume完成，receipt SHA `682a78366b10a60895c4530f2fc6cc401e6439b625f4ba0a8aa996dc2fed79a7`，五真实动作witness已导出。阶段C十条U/Z2未实际部署；五U的模型Q下降不等于物理收益或机制通过。C3完整在线调度、P1、独立泛化和整套研究闭环仍未完成。详见[最新实际产物与SHA](D2P_DOWNSTREAM_PARALLEL_MAIN_20261029_v2.md)。

2026-10-29 并行追加终态：P2 原生摘要 v1 的跨 O/I/F selection_digest 强等错误已通过独立 v2 修复，MAIN 65 测试、实际 analyze、saved whole verify 完成；全部12槽/84位置/84配对保留，结果 SHA `bcc292165eb1dffc84a7db5b1cb478be0455f64256a4061ed8c270bc2eacc0d6`。它修复分析身份域，未改变原物理结果或科学门。C3 native Catalog 接线 MAIN 18 测试完成，仍缺完整在线/磁盘生产链；五路下一阶段实现继续并行，阶段B能力测量/阶段C探针尚无新增生产结果。详见[最新并行验收记录](D2P_DOWNSTREAM_PARALLEL_MAIN_20261029_v2.md)。

2026-10-29 最新终态（覆盖下方“suffix部署尚缺/五端点编码下一步”等历史快照）：P2 control15 八条 I/F 实际原生部署、四份完整 saved consume 与12槽join全部完成，320 public/3232 physics/1008 PNG，errors=[]。base0 新后缀 jointXY 改善但 native7 恶化，仍无 native success；另六槽 incumbent 的真实端点保持原 O。P3 五个实际cp10端点也已完成 MAIN 主机CUDA编码与完整consume（5encoder/5projector、无新predict/AD/physics），为七signal/五U能力测量提供真实三帧输入，但测量和十条probe尚未运行。全部仍 known2、科学门不变；P1/强基线/组合机制/等预算调度/独立泛化及三组核心实验仍未完整闭环。实际SHA与验收见[并行记录末两节](D2P_DOWNSTREAM_PARALLEL_MAIN_20261029_v2.md)。

2026-10-29 后续并行最新实测：[六目标真实执行及后续 P 线程验收](D2P_DOWNSTREAM_PARALLEL_MAIN_20261029_v2.md)。本轮六个 full15 与四个 full30 原生物理分支、P3 割线模型交付、六槽 24 图及 full30 28 图主机 CUDA 编码、secant first5 原生部署及各自完整 consume 已完成。secant 保四组：一组51帧精确守卫后真实部署、三组 no_witness 零派发。共同资源调度与固定反馈对照继续推进。六槽全 24 配对的模型→真实编码严格反转为 3，真实编码→jointXY 为 1；两已见根诊断及动作执行不等于 P1/P2/P3 科学准入。

同日追加：六目标三层向量分解及 full30 三层全表均已 MAIN 全审、独立测试、实际 saved build/analyze 与整份 verify，不再是待交付。full30 的 12 forecast 配对中，模型→真实编码/真实编码→jointXY/真实编码→native7 分别反转 1/2/4；各有六个 tie，cp0 模型保持 null。P3 同时间 Z/P 原生对照首切片已批准实现；P2 control15 反馈与 C3 routing 仍待最终工程验收，不将其测试或协议算作真实收益。

同日再追加：P2 control15 反馈 v1 真实 prepare 捕获 set→JSON 比较器代码错误；保留 v1 后，独立 v2 已修复，经 MAIN 51 测试、真实 prepare/verify、主机 CUDA 全 216 predict/8 J 和完整 saved consume 全部通过。8 新 job/12 O-I-F 槽保留，无训练/CPU 神经降级；其中仅 base0 transfer 的两 job 在自身输入下模型 Q 改善，其余六保 incumbent。真实 suffix 部署尚缺，不能将不同 context 的 I/F Q 直接当反馈收益或科学准入。新分析/负对照/反馈 v2 四组联合回归 188 passed / 15.96s，详见上方并行验收文档末节。

同日 P3 同时间对照追加：四条 Z/P 已 MAIN 全审、64 测试、实际 prepare/verify、主机原生执行和完整 consume 通过；80 public/816 physics/264 PNG，四个 prefix51 exact、四个 control10 端点齐全、0/4 native success。候选 C 在 root0 同时间略好于 Z、明显差于 P，不能满足预先双对照即时退步合取。下一步五真实端点编码及统一主动 probe，不能用 cp5 或预测状态替代。C3 routing/static/DiskCAS 组件已交且正在独立验收，仍缺完整12槽生产接线。

**2026-10-28 P1固定归一化干预已完成**：MAIN完整审读/42测试、主机CUDA328预测、
完整strict通过，public全164context/objective逐位复现。archive统计反转34→36，
2修正/4新增，24真实零响应保留，终态平均MSE约降0.25%不等于排序收益。
strict SHA `95d7d183dd98a4edd7df0bcd8750fc8cede2270939da04c32e16aa661813a8d0`；
MAIN新独立5测试及原blob/真实Q复算SHA
`664540c0ceb34ae8a98ecd1e02ea7c75da7e4012f50f8253ff5966636f7ca015`。
本替换不是充分修复，未知训练scaler仍未认证，不改部署统计，不归为理论失败。
详见[实际结果与仍待排因项](D2P_FIXED_NORMALIZER_RESULT_MAIN_20261028_v1.md)。

**2026-10-28 P1锚点/响应与Hi-LeWM实际CUDA定位**：固定80对的oracle共同
锚点替换后仍21反转（原34中19保留、另2新增），24真实零响应保留，不将回顾
oracle当部署修复；新工具9测试通过。Hi-LeWM已有五路layout对照经MAIN34测试、
主机CUDA、完整consume及直接NPY独立复算完成：source append/stack逐位一致，
legacy slice/reshape首步投影失配，连续化恢复、反向stride复现，旧3473元素/
1.49e−7失败精确重现，证明本固定参考转写错误而非理论负结果。
v2 report SHA `8f19346c99aba5c3cc5dd6efec878356938c24bcb53023444c37791f94ff6771`。
修正四项v3现已MAIN完整审读、36测试、主机CUDA和完整consume通过，22次high
predictor、8顶层dispatch，四case逐位一致，wall3.205秒。strict SHA
`6151e93c40fc1c5b053b81f8ac73a8dca7644a8a037492686a50a2718e24ddc6`。
不认证完整基线，不转授P1根因。
真实高层模型+CEM随后也已MAIN39测试、主机CUDA及完整strict通过：12真实cost/
168候选行/48 high predictor，receipt SHA
`7dfb4b49d552a6248da8b603e027a95c541c5db95af4fab777828ed412886e43`。
合成integration不是任务收益或完整Hi-LeWM baseline；P1与完整研究未完成。
详见[本轮证据、修改与完整缺口](D2P_ANCHOR_FACTORIAL_AND_HILEWM_REPAIR_MAIN_20261028_v1.md)。

**2026-10-28 raw10完整排因终态（覆盖下方启动快照）**：固定两known根×off/on，
164模型预测/164真实分支/164真实终态主机CUDA编码及各自完整strict全部完成。
MAIN原数组80对独立分解SHA `703142be3073336d76cd8cfb9cd8ef8fc41d0c3be0dd7141cc222a635e52c061`：
34目标方向反向、45固定baseline误差项相消、80非精确共同offset；四baseline
state/snapshot/PNG精确复现。新增可观测性分栏SHA
`1411a87e95d36ae84971b27e2bbaaed848fbc5c277e9675d591d374125614d3f`：
24不可见且均pose改变、22同向、34反向，0同RGB编码不一致；不可把24零响应
计为支持。新证据分类与矛盾拒绝已实现，五文件43测试通过；非历史P1根治。
没有新独立根，不用真实梯度进入选择器，不改gate；训练血统仍未认证。
详见[完整产物、修复及剩余问题](D2P_RAW_COORDINATE_CAUSAL_RESULT_MAIN_20261028_v1.md)，P1/P2/P3仍false。

**2026-10-28 scalar方向排因增量**：MAIN独立复算旧固定FD32项，8项模型/真实
编码方向反向、25项误差相消；不是原80/76面板，也不是新raw10 cohort。10测试
通过，新增分析器schema接缝已修，未证明历史损害来自该接缝。新164 raw坐标
对照仅协议固定、实现进行中，未报执行完成；P1/P2/P3科学准入仍false。
[复算与边界](D2P_SCALAR_FD_CAUSE_MAIN_20261028_v1.md)。

**2026-10-28 真实观察后缀的完整新增闭环已完成**：主机CUDA重编译72predict/32scalar reverse、全新物理80public/816physics/264PNG、四张新真实终态CUDA及各自完整strict均结束。MAIN再次完整重建并独立读原NPY分解（SHA `55ee1bef2ef77b16fe2153a4d2eecfa494c909da3b82ef2c514bfd56eac24d22`）：同actual cp5下模型Q全降，但root0两条件实际编码Q反升；root1两条件编码Q改善但angle均损害。jointXY三优一劣、angle全损害、成功0/4，仅2known根。代码方面跨schema JSON hash误用已最小修复、step6前exact cp5守卫已实测通过；不是历史负结果的统一根因。held-old2实际4predict干预的误差影响约1e−8，不能解释本批约.01–.17差异。四对仍误差相消、无精确共同offset；未证明理论不可能，P1/P2/P3仍false。七文件最终241passed、diff-check通过。详见[实际新增证据、修复与剩余断点](D2P_OBSERVED_SUFFIX_INTERVENTIONS_MAIN_20261028_v1.md)。

**2026-10-28 真实中点与新增标量导数实验完成**：固定四条同动作cp5真实history3已主机CUDA/full strict（SHA `127c650eace2c41e43dc0361aa13009981acfc8e9509dfff7a1f924283f87521`）；MAIN原数组独立重建后endpoint三优一劣、goalQ预测绝对误差两优两劣，不能统一归为imagined前缀。前两旧观察重编码约1e−6漂移保留，held-old2诊断另行实现，不伪称last-only干预或控制收益。新scalar路径固定240预测FD亦主机CUDA/full strict（SHA `71486805c02bad9cdb977ee833019dc410f3a64e06d0950140215b5e145aaf45`），MAIN独立字节复算全部120对/12向量一致，未见粗大导数接线错误，不等于物理正确或P1通过。真实观察重优化生产adapter正在审读测试，尚无新动作收益。详见[实际完整结果](D2P_OBSERVED_SUFFIX_INTERVENTIONS_MAIN_20261028_v1.md)。

**2026-10-28 下一项区分干预已预封存**：固定动作真实cp5条件预测、真实观察后独立后缀重编译及新增scalar-gradient的240请求FD正在独占实现，尚无新CUDA或物理效果。MAIN纯saved分析协议已排他封存SHA `d421793a24ff9c38789f9baa419ff183ecbb36d2bd0082884f313ef763461a18`、14测试通过。现有168候选四槽均有模型先退步且终态优于root的程序，但均不及全库最佳直接进展程序；不能称全无非贪心候选，也不能强选类别宣布C2。详见[真实观察后缀干预](D2P_OBSERVED_SUFFIX_INTERVENTIONS_MAIN_20261028_v1.md)。

**2026-10-28 原allmode真实编码配对已补齐（覆盖下方待认证快照）**：MAIN完整重建旧/新RGB consumer，认证历史image9/15/21/27与原allmode22/22/23/23的raw10/task/goal/step10 snapshot/PNG及目标192坐标，无额外神经或物理执行。新非线性suffix四槽均预测Q改善，实际编码三优一劣；其中两条实际编码改善却jointXY变差，另一条预测→实际编码排序反转。四对均存在误差项相消，不符合精确动作无关偏移。结果SHA `f2665a3c4f500f80f55da2f8b5519edf91fdb2203b3727bfcdc10b6c1281ed49`，仅2known根；P1/P2/P3仍未科学准入。详见[完整排因](D2P_SELECTOR_CAUSAL_FACTORS_MAIN_20261028_v1.md)。

**2026-10-28 目标表示独立对照终态**：同完整物理goal的双renderer目标集合A已完成全1056既有候选saved analyze与从真实来源完整重建verify（SHA `639fc9d15ea4ab02bddd8428f547acbc081d3c06cb324032fd45c651777508d1`）。原single-goal非线性terminal winner在八panel全部不变，只有两panel的active目标分支改变；没有动作选择first-stage效应，不重复相同物理执行冒充新收益，不放宽原P1合同或临时追加目标。下段“analyze进行中”现由本段覆盖。MAIN最终八文件288 passed /50.70秒、diff-check通过；详细结果与尚缺原allmode编码配对见[完整排因](D2P_SELECTOR_CAUSAL_FACTORS_MAIN_20261028_v1.md)。

**2026-10-28 最新完整非线性suffix对照**：已实际完成固定168候选3376predict/1344scalar reverse主机CUDA（334.264秒）、全strict、新四条full10物理（80public/816physics/264PNG）及四终态CUDA/fullstrict；对应回执SHA `337330eda5e8f191deec615422cae53e0c260364f63db0cd14d58218033ff019`、`2add0915ecd4d1fa16d72d77a44e1db5cc4d41b1f1961e1ad06e216fa76119d4`、`18d281427d34c1afce2054308208e7a9561df24a5d117cc851a5d143f17939e4`。全部168原prefix模型Q能改善，但四winner仍非模型先退步；相对原allmode jointXY一优三劣、angle三优一劣、成功0/4，仅2known根。原ridge近似确实限制模型目标，但优化改善不能替代任务收益。首次v1因遗漏8个内置模型类源码绑定、candidate0前failclosed，已用独占v2补source hash修复并保留失败，不算模型负结果。旧allmode22/22前两终态编码来源仍须认证，不能借mixed51/51替代。双renderer目标集合A已实现审读/37测试，生产saved analyze进行中，不改变原Q/P1门。详见[完整证据](D2P_SELECTOR_CAUSAL_FACTORS_MAIN_20261028_v1.md)。

**2026-10-28 最新真实RGB配对（优先于下方历史快照）**：旧surrogate四终态主机CUDA与完整strict已补齐（SHA `4f40159b636dc5252ce2c24a6be29f2374c7850b3723f81e8b062f745b8ebec8`），whole-row继承认证无误。MAIN独立原NPY配对显示三条改变动作均预测Q/实际编码Q同向下降，但jointXY两条损害、angle三条损害；本批问题不能统称预测排序反转，编码目标与物理任务存在独立缺口。相同动作控制行state7/实际latent精确一致，不能外推旧80对。结果SHA `b740fe3e086d8e49c6bb4b0c7dbd3016d1f5761194060bba8991384bb022c67d`；固定42prefix非线性suffix对照继续，不改gate、不训练模型。详见[最新配对与完整排因](D2P_SELECTOR_CAUSAL_FACTORS_MAIN_20261028_v1.md)。

**2026-10-28 最新选择器与强基线终态（优先于下方历史快照）**：MAIN对mixed222全部888候选精确分解，舍入影响至多3.28e−9、有限步非线性代价差至0.356，原stage+ridge与完整terminal witness在3/4条件选不同动作；新selector全1056候选槽分析/verify完成，但全部winner仍不满足相对root预测先退步。新mixed四条全10步物理及完整264PNG strict已完成（SHA `0f644f89d5864f818805ce95b663d19d4a5d094b65e8f320ef30e3ef3491e8d4`），jointXY一优两劣一同、改变的三条角度全损害，新旧成功0/4，不能称换评分已解。真正RGB反馈gradient MPC也已主机CUDA+完整strict（SHA `09b3a3a5479a21e16959d94989f4da5fd36c97597e89d930d2591fbe8a22620d`）：root1第10步原生成功、root0完整20未成功，重复槽不算独立样本，仅1/2known物理根成功。它补齐强基线且反对“模型完全不可控制”的笼统解释，不证明P1/C2/C3。新终态RGB配对与非线性suffix候选对照继续；详见[完整排因、实际产物与边界](D2P_SELECTOR_CAUSAL_FACTORS_MAIN_20261028_v1.md)。

**2026-10-28 最新三层终态**：zero8真实终态主机CUDA编码及完整strict已完成（8encoder/projector、模型未变、errors=[]，strict SHA `1dcd057eeaf06f9f5156fe27ae7c44811e19a2c48fcd375af5e5a607cba3a1a4`）。MAIN从三组原NPY独立重算32预测/实际Q：同批anchor对照仍2/8预测→实际编码排序反转，无guard下编码→jointXY仍3/8反转；zero guard改变的一条真实Q下降而角度/速度损害。两层缺口同时存在，不以保旧或改锚点声称根治。P3消费者加固亦已正式strict（SHA `84a007329c7813fde4c2ed332e233e7d8cf4afabb5216f2da58381f02855104c`），全量event census与20张预处理逐bit通过，原任务损害与科学门保持。下方“zero尚未编码”属保留历史快照。[最新完整证据](D2P_FACTOR_ISOLATION_MAIN_20261028_v16.md)。

**2026-10-28 最新物理消融终态（优先于下方历史快照）**：zero侧8条完整10步主机原生物理及full strict已完成，160public/1632physics/528PNG、errors=[]、成功0/8；strict SHA `9c125a32268899ece92144d95382529aa2a2f406de36cfd88c05828d4e5ac00d`。MAIN以实际dispatch动作位串及state7独立重建32cell：无guard时incumbent anchor相对zero的jointXY为5优/3劣；有guard为4优/4劣；guard仅zero index4改动作，XY改善但角度/速度损害。四设置成功均0/8、仅2known根，不把模型保旧等同物理安全或P1成立。zero终态编码尚未完成。训练归一化manifest及verify完成：四处mean/std位串一致，但checkpoint历史训练身份链仍缺，不按结果换统计。首次prepare因最后补测导致SHA漂移而在任何物理前拒绝，保留失败后重生成MAIN02，没有跳过封存检查。[完整产物及归因边界](D2P_FACTOR_ISOLATION_MAIN_20261028_v16.md)。

**2026-10-28 新增干净优化消融已实际CUDA验收**：同一提案公式、相同四alpha下，仅比较zero/incumbent线性化与保旧开关。缺失32预测已主机CUDA完成（8.673秒、模型未变），原数组/完整journal/输出census及32cell严格消费通过，SHA `8a59a189feaedf3b215b7aa9245c3b027228e2a51cfef32a47df286c664e5d89`；MAIN无实验模块导入独立复算32个NPY Q与32个argmin/rawbits全部一致。incumbent线性化模型Q为6优/2劣，guard仅在zero侧index4改变动作，不能一概归因为guard收益。新zero侧物理反事实尚未执行，不能据模型分数宣称控制收益。renderer/goal匹配80对saved诊断也已严格完成：渲染排序反转7→4，任务失配仍在，不是P1预测反转17→4。完整结果与修改边界见[v16页末](D2P_FACTOR_ISOLATION_MAIN_20261028_v16.md)。

**2026-10-28 当前核验终态（优先于下方历史快照）**：保护性v2的CUDA编译、实际完整10步物理、终态RGB编码和完整strict均已完成。MAIN独立原NPY复算：8/8模型Q与真实编码Q均改善，jointXY 7改善/1变差，angle仍有3损害，新旧成功均0/8；terminal strict SHA `a66d4187cdbf714509a8bc931d12ef6c51410603da8c4445c7884bbde2e143e0`。P1新40分支的日志信封/阶段goal验收错误已在新saved v4精确修复并通过全量消费，新80对仍有17预测→实际编码排序反转，不能以验收修复声称科学通过。全480次raw有限差分主机CUDA及独立24矩阵复算完成，本批无粗大导数接线证据。P3同反馈完整20步实际执行及原strict完成，虽整体distance下降，object位置和angle均退化，成功0/4，不支持C3收益。全部为2 known roots校准；P1/P2/P3仍未科学准入。具体修改、完整SHA及归因边界见[v16最新终态](D2P_FACTOR_ISOLATION_MAIN_20261028_v16.md)。

**2026-10-28 保护性优化实际新结果**：v2在合法原incumbent处求导、固定候选模型Q选择后，全部8计划模型Q下降；实际完整10步物理及full strict现均完成（strict SHA `7e71cf504691cb1b132c437267319f3291a62b0f6c9f839f0652b42a974f8de8`）。MAIN独立state7复算：jointXY 7改善/1变差、velocity 8改善、angle 4改善/3变差/1相同，成功仍0/8、仅2 known roots。不能以模型目标下降或校准改善打开P1/P2。P1新40分支执行完成，但consumer捕获日志信封比较与warmup目标绑定时序两处代码错误，新saved修复中，尚无严格通过结果。P3同反馈20步新驱动已审读/54测试通过，主机实验已提交。完整证据见[v16最新追加](D2P_FACTOR_ISOLATION_MAIN_20261028_v16.md)。

**2026-10-28 新反馈控制已真实执行并完整验收**：8条suffix重编译HOST CUDA及完整10步物理strict完成，回执SHA `c191b8076469f78fb7e02c793e0b061bcf6706266b1af6e90f3944748db2e11d`。MAIN原state7独立复算：jointXY 6改善/2变差，objectXY/angle各5改善/3变差，成功仍0/8、仅2 known roots。另定位零点局部ridge后无条件dispatch缺乏模型下降保障，3/8连同条件模型Q都比原suffix差；新优化版本须保留合法incumbent，不改旧合同、不以模型下降当P1通过。P5目标面积规则仍3/6任务排序反转；P4真实gzip-envelope schema错误已新v2修复，MAIN56测试通过，非旧P1根因。详情及原报告/独立算术SHA见[v16追加](D2P_FACTOR_ISOLATION_MAIN_20261028_v16.md)。终态RGB桥准备推进中，未闭环。

**2026-10-28 分层排因与新反馈CUDA终态**：原contextual8计划的24次真实0/5/10编码+8次同suffix预测已主机CUDA/完整strict完成，SHA `3d3d020c113959aec25f4e310b8512b814f7a78976f2bd7d5e58e0e435027f4b`。MAIN原NPY独立复算：8root重编码MSE=0、7/8终态预测MSE下降、6/8绝对Q误差下降、四槽预测→实际Q排序3/4→4/4；但物体角度仍2/4反向、原生成功0/8，无新控制收益。P1三项结构正式analyze/verify与P5全256patch同时间桥均完成；P3 full48第一包核账完成，不能替代完整20步策略。实际反馈suffix重编译与P3完整闭环驱动推进中。详见[v16分层因素、证据及修改](D2P_FACTOR_ISOLATION_MAIN_20261028_v16.md)，研究总门不放宽。

**2026-10-28 同时间任务桥CUDA已完成**：固定四词真实frame20+goal共5次主机encoder、无新predict/physics，完整strict SHA `542f2717e3860903d9ea87577edef2725a1fd7ae919c5aa216b45b75c0a29684`。MAIN原NPY独立复算精确一致；6无向对预测→真实编码1反转、真实编码→coverage4反转，排除frame15/20混用后度量失配仍在。一个已见接触根，不是4独立根、不互换moving成功合同、不改gate。详见[frame20任务桥](D2P_FRAME20_TASK_BRIDGE_MAIN_20261028_v1.md)。P2实际0/5/10 RGB桥在实现，保留确认四槽不借校准模式打开。

**2026-10-28 P2新完整执行终态**：480候选CUDA生成的8条full10动作已主机真实执行并完整strict验收，160public/1632physics/528PNG，errors=[]、native成功0/8。strict SHA `cc2de96eb6569319228644b62afe45218d0e4df1261c2bdabf6c0a9b052ad818`。composition相对joint的四维XY变化四槽为+21.93597/−46.83959/+23.81956/−59.74539；条件依赖，两个已见根，不称P2科学成立。下一步同条件0/5/10真实RGB编码归因，详见[v15终态](D2P_NATIVE_GEOMETRY_AND_DOWNSTREAM_MAIN_20261028_v15.md)。

**2026-10-28 P1优先实质归因**：同动作同时间正式分解及第二次完整verify均已exit0，完整上游闭包重建，40槽/80对原位保留，SHA `f723c5eb73ecebd0615a2ded0aa4b741c6f132ca5f6063c673f12d5e98539143`。12/76预测排序反转不是算术符号误差。MAIN独立原生distance轴分解发现36个真实编码→distance反转中34个涉及速度项翻转pose排序；但object XY/角度仍22/33个反转，不以整体XY的4反转掩盖。53联合测试通过，指标/gate未改；详见[P1同时间归因](D2P_P1_SAME_TIME_MARGIN_MAIN_20261028_v1.md)。P3完整strict已终态通过SHA `e138a33d175e1de850264c41f1df459354d77b1cedc7d3a3b0b50ed09c1e327c`，不等于C3收益。

**2026-10-28 同时间归因更新**：MAIN独立复现P5窄诊断（9测试通过，saved SHA `f7d89438ab63a23e6badc9c29daddfb3d7411ea4b59f003155a3291107f8100b`）：goal改色未改变实际像素，frame15仍为共同动作前缀，不能据此声称encoder颜色不变性或full-word终态差异。新frame20固定四词准备已派发。P3新8条真实物理完成、errors=[]、native成功0/8，完整strict合并28复用仍进行中。Hi-LeWM真实CUDA两direct逐位一致、多步rollout出现1.49e-7差异，保持失败并追查布局/执行路径，不放宽容差、不宣称基线完成。P1同动作同时间T3/T4分解实现继续，科学门不变。详见[v15最新追加](D2P_NATIVE_GEOMETRY_AND_DOWNSTREAM_MAIN_20261028_v15.md)。下方运行状态为保留历史快照。

**2026-10-28 P1理论复核**：[条件定理、正反例与T0–T6判据](AGENT_P1_THEORY_GATE_ANALYSIS_20261028_v1.md)已交付并由MAIN完整审读、独立有理数算术复核。P1机制可存在，但局部SVD/ridge与现有准确性假设不能推出当前强gate必过；选择条件、预测排序误差、latent→物理任务桥及独立机制/强对照须分别验证。非P1科学通过，不改阈值或回填历史。P2新480候选/8plans严格saved验收与完整verify均已完成，SHA `578c5bd93394ae9c3863cb420afe4cfd0c59633fbbea503b4012c029211e063f`；P3新物理交接为28精确复用/8待执行/10拒绝/2未到达，主机run已提交、终态待核。详见[v15](D2P_NATIVE_GEOMETRY_AND_DOWNSTREAM_MAIN_20261028_v15.md)。

**2026-10-28 P5新终态**：新live raster十图主机CUDA及完整strict均已完成，10次官方encoder/16项交叉评分，模型未训练；strict SHA `594ae118088c9c02f7de671d456f396dd8dd0d79765297fc4d889f42fed8776a`。固定指标分析尚待，不以编码完整性宣称P5/C2通过；新frame15与旧coverage frame20不混作同时间任务真值。P3新配额物理交接已MAIN审读、40测试通过，实际prepare进行中，复用/新执行数待真实产物确认。以下“编码已提交”保留为旧快照，详见[v15实际证据](D2P_NATIVE_GEOMETRY_AND_DOWNSTREAM_MAIN_20261028_v15.md)。

**2026-10-28 v15 主线程实际新增**：P3新成对配额主机CUDA及全数组strict完成，两侧各634predict，36compile/10paid-refusal/2unknown；strict SHA `3b62bc4cbf669ecbea10f5ecbf9863a1d96d2432ce029b75059c4edbc2838e6f`，非等FLOPs/C3收益。P2新自身预测上下文组合/无组合四槽480候选CUDA完成，双newline日志消费错误正由saved-only新版本修复。P5主机434步几何观察器及605步/85render新颜色协议均完成，前者strict通过；后者RNG时间接线错误经新consumer严格解释单次ctor draw后完整验收（SHA `f9919c23924b41d118d3da038e7cc6f0e3f88b291d36f83df6fccaeb910671c8`），十图CUDA编码已提交。旧renderer失败保持，历史等价仍不认证。P1模型封存选择的40槽同位回顾评价严格通过；Hi-LeWM完整解析代价CEM CUDA接线及850数组安全字节导出/verify完成，可信loader/真实模型CUDA仍未完成。详见[v15实际证据与待办](D2P_NATIVE_GEOMETRY_AND_DOWNSTREAM_MAIN_20261028_v15.md)，不以工程验收替代科学闭环。

2026-10-28 续推更新：MAIN实际完成P1全40槽/80路线主机CUDA及完整saved消费，public/cached原生buffer逐项完全相同；原排序12/76反转保留。诊断字段曾将shifted context与三行预测误比，MAIN原数组复算确认40个对应终态亦与旧终态精确相同，此误比不构成模型漂移。P3预算执行层39项、baseline resolver15项联合54测试，P1 API44项、P4补证需求25项、P5空间归因17项MAIN测试通过；P3新主机CUDA334.89秒及完整48槽原生数组/交错预算strict消费均已通过，41compile/5refusal/2not_reached，回执SHA `14c87cd5c69562a53cacc7d3082a4938fcb776b814c9fc2b00dacd99df20bd0d`。fixed/adaptive实际391/634predict，不宣称等实际计算收益。P4需求请求实际落盘且保持blocked；baseline恢复部分控制证据并完成逐字段比较但未matched。新增P1语义/P2比较48项MAIN联合测试通过。详情与精确SHA见[v12续推记录](D2P_DOWNSTREAM_NATIVE_VALIDATION_MAIN_20261028_v12.md)。这些进展均不打开P1/后续科学门。

**2026-10-28 本轮后续终态更新**：P3原生子节点物理已完成993public/10094physics、41执行与4原生成功早停（含重复轨迹，非独立成功率），saved-only精确修复后完整strict SHA `93785928ebaf03c1711610cba58ac9995e16b47d0fa2d741c6394867471b4f08`。P1 RGB40 schema修复后partial39主机CUDA及完整strict已完成，index12仍null、P1不放行。P5四词原生coverage和冻结goal scorer均已主机CUDA/strict完成。P2 Stage B cache及16路线compare均已主机CUDA/strict，public/manual buffers逐项相等，但组合机制和真实任务收益仍未证明。GRASP、matched sampling、moving gradient已有结果完成saved consume，仍非论文原协议复现。详见[v12最新终态](D2P_DOWNSTREAM_NATIVE_VALIDATION_MAIN_20261028_v12.md)，覆盖下方历史快照；整套研究仍active。

**2026-10-28 后续P最新验收**：P2缺三根历史已主机真实重建并完整strict，四根八臂历史保留，stageB完整接口尚待；E1八端点HOST CUDA及独立完整RGB验收完成，两known根跟踪代价下降但native仍0/4、root0终态Q更差，非P1通过。P5全四词物体响应分析与MAIN原生double独立复算完成，非C2任务收益。P3子节点桥53测试、完整prepare通过，主机run已启动但终态待核。P1 RGB40真实prepare捕获`physics_step`/`internal_physics_step`字段接线错误，已交新版本修复，不归理论负结果。详见[v12实际证据与边界](D2P_DOWNSTREAM_NATIVE_VALIDATION_MAIN_20261028_v12.md)。以下运行中/待编码文字保留为历史快照，最新状态以本段及v12为准。

**2026-10-28 E1物理独立验收补齐**：MAIN完整审读新consumer并独立30测试通过，四分支/六科学槽、2414事件/88PNG/12checkpoint全严格消费落盘，回执SHA `3926373a79e5c0eb9a6295dadd4ce1403dfca13d9baad51639cd7012c913c12b`。native成功0/4，不把compiled物体XY局部改善改称全任务胜出；真实latent tracking误差尚未编码。后续P2历史重建、P3子节点物理、P1全40 RGB与P5真实响应分析继续独占实现，原研究目标保持active。详见[v11](D2P_COMMON_CONTROLLER_AND_CHILD_MAIN_20261028_v11.md)。

**2026-10-28 最新完整消费终态**：P3 child compiler主机CUDA657predict后，完整strict已落盘SHA `d6322717231d72bb18e26eab1acbf6afe2dbc65eaa601af1fface21083a7e91c`，全48方法分母保留。P5四词真实物理80public/804Space与155实际callback完成；旧8MiB日志读取容量故障由新saved ReaderBridge修复，原full validator全量通过，回执SHA `f6cd1f53d7537a6292c49647a3e20cfd61caf8a4b5e5e9d395223fc5845a39ee`。不重跑、不删日志、不放宽科学门，仍非C2/C3/完整研究通过。详见[v11终态追加](D2P_COMMON_CONTROLLER_AND_CHILD_MAIN_20261028_v11.md)。

**2026-10-28 后续P新增实际执行**：共同低层控制器576次主机CUDA及完整strict通过，四条动作真实物理80控制步/816physics/88PNG均完成，工程失败0、native成功0/4，独立物理consumer及RGB误差量化待补。P3付费child程序编译657predict/41compile实际CUDA完成，41已执行/5预算拒绝/2未知不删分母，完整strict进行中；不称完整树或等实际算力优势。P2完整八槽history inventory实际落盘；P5四词新物理运行已启动，终态未核不放行。详见[本轮主线程实际证据](D2P_COMMON_CONTROLLER_AND_CHILD_MAIN_20261028_v11.md)。原研究完整闭环仍未完成。

**2026-10-28 本轮最新完成（覆盖下方运行中快照）**：P3全46获取主机CUDA及完整strict现均完成，完整回执SHA `efccb0ad57ff76e126e7027e9518b1c24b092c806a71a0ed55dcd96fa7ac90b5`；原cap/精度/48方法不改，41项pending/5项预算拒绝/2项not_reached已交child compiler。P5公开接触context四word的原生JEPA主机CUDA和全43数组strict亦完成，回执SHA `247a8a6328053f34b8367f39dca63f81fb009ca2a41734db0f618d1ea6eab2c3`。两项均不等于物理收益或科学放行；P1全40物理观测partial、P2目标不一致、P4历史认证及三实验闭环缺口仍保留。详见[完整本轮证据](D2P_PHYSICAL_AND_STORAGE_MAIN_20261028_v10.md)。

**2026-10-28 P3全量CUDA终态与P2新归因**：无损存储版HOST CUDA已完成236.18秒，全46获取/368predict/92J/17664scalar VJP/46schedule，errors=[]、模型未变，输出136MiB，报告SHA `cccb96d7d57dcf041021f8c43ab1f027a1012732ec56068ef1fb85348829bf5a`。完整strict到CLI写出时又发现64MiB通用metadata cap不足以保存90MiB全rows回执；原validator不改，新saved-only完整receipt writer重新验证中，不称strict已放行。P2 lineage的MAIN实际saved分析确认older2非imagined、全4known根actual terminal Q与jointXY偏好不同；这与预测排序误差是两层问题，不改Q/成功门槛绕过。P5公开原生context四程序CUDA已启动。详见[本轮实际终态记录](D2P_PHYSICAL_AND_STORAGE_MAIN_20261028_v10.md)。

**2026-10-28 全40真实响应新证据**：共同库完整800控制步/8160physics返回；39支完成、一支末步agent_y514.7196超出canvas导致partial，8个蓝色边缘像素仍存在，不能补图/放宽守卫。partial严格回执SHA `44acd71c60c8edc7fbfa196c2f46f7d8692ef487ec0c3a01b6b8f7e5071e9212`。MAIN独立复算40实际last_return原native成功0/40、31支root后物体pose改变、161返回控制步带非零接触标签；不是“全无物理响应”，亦非非贪心机制已成立。P3 v2经全源码审读及67项回归，保持原cap/精度/分母的主机CUDA全46acquisition重跑已启动，终态未核不得放行。详见[本轮实际执行记录](D2P_PHYSICAL_AND_STORAGE_MAIN_20261028_v10.md)。

**2026-10-28 P2固定suffix实际归因完成**：全四known根八臂仅替换实际中点最后latent，原后缀动作/goal/Q不变；主机CUDA八predict完成，strict SHA `36864d57aa6914f0d09ff4e947ef141ccf0037a2fc23f9d7e3f414fb217d04ac`。MAIN独立原生NPY复算确认八臂latent MSE均下降，终态Q排序一致1/4→2/4（root3修复，root0/2仍反转）。这证明中点误差有影响但单独替换不足以修复全部排序；无新物理收益、无P1/P2放行，不归为理论失败。新40分支物理adapter的MAIN回归捕获3个测试失败，已暂停真实执行并回传修复；详见[本轮记录](D2P_EXECUTABLE_PROGRESS_MAIN_20261028_v9.md)。

**2026-10-28 后续P新增实际终态**：P3 child HOST CUDA因 `output engineering cap` 中止：19 child返回、304predict/76J/14592VJP；确认为重复完整数组与账本JSON耗尽256MiB普通journal额度，是工程存储失败，不是科学负结果，partial不得放行。失败报告SHA `c0279c6a48d86bb46ed90b1d098bd436c76a9e85daa00893d34a119afa574641`；保持原cap/精度/分母的新无损版本推进中。P5新固定接触源真实两臂完成，96原生agent-block回调，推送改变物体而zero不变，strict SHA `d13719d2a7914d7ca28a25e56f16eaf2e2fcdbd868ee388d3f162322bc7e6dc0`；七NPY盲输入已本地分离，非GitHub发布、非fresh/P5机制准入。详见[本轮执行记录](D2P_EXECUTABLE_PROGRESS_MAIN_20261028_v9.md)。

**2026-10-28 新共同库实际终态**：P1八teacher锚点共同对称库已完成HOST CUDA：8encoder/projector、96predict、8J/1536数学VJP，模型未变、errors=[]；完整strict核原生数组/事件链并保留全部40动作计划，回执SHA `d82109eb8de582e59a4f0485307c01c9b91f9f84551c218c61cca4da24335213`。真实40分支执行仍待完成，不是P1或非贪心C2通过。同身份五基线×两known根的saved全闭包分栏亦完成，10个原生失败完整保留，非等总算力/跨论文排名。P3 child信号源码经MAIN完整审读与45项回归，新实际prepare推进中；详见[本轮执行记录](D2P_EXECUTABLE_PROGRESS_MAIN_20261028_v9.md)。

**2026-10-28 P2新增真实边界证据**：token/precision接线修复后，固定四known根八臂step5主机CUDA编码、完整strict及MAIN独立8数组数值复算均已完成。报告SHA `7b0b5c00735a7326326a4619c86539fdb51b1bf99f384fe8794a71a6b4e6321c`，strict SHA `4e61d9251f86550d0830c33d5607cfe58d4b308b97865f7fc64de0013d25d135`。root0两臂同prefix5、midpoint差0但终态反转；root2由step5同向转为step10反转；root3在step5已反转。不是所有失败都始于组合第二段，也不能凭这些边界归因理论。后续全八臂固定suffix的actual-midpoint条件预测已派实现，无训练/筛根/阈值修改。P3完整child prepare/verify/48槽plan已完成，46槽待新付费信号、2槽未到达；无科学放行。详见[最新终态](D2P_DOWNSTREAM_EXECUTION_MAIN_20261028_v8.md)。

**2026-10-28 后续新增执行与代码归因**：beam八图主机CUDA与完整严格消费已完成，两根root/goal重编码逐位复现，后继仍有预测误差，两根native成功均false；不是官方kinodynamic/RRT或等FLOPs结论。P3 child接口原失败定位为export根SHA为null被误接入严格binding，现沿已绑定sourcepack同slot/blind身份桥修复，MAIN 39项回归通过，不放宽validator。P2 midpoint新桥的197/257 token硬编码错误及数值守卫仍待冻结修复、审读和真实CUDA，不作为科学负结果。P2–P5实现并行推进，但child付费信号、fresh认证、可接触源等缺口未被接口或测试代替；详见[最新后续执行记录](D2P_DOWNSTREAM_EXECUTION_MAIN_20261028_v8.md)。

**2026-10-28 新实质归因**：P5 JEPA全12固定action-word原生物理及strict验收完成（report SHA `796d757c6ffcf623f4e307f094dba9e39eda2a4810a140e5ab3cf0897b3c73a5`；strict SHA `2248899fb03439bc4c52ee741dc0e2b497e15ce692d1d4d306f32fee7d476883`）。MAIN独立审计全部后缀原生public状态：12臂block pose逐位不变，120个控制步返回contact计数全0；推杆终态差异仍存在。因此已有latent action-word差异不是物体重配置证据，本组也不足以证伪C2。zero5与selected-prefix5同时间signed比较实际完成，MAIN独立复算112个变化量通过；只控制elapsed、非状态/接触中介隔离。具体产物见[后续执行记录](D2P_DOWNSTREAM_EXECUTION_MAIN_20261028_v8.md)。原三贡献、三组实验、强基线与独立确认闭环仍未完成。

**2026-10-28 后续P最新终态**：P2四根八物理分支、P3 fixed5四物理分支、P4三反馈边界八臂及P5 JEPA十二固定两包程序CUDA均已完成各自严格消费；不是P1--P5科学放行。新增P1 zero5同时间控制20/20真实分支的严格消费亦已完成（回执SHA `7c22af94f7aebaabbe4857cb1797c1b7fe857020145829273fb2349b45ea8e19`）；P4独立rolling接缝消费通过（SHA `182d5ce14a6363b287607c511d3bc4dde85769618e8c8082029f090fb19c1232`）。当前并行推进固定探针的时间对照分析、P3资源与原生指标核账、P4新根前置和P5物理适配；未交付部分不得算完成。实际产物与边界以 [后续执行记录](D2P_DOWNSTREAM_EXECUTION_MAIN_20261028_v8.md) 最新终态为准，下文“运行中/待核”属于历史状态。

本页对照用户给定的 Planning Compiler / Runtime 提案，不把“完成一轮诊断”缩写为
“完成研究”。原始范围包括 C1--C3、三组实验及 P0--P5；同名 SRA 与 D2P 分支不互相顶替。
本页不是降低门槛的新方案。当前总状态：**未闭环，研究目标保持进行中**。

**2026-10-27 新归因证据**：MAIN修复archive探针遗漏proprio字段/未注册压缩codec两项工程错误后，
固定episode1/6229×lag0/1四full20主机物理探针完成、errors=[]；原顺序lag0状态重建误差<=4.96e-5，
但相同state0的archive带绿色目标overlay，with_targetFalse的live无该轮廓。
这是明确的archive/live图像协议差异，不能再默认“只有理论/冻结模型边界、无待核协议因素”。
尚未证明它是全部排序失败根因或checkpoint训练配置，需新增同body渲染与后续CUDA对照，
不得更改旧合同、直接称P1通过。报告SHA
`affb90a6e67c096623b62ac1d4a1f9f2e11ceaa5d5fe8436f365d39536de2e05`。
同body显示对照现已实际完成：两episode原lag0全20步，五checkpoint各off→on→off，
10triplet状态/goal/renderer不变、off replay及旧off逐位相同；30PNG/40public/408physics、errors=[]。
显示默认轮廓使对archive的平均RGB MAE 1.970554→.411711（约−79.1%），
报告SHA `184809d72b4e4a6194cb9d7591b26f4a2fe09d423a826ab02c1fcb5cf340c430`。
这解释大部分像素失配，但尚不证明模型预测/规划排序修复；matched factual CUDA仍待实现与执行。

**2026-10-27 最新moving-history校准链已完整执行**：固定两recipe派生新root/goal，
实际非零warmup、两次源重放、CUDA盲编码、GN/D2P四计划、四full10物理与8张RGB编码均完成。
源/规划/物理/RGB均errors=[]，模型未变；四计划仍0/4 native成功、D2P两根仍为bank即时贪心。
RGB报告SHA `7ec03c0a6a6e8411ff0dd0115044a872d099106edc95e4529b97c38c2b9b8f71`；
slot0模型terminal GN/D2P .60033/.65476，实际Q .48217/.18266，跨arm排序反转。
固定factual诊断另完成2encoder/8predict，真实prefix反馈对终态预测改善非一致。
这些是校准与归因进展，不是独立确认或原非贪心机制成立。
观测反馈suffix重编译及四分支真实执行现已完成：20次CUDA predict、4 Jacobian/4 SVD，
四分支原prefix逐帧精确复现、errors=[]；仍0/4 native成功。
D2P两分支native距离变化−30.37849/−27.14667，GN两分支+96.31167/+13.92989；
并非一致改善，更不是P1通过。D2P slot0四维XY误差20.24546，必须按原<20阈值计失败。
新physical报告SHA `3376eda7d0a88b31e8fb56a5f3b2cc19ef4854ef5b12f35ae196da0744c71f0e`；
主线程核2426事件链、88PNG及68相同prefix帧；反馈terminal RGB4亦已主机CUDA完成，
报告SHA `3c3e2f66b48bf4cd8c427d318b0a456e2498c5499937148c6a8a1ca06e547caf`，4encoder/projector、errors=[]。
MAIN原NPY复算：同实际prefix条件下旧/新suffix模型Q方向与真实Q方向3/4反转，
不能把接入反馈视为排序问题已修复。跨arm D2P终态更好仍不等于非贪心机制或任务成功。
本批native成功要求agent+object四维XY与角度；下文2026-10-26 object-pose evaluator文字是
历史的另一合同，不可直接套到本批或互换成功数。详细新证据与SHA见文末及当日并行进度。

**2026-10-27 真实前缀反馈的误差隔离已实际执行**：覆盖原16分支的17个内部
packet边界，新主机CUDA encoder/projector各17次；随后保持动作不变，以实际
前缀因果history完成17predict/action_encoder条件预测，未训练、未新增物理。
17/17项native预测MSE下降（含重复轨迹，非17个独立效果），但旧Q受损第2根
校正预测排序仍与真实Q反转；第3根仍是实际Q改善而物理pose损害。
因此反馈可减少误差，但仅修回真实前缀不足以解除排序/任务度量双重缺口，不能
改写为P1通过或理论失败。实际条件预测报告SHA
`c5bd1f15a4a8aa7dd60050f8ce5a681aa20c1966e1715cdc3e1df013e9fba499`；
目录 `le-wm-main/results/p3_prefix_teacher_forcing_20261027_v1_cuda_main01`。
主线程独立保存数组复算与157项相关联合测试通过，测试本身不证明研究效能。

**2026-10-27 新P3全链校准已实际完成**：新joint-H2/H3四臂主机CUDA
1066predict/1066action-encoder后，16新鲜分支165selected+240warmup真实执行，
16CUDA终点编码完成。主线程独立保存数组审计、官方指标/PNG/动作/journal/预算/
两Q数值复算通过；全部官方/object-only成功仍0/16。旧Q动态调度两根pose损害
分别存在“预测/真实Q排序反转”与“真实Q下降但物理pose变差”，不能混作唯一
根因或理论反例。新Q三根调度动作相同、一根mixed-H，尚无独立C3真实收益。
真实报告SHA `6c3659e199a8151fee2fa2c020a0cfe63db00b691b947cdbeb762c3c18d6004a`；
目录 `le-wm-main/results/p3_joint_horizon_goal_metric_physical_20261027_v1_main01`。
P2新168/168预测前沿runner已通过79tests/封存verify并实际完成主机CUDA：
1328predict/164J/480selections，8个赢家全槽导出复核通过；reportSHA
`4023a0e3896a90407188d260a45dd334e0802ba17ecd59ef62ea931603eba173`。
扩大joint库后组合仅2/4预测更好。新P2动作随后已实际执行8分支，80selected+
120warmup共200 public steps，40 resets，8次主机CUDA encoder/projector完成；
报告SHA `9144c3f25f2880af62d94dd88257918fa508b5ddbea51d98e9ccbc3d3b7532e2`。
主线程独立由原始状态复算pose、官方指标及80步reward；official/object-only均0/8，
组合仅1/4根改善pose。仍有模型/真实Q排序反转与Q/物理目标失配，AD/FLOPs不匹配。
P1/P2/P3科学准入不由这些工程结果打开。详见当日并行进度文档。

**预测样本匹配校准已实际执行（2026-10-26）**：按计数而非物理收益将CEM
固定为67样本×4轮×2步，原两根主机CUDA重跑完成、7门通过。D2P/CEM实测
均536预测样本，调用仍122对8，不能宣称总算力匹配。关节终距一根D2P较好、
一根CEM较好，只有校准描述，无outcome准入。未继续按结果调参。详见
[P0预测样本匹配核验](D2P_P0_PREDICT_SAMPLE_MATCH_20261026.md)。

**P0计数已完成真实CUDA核验（2026-10-26）**：原seed20261040/41两根仅校准
重跑完成，7项原门通过，D2P/GN动作仍完全一致，无outcome比较。每根实际
predict调用/样本分别：D2P与GN 122/536、GD 32/32、restart-GD 34/40、
CEM 8/256、random 0/0。相应旧字段是260、16、17、128、0，不能当作
同一计算单位。默认D2P实际预测样本约为CEM的2.094倍，现有默认协议不是
predict-sample匹配；这不是事后重调预算许可或完整FLOP/反向工作比较。
结果：`le-wm-main/results/reacher_d2p_p0_predict_accounting_calibration2_v1_20261026_cuda/report.json`。
本条更新下文“尚未重跑”的历史状态；无新P0效能或P1准入结论。

**P0实际调用核账已接线（2026-10-26）**：Reacher P0的D2P/GN/action-GD/
restart-GD/CEM现在分别记录真实planning predict调用与batch样本数；random
动作生成明确为0预测。旧`forward_rollout_evaluations`只为档案兼容保留，不能
再当统一预算。计数器退出/异常均恢复原predict；encoder、backward和执行后
诊断仍不在该局部收据中，不能声称全资源匹配。元数据假对象接线及原P0测试
38项通过，无CPU神经计算；尚未以新接线重跑真实P0，历史等预算不追认。
文件：`le-wm-main/experiments/spectral_reachability/d2p_model_call_instrumentation.py`、
`le-wm-main/experiments/spectral_reachability/run_reacher_d2p_p0_goal_progress.py`。

**并行后缀实验已独立复核（2026-10-26）**：3968主机CUDA预测、128真实
分支、11门通过，identity-GN4复现历史动作/终态。去推杆Q虽降低共同Q实际
编码成本0.063998，物理pose cost却增加2.930957；四根均恶化，64对中13好/
23坏/28同。不能称修复成功，亦不能把后缀局部干预当全链或理论否定。
并行核账确认旧P0混用predict调用与样本单位，强基线等预算仍未确证；新
ledger只是实现，需实际instrumentation与重跑。详见
[后缀真实执行与证据边界](D2P_PUSHT_GOAL_AGENT_INTERVENTION_20261026.md)。

**候选Q物体保真检查已实际完成（2026-10-26）**：44次主机CUDA目标编码、
9完整性门、44个独立最小二乘残差复算通过。有限goal差分投影对24个物体
平移/旋转干预保留编码能量中位0.867725（最差0.726814），对16个未见推杆
位置保留中位0.002828。这支持继续检查候选，但不保证物体控制信息完整、
预测排序或真实控制收益；显式goal集合输入合同不能隐藏。尚未完成全链接入
和物理效能准入，P1不回填。详见[目标干预后续验证](D2P_PUSHT_GOAL_AGENT_INTERVENTION_20261026.md)。

**目标合同干预（2026-10-26）**：已验证PushT目标渲染同时约束推杆和物体，
而物理评价仅约束物体。保持动作库/物体目标不变，只平移目标推杆，20次主机
CUDA编码、9门通过；full192 one-shot/GN4的固定库terminal argmin在16种
干预中分别改变4/2次。960个成本独立复算一致。这不是原selector变化或
控制改善证据。有限goal差分span投影候选已实现，尚未全链接入、物体敏感性
验证或物理校准；不得写成P1修复成功。见
[目标推杆干预与候选度量](D2P_PUSHT_GOAL_AGENT_INTERVENTION_20261026.md)。

**PushT进一步定位（2026-10-26）**：全2336736条归档动作仅字段分块统计完成，
std比公开示例高约3.3%，但尚未认证该归档与官方权重训练版本相同，未换常数。
全部192条真实分支的保存数组复算发现，三种后缀求解器分别51/47/40条物体
终点姿态精确不变，推杆均有位移；其中29/33/28条真实编码RGB目标仍下降。
这进一步具体化了目标代理/后缀执行缺口，不是接触缺失的直接证明或理论否定。
原4个封存selected后缀全部未改变物体姿态；本批zero与initial物体pose相同，
改参照也不能修复3/4非贪心门。新增16项测试通过，未新增神经/物理实验，
不回填P1。详见[动作统计与真实物体响应审计](D2P_PUSHT_ACTION_STATS_AND_OBJECT_RESPONSE_20261026.md)。

**最新机制合同与状态准入审计（2026-10-26）**：已完成696个新同度量拟合，
明确区分shared-residual能力增益与原提案own-state可达目标；诊断合同确有
差异，但未恢复native完整见证。另已完成32根纯模型条件主机CUDA筛查与
独立全量复算：149184预测样本、15完整性门，单执行器“有限预算梯度平台
且采样逃逸”合格根仅1个，不足冻结的8根；准入队列为空，不补根、不改阈值。
没有物理未来筛根，也没有打开P1或依赖阶段。完整相关回归1591项通过。
PushT当前可见训练/导出资产仍不足以认证官方权重的精确normalizer；额外
除100的既有修复不能自动补齐这条溯源。详见
[机制合同、模型条件队列与环境前置审计](D2P_MECHANISM_CONTRACT_AND_MODEL_COHORT_20261026.md)。

**已完成的评分验证与完整编译器验证（2026-10-26）**：目标图像
co-metric评分的独立八根202654000–007现已完成403920个主机CUDA预测样本、
456物理分支及23项完整性门，并完成独立复算。相对原固定greedy，gain-only
在全驱动/仅肩/仅腕的关节终距均值差为−0.00412/−0.00546/−0.00599；
consistent-Q为+0.00249/−0.20227/+0.07573。任务与根间不稳定，保留所有损害根，
不事后挑模式，不视为P1通过。该轮未重新优化动作或后缀。

另一批202655000–007的完整Q一致编译器v2现已完成805536个主机CUDA预测样本、
912物理分支及18项完整性门；1632个独立QP和全部动作见证复核通过。Q已接入
谱方向、动作/后缀优化与own-state可达代价，全部预测先封存后执行物理分支。
三任务均0/8模型重配置见证，24根×任务中23次选中自身greedy；关节终距对
原固定greedy均差为+0.003010/+0.001808/+0.016933，不能称修复成功。
完整库仍有16/29/21个真实关节与原RGB首段/终态合取候选，但全部未通过模型
完整判据；合取不含真实能力改变，不能冒称C2或部署oracle。新能力探针也已
完成58单元、3116真实未来、6522主机CUDA预测样本、12门和1044独立拟合，
角色仅取封存native/common/greedy/zero，不按物理候选筛选。common模型见证
1/2/6个中，固定Q真实RGB能力改善0/2/5个，但原物理关节C2诊断及同根全部
要件仍均0。局部AD/FD与真实RGB响应方向证据较强，未发现明显新的AD错误；
它不能排尽所有代码问题，也不能把代理能力改善写成任务机制成立。
新增源码绑定和CEM返回值接线错误均在物理执行前修正，失败归档保留，不计
理论负结果；本批数值审计通过也不代表排尽所有代码问题。相关回归1557项通过。
固定协议、产物、实际命令见
[目标度量新根验证](D2P_GOAL_METRIC_PROSPECTIVE_CONFIRMATION_20261026.md)。

**最新增补（2026-10-26）**：原独立八根真实能力探针现已补齐，57单元、3074
条物理未来、6433个主机CUDA预测，12门通过；1026拟合独立复算。仅肩3见证中
1个原关节C2诊断、2个固定Q RGB能力改善，但无同一根全部成立；仅腕亦无。
8目标直接渲染和32轨迹精确重放，排除本批陈旧目标/轨迹错配。新的目标编码器
输入co-metric候选已实际执行1536 CUDA VJP、104 forward、408证书QP，14门
通过；独立复算全408QP。两种固定评分的物理首段/双终态合取最多全驱动1/8，
仅肩/仅腕0/8；仅重评动作库，未重新优化后缀，不是完整修复或P1。
详情及全部产物见[能力与目标度量审计](D2P_PREFIX_CAPABILITY_AUDIT_20261026.md)。
本增补覆盖下文历史“未补新根能力探针”状态，不追认历史P1或宣告理论失败。
C1执行证据不能替代C2物理能力改变；C3多任务自适应算力及原三组实验仍未闭环。
P1正式独立校准与outcome仍未通过，依赖它的P2/P3/P5不能开启；P0/P4既有
诊断/合同结果不等于原提案全部强基线、等预算与泛化要求已经满足。

**最新渲染修复已通过完整测量**：v1/v2因同状态单像素分叉在第七根严格停止，
现以固定两次真实渲染、只留第二次的显式`fixed-warm-render-v1`完成主机CUDA
八根/三任务/17程序：408单元、22576条真实未来、47192个predict样本，13门
全通过，包括原52单元逐位对齐、参数/buffer/源码未变及独立渲染收据检查。
68592次观察请求、137184次真实draw/完整状态检查；预热与保留有5次最大1单位
像素差，如实核账且没有结果选取或重试。原失败根及后续全部完成；旧失败不追认。
这修复了完整测量的运行时阻断，不解释所有历史P1反转。见
[单像素渲染分叉](D2P_RENDER_REPLAY_AUDIT_20261026.md)。

**最新补齐前缀后的真实能力测量**：8已知根三任务四角色，52去重单元、2864条
真实未来、5988个主机CUDA预测样本、11门完整通过，624个有界拟合独立复算。
模型AD与真实RGB-FD余弦中位数0.984/0.984/0.992，未定位新的明显梯度实现错误。
但关键202652000虽真实非贪心且终态获益，其关节共同可达增益低于zero和greedy，
不能冒称能力改变。RGB局部增益与真实关节任务增益明确分开。
另完整24根全部17程序覆盖审计发现，改善后缀后18/19/19根分别存在物理非贪心且
双终态指标胜greedy的候选，模型识别9/18/16根，旧common仅选中1/0/1根。这是
资格判据/选择缺口的诊断，不是oracle选择或P1通过。完整408单元现已全部独立
完成4896个拟合，另完成408程序拒绝谓词重建及1632个响应/残差两因素拟合。
全驱动/仅肩/仅腕有3/20/7个物理诊断候选（覆盖2/5/3根），均被旧判据拒绝；
全部30个候选的真实RGB共同增益也未胜控制。四种响应/残差组合均未恢复正
margin，原QP独立复现最大差4.95e-12。优先处理RGB能力代理与物理任务能力
的度量合同错位，而非继续把它混称为梯度/残差代码错误或理论失败。不能将
这些事后候选用于部署挑选、筛根或回填P1。七项产物SHA与两张图已复核。见
[前缀能力与完整库覆盖审计](D2P_PREFIX_CAPABILITY_AUDIT_20261026.md)。

**固定模型根度量候选已完成离线审计**：只用原模型zero前缀后的预测Jacobian
构造Q=.01*(J0*J0^T+.01*I)^(-1)，不使用真实状态、标签或后训练。全部408程序
的1224拟合保留所有任务/根；仅肩20个原物理诊断候选中，model正margin8个、
真实RGB两尺度正margin9个、同时5个，原欧氏对应均0；全驱动/仅腕无同时成立者。
这只是能力代理预条件的局部数值证据，未更改正式selector、成功定义或P1门。
CUDA证书化评分现已完成：24个SVD、408个QP、8门全通过，全部旧决定精确
复现，无新神经推理。独立408个SciPy点解位于对应证书区间。保留全部原判据的
实际模型选择为：全驱动0/8见证，仅肩4/8中只有2个同时符合原物理诊断和
新度量真实RGB双尺度增益，仅腕3/8中为0个。不能用候选库中的5个局部正例
替代实际选择。202653000–007独立源episode主机CUDA配对验证已完成，
排除224个既有源episode，保留三任务/17程序和全部控制；预先封存403920个
模型预测样本及新旧选择再执行456真实分支，17项完整性门全部通过，独立408个
模型QP及全root/action/source/源码绑定复核通过。仅肩新评分选中3个模型见证，
3个均真实首段劣于zero且终态RGB/关节优于greedy，另5根保持回退；全8根关节
均差-0.270743，描述性CI[-0.655816,-0.001290]。这提供了评分实现可恢复部分
行为的独立证据，不能把旧零放行直接当理论失败。但仅腕5个模型见证只有1个
满足上述真实合取，RGB终态5胜/0负/3平却关节均差+0.100748；损害根保留。
未补新根真实能力增益探针，故仍不是正式C2/P1或完全算力匹配，不开启后续门。
最新广覆盖回归1472/1472通过（48.90秒），`git diff --check`通过；逻辑合同
与真实预测/控制效能继续分账，不以任一局部诊断关闭原C1--C3/三实验/P0--P5范围。

**新增明确的代理选择根因**：冻结官方模型的16已知根、三任务816程序，完成240条
真实分支、35088个主机CUDA预测和15项完整性门；1632个新QP独立复核，96条旧
选择完整物理重放。全驱动202650004仅将零切点局部线性代价换成已有非线性可行
动作代价，不增加搜索，即把终距1.654116降至0.024495；固定两次重线性化进一步
降至0.018549。这是一处实现近似造成的误选，不得称为理论负结果。但三任务仍
没有相对zero的非贪心被选根，全驱动均值收益主要由已知单根主导，P1保持关闭。
来源隔离新八根与旧八根对齐控制的CUDA确认也已完整完成：240分支/35088预测/
15门全通过，1632个新QP独立复算、旧八根五种执行逐位复现。新仅肩只换评分
均距0.697934→0.560880（2胜/0负/6平）；同优化native对greedy为3胜/0负/5平，
描述性区间[-0.799799,-0.124412]。但三任务仍没有相对zero非贪心被选根，不能
把分支修复或优于greedy等同原P1通过。另完成全部24根的模型单独stage下界审计，
只有2根被stage代价全面排除，否定该解释普适性，不据此事后调beta。
见[零切点与非线性可行代价审计](D2P_RECENTERED_REACH_AUDIT_20261026.md)。

**最新因果残差执行检验已完成**：同16根、三任务、三固定程序、四执行方式，
576条真实分支、29376个主机CUDA predict样本、12门全通过。原三组对照的432
分支/1296包完整动作、192维预测、物理观测和求解receipt逐位复现。仅用上一包
已观察误差的固定单位增益补偿，在同动作分析中纠正004却改错另一个根；实际
执行004仍劣于同反馈greedy。三任务相对同预算目标重规划的距离/RGB区间均跨零，
不能作为P1修复，未设默认、未调参或删根。尚未定位剩余反转的新代码根因，
也不据此宣判理论失败。详见[因果残差审计](D2P_CAUSAL_RESIDUAL_AUDIT_20261026.md)。

**最新逐包定位和因果执行修正已完成**：全16已知根×三任务×三固定程序，144条
真实分支、1728个CUDA预测样本，官方/手写rollout全部432包逐位一致；真实历史
单步预测仍保留全驱动202651004的终态反转。完整192维margin误差精确分解的最大
恒等式误差1.083e-15，不用oracle未来做部署修正。随后保持原前缀，对三类程序
统一比较replay/目标重规划/路标跟踪，432条真实分支、19728样本、12门全通过；
1296组因果选择执行链独立复核。全驱动native路标跟踪均距0.127644→0.113647，
12胜/4负，描述性CI[-0.024235,-0.005307]；肩/腕任务没有稳定收益。关键004在
同样路标反馈下native/greedy真实均距0.020819/0.029829，原劣势转为优势，但
是动作改变后的执行改善，不能声称同动作排序修复或稳定C2重配置。新入口另纠正
双关节wrap指标复用，全部432实际端点数值不变，不冒称反转根因。P1保持关闭；
相关广覆盖1131项通过，详见[逐包反馈执行修正](D2P_MASKED_PACKET_FEEDBACK_REPAIR_20261026.md)。

**最新同前缀求解因果证据**：官方冻结模型、同八根、三任务全部17前缀与旧判据
不变，后缀改为保留原解的CEM后GN4。1224条真实分支、349320个主机CUDA
predict样本，16门全部通过、原408分支完整逐位重放；816个完整响应QP独立复核。
全驱动/仅肩/仅腕的全前缀真实平均终距0.779198/0.897793/0.861826降至
0.147112/0.650112/0.711909，根级胜负8/0、8/0、7/1。这支持求解不足，不能
归为理论负结果；但旧选择器整体终距未改善，仅腕两个新模型候选仍有一个首段
RGB符号反转，不能开放P1。另将原提案的state-specific residual与旧common-gain
诊断显式分开，用独立公式实现 `prefix_MSE + own_V/192`，固定beta=1，在新根
结果前封存，保留旧选择器作对照，不事后调参。来源隔离的新八根确认中，发现
官方源状态202651005有真实软限位穿透，旧目标wrapper将名义限位当硬边界而拒绝。
已新增显式v2合同，不剪裁源状态、不删根、不放宽目标限位/完整偏移；16根来源
与渲染审计完成，15个原有效上下文逐位不变、唯一旧拒绝精确复现后保留，全部
七门通过。全八根CUDA基线456分支/59568样本/14门与后续1224分支/349320样本/
16门现已完整完成，408基线分支逐位重放，另816个QP独立复核。新三任务全前缀
均距0.699495/0.842140/0.839171→0.073673/0.589849/0.762679，前两任务八根均
改善，腕任务CI跨零；不能外推为普遍物理收益。proposal-native三任务均0/8模型
重配置、没有真实非贪心被选根；仅肩progress选择改善一根，其余并非稳定优势。
旧common-gain全驱动004新候选首段确实非贪心，却有真实终点反转，P1仍关闭。
独立代码复核另发现flat CEM缺singleton原解兜底、后缀初始NaN/形状检查缺失，
均已修复；全部16根三任务97392个CUDA样本、10门的差分回归保持48组flat控制
与零前缀后缀的动作/预测逐位一致，816份旧后缀receipt均有限，故不冒称本批反转
根因。全根图和9项产物SHA复核完成；最新1072项广覆盖回归通过。
详见[后缀求解与残差合同](D2P_MASKED_SUFFIX_SEARCH_AUDIT_20261026.md)。

**严格执行器任务合同已完成主机CUDA预检**：旧5%力矩限制不等于关闭执行器；
新增全驱动/仅肩/仅腕 reduced raw chart，未启用执行器始终字面原始零、历史不变，
归一化仅一次。全部八新根×两时域×两幅度，共976条真实未来、2272个predict样本，
12项完整性门全通过，独立复算完整192维RGB/qpos端点与384个线性拟合。幅度0.25
两包最小物理响应奇异值均值为0.489252/0.032425/0.024709，有界线性残差比例为
0.092225/0.559978/0.478930；这证明本批单驱动的短时条件变差，不是不可控定理，
也没有证明重配置有效。另一批与全部200源episode隔离的新八根的全三任务17谱
前缀/完整动作见证审计亦已完成：456条真实分支、59568个predict样本、14门全通过；
408个程序分段/整段预测逐位一致。三任务均0/8合格见证，native全部回退greedy；
真实非零前缀符号一致94/128、77/128、91/128，不宣布排序修复或理论失败。
全驱动flat GN/CEM+GN incumbent均距0.865052/0.028534，仍需查搜索盆地/候选覆盖。
首版新writer误用最后root身份被门拒绝；修复后完整原样CUDA重跑，456条分支与
1368包逐位相同，预测/决定除耗时外不变，独立归因通过，不冒充旧P1根因。全部
全根图与产物SHA已核验，最新989项合同测试通过，不借此开放P1。
见[严格执行器任务审计](D2P_EXACT_ACTUATOR_TASK_AUDIT_20261026.md)。

**反馈后的直接物理响应审计已完成**：全8根×4类计划×2执行模式，在第四包真实
后继测量两包未来、二十个command坐标、正负方向及0.1/0.25两幅度，共5184条
真实未来、10496个主机CUDA predict样本，12项完整性门全通过；448个有界增益
由NumPy/SciPy独立复算，双图及产物SHA已复核。模型AD-vs-model-FD相对误差
均值0.001126/0.007038，而AD-vs-physical-RGB为0.189516/0.136191。native相对
greedy的真实qpos响应增益差CI均跨零，replay幅度0.25的真实RGB增益0正/7负/
1平，不支持稳定的能力改变优势。普通全驱动任务前提仍未确立，不上升为理论
否定或P1通过。首版新审计有一次严格RGB重放失配，完整512边界隔离和修订版
均未复现；原因仍未确定，失败证据保留，不宣称已确诊某个像素量化bug。修订版
使用两轮精确验证的真实观察/完整状态，不放宽身份阈值。新增32项合同回归纳入
最新906项全覆盖通过。见[真实动作响应与重放审计](D2P_FEEDBACK_PHYSICAL_RESPONSE_AUDIT_20261026.md)。

**多包历史误差定位及因果反馈对照已完成**：官方原生rollout与手写递推在全八根
64个完整192维包边界逐位相同，最大差0；每包真实已执行历史单步预测消除剩余
003的符号反转，但只作evaluator定位。进一步封存32条原计划/全部waypoint后，
只用自身已观察RGB进行固定4次GN反馈，64条30步物理分支、384包、5536个
CUDA predict样本全部通过完整性检查。native均距0.044392→0.024082，7胜/1负；
末包执行偏差1.966→1.245，但描述性CI跨零，原前缀非贪心条件仍未恢复，不开放
P1，也不宣布理论失败。新增83项合同测试通过，四张全根图/全部产物SHA已核验。
排除历史184个episode的新八根202648000..007已在新结果产生前封存，四阶段确认
现已完整完成：原方案128条/反馈64条物理分支、共401878个CUDA predict样本，
外层6项与子报告14/11项完整性门全通过。native初始计划反馈均距0.063315→
0.027490，8胜/0负，描述性CI[-0.045332,-0.027331]；末包L2偏差2.740→1.513。
七个原模型witness经反馈后相对固定neutral均真实非贪心，新反转根001执行条件
恢复；但反馈改变了动作，不是同动作排序修复，旧批003仍失败。原方案七个witness
仍有一个反转，对同反馈greedy/CEM等对照的CI均跨零，不开放P1。两个新图包四图
及全部产物SHA已核验；本批无目标反射，不能当BN或限位修复效能。见
[历史误差与因果waypoint修正](D2P_WAYPOINT_TRACKING_REPAIR_20261026.md)。
另完成过去观测单位残差的固定双变体CUDA诊断，136个predict样本、原基线全部
逐位复现；终点修正平均L2虽降至1.229，却多引入004反转，每步修正则增至2.464。
不将它部署成修复，也不把平均误差改善冒称排序成功；详见同页。当前广覆盖840
项通过，另4项pipeline及13项残差合同回归分别通过。
随后补上全8根×4包×2预测的精确margin误差分解，恒等式残差至多7.66e-16：
003反转主要是动作相对响应误差而非公共neutral偏移，oracle公共平移亦不足。
两张完整根图已目视和核验SHA；纳入双neutral参照合同后最新全覆盖874项通过，
不据此宣称理论失败。

**合法目标新8根已完整完成（202647000..007）**：官方原始权重、8个唯一且与历史
176个episode隔离的来源根，136词/128条真实分支、398540个CUDA predict样本，
14项完整性门全部通过。native dispatch/同库greedy/CEM平均关节误差分别
0.032802/0.032761/0.024858，对greedy为2胜/4负/2平且描述性CI跨零。6个新
witness中5个保持真实RGB非贪心，合法目标根202647003仍反转。这批无目标反射，
不能冒称限位修复的配对收益或BN修复泛化；两张全根图、全部产物SHA与独立后处理
均已复核。P1仍未通过，不据此宣布理论失败。详见
[合法目标新根确认](D2P_GOAL_JOINT_LIMIT_REPAIR_20261026.md)。

**2026-10-26原六根反转已定位**：固定原root/goal/动作/真实RGB，官方原始
权重排序6/6，旧joint-repair为4/6。进一步仅恢复pred_proj的3个BatchNorm运行
buffer、保留joint-repair全部学习参数，即恢复6/6；反向统计交换令官方降至5/6。
两套原基线逐位重现，FP64动力学不改变结论。已在三条repair trainer默认固定
base BN统计，并在三条实际训练循环用主机CUDA真实optimizer step验证：新模式
统计不漂移、参数有更新、梯度有限，两种模式共6个优化步。原两个动作被正确识别为贪心，因此这是错误见证/训练
归一化合同修复，不是P1机制成功。此前对原六根的“冻结模型边界”归因由此收窄。
见[预测头归一化根因与修复](D2P_PREDICTION_NORMALIZATION_REPAIR_20261026.md)。
完整向量独立后处理、全根图与SHA manifest已复核，相关广覆盖757项通过。
新增分析器回归明确拒绝部分gate、重复根、改变视觉坐标及非逐位基线；这些
完整性测试不替代P1效能证据。

**2026-10-26最新代码根因**：历史目标生成把wrist当成周期关节；实际模型wrist
有限位±160°。官方HDF5与源码核对发现两个最大误差根202644006/202646001的
目标分别为+3.030815/+3.132241 rad，均越界。这覆盖此前将两根严重失败归为
checkpoint边界的解释，但不解释其他合法目标上的首段RGB反转。新增限位合法
目标wrapper保留全部根与完整偏移幅度，全16根32条主机CUDA分支已完成，
73728个predict样本，全部门通过。同样CEM设置的两根距离分别
0.831375→0.075630、0.753711→0.086552，终态RGB成本也恢复；其余14根图像、
轨迹、动作及终态精确不变。这是显式新任务合同，不能回填旧P1；最新广覆盖
684项通过。见[关节拓扑目标修复](D2P_GOAL_JOINT_LIMIT_REPAIR_20261026.md)。
另精确重放用户引用的原6根margin校准：全部目标合法，原反转根20274802/
20274804不受限位修复影响。该代码原因不能外推解释原6根排序；本轮已另由
上述预测头BN干预定位原六根反转，P1仍需独立机制/效益验证。

来源隔离的全词评分独立确认现已完整完成：8新根136词、128条真实分支，
413302个predict样本，全部完整性门通过。新dispatch/greedy均距
0.176866/0.176877，3胜/1负/4平且CI跨零；4个新witness仅3个实际首段RGB
非贪心，合法目标007仍反转。逐包RGB反馈对照亦完成：8已知根24条独立策略
加16次全程重放（共40次30步执行），129480个predict样本，全部门通过。
开环/冷反馈/精确carry反馈均距0.152007/0.117004/0.135600；冷反馈7胜/1负，
但越界目标坏根未恢复。不能把增加反馈与计算当作P1机制收益。
见[逐包反馈审计](D2P_PACKET_FEEDBACK_AUDIT_20261026.md)。最新广覆盖658项通过。

2026-10-26新增全词评分/执行合同审计已完成：8根全部136词、128条30步
主机CUDA分支、353414个实际predict样本，旧六候选核心、fine8轨迹及历史完整
动作/终态精确复现，全部门通过。统一数值评分改变6/8个决定，旧runtime/新
dispatch/同样修复greedy/CEM均距分别为0.124054/0.128201/0.131864/0.119369。
新dispatch对greedy为3胜/3负/2平且CI跨零；6个新witness仅2个首段真实RGB
非贪心。因此评分合同差异不能充分解释本批失败，不作为正式修复部署，更不能
据此宣布理论失败或穷尽代码原因。原定新seed块的批内episode重复在读结果前被
拒绝；固定替代块202646000..007与历史168个episode隔离，独立确认已在主机
CUDA完成，结果与新发现的目标构造缺陷见本页最新条目。原审计当时596项回归
通过，两张全根图已目视与哈希验证。详情见
[统一数值后缀合同](D2P_STATE_NATIVE_SUFFIX_CONTRACT_20261026.md)。

同轮全库终点选择消融也已完成：全部8根、32条主机CUDA分支、额外16260个
predict样本；原基线终态和4个重复决定的动作/终态精确重放，所有门通过。
相对非贪心优先selector为3胜/1负/4平，但均距0.128201→0.136998，最难根
0.800530→0.919481；其真实终态RGB也恶化，不仅是qpos度量问题。因此这条
选择器限制不是充分修复，不替换默认，不声称理论失败。

同轮CEM搜索覆盖审计亦完成：原128×3/16elite与1024×12/64elite在固定128微批下
比较，8根32条主机CUDA分支、804992个predict样本，全部精确重放和完整性门通过。
完整六包开环计划7胜/1负，均距0.152007→0.123128；固定native前缀仅扩大后缀
搜索却3胜/5负，均距0.128201→0.142566，最难根0.800530→0.925870。
模型成本选native/wide也不能避免该恶化。搜索覆盖是部分原因，不是充分修复；
这是已知根联合搜索干预、非等资源、非逐包CEM-MPC，不开放P1。
最新广覆盖632项通过，全部根图与复现仍见上述统一数值后缀合同页。

2026-10-26新增数值盆地定位：同prefix/同初始动作的8根48条主机CUDA分支全部
完成，前向/解码精确相同、链式法则/FD/原终态重放均通过。根202643006的原
incumbent距离1.476133，tanh/box再32次仍约1.59，同模型CEM后缀则为0.028642。
这排除了“该根严重后缀失败必然是冻结模型没有有效动作路径”的解释，而不是修复
全部前缀排序。模型成本选择的全8根均距0.215253→0.035700，6胜/2负且描述性CI
跨零；不声称物理单调性。排除160个episode的新8根/四类prefix/192条主机CUDA
分支确认也已完成，全部门通过。dispatch均距0.361358→0.124054，greedy为
0.783812→0.121340；新根202644003同prefix从1.623534恢复到0.038918。
修复后的dispatch对greedy/完整CEM的区间均跨零，7个witness仍有1个首段RGB
反转，202644006所有控制仍有高误差；不宣称P1通过或理论失败。相关回归535项
通过，新增全部根的PNG/PDF已目视检查。详见
[动作坐标与多起点修复](D2P_SUFFIX_ACTION_CHART_AND_MULTISTART_20261026.md)。

2026-10-26计划连续性修复：对旧8根全部136个动作词执行408条主机CUDA后缀
分支，原cold动作、轨迹和终态全部精确复现，权重/封存/实现gate全通过。原实现
丢弃imagined suffix并固定从零重编译，cold/warm模型成本incumbent将封存prefix
均距0.458002降至0.124733，6胜/0负/2平。这证明部分闭环负结果来自代码/协议，
并未修复首段RGB排序。排除此前152个episode的8新根确认已完成，80条分支全通过
完整性检查，dispatch均距0.467475→0.215253（对cold为3胜/2负/3平），对同样修复
greedy为5胜/0负/3平，但CEM incumbent均距0.039051。五个witness仍有一个首段
RGB反转；不能开放P1。小样本bootstrap上界对抽样次数敏感，不宣称显著性成功。
完整证据与复现见 [计划连续性修复](D2P_SUFFIX_CONTINUITY_REPAIR_20261026.md)。

2026-10-26继续推进：已完成24个既有任务/根单元的自身残差/Jacobian交换主机CUDA
审计，113个响应矩阵、旧决定与120条完整物理分支精确复现。新诊断未增加witness，
反而拒绝唯一原有弱肩好分支，故不替换部署策略。另实现原提案允许的有限动作顺序：
冻结一包Jacobian生成17个四包动作词，在排除此前144个已测episode的8个新根上，
完成全部136词及16条平坦GN/CEM对照，共152条30步真实分支。多包模型分段与
逐包真实历史gate通过；6个模型witness仅3个仍真实RGB非贪心，dispatch均距
0.458002，对同库greedy的区间跨零，仍弱于完整CEM的0.120545。没有调门或开放P1。
全128个非neutral词的margin为92一致/36反转，64个固定顺序对54一致/10反转。
这补上了程序类型的真实实验，不是排序修复或理论负结果。新代码中的float32秩一
伪方向与绘图NumPy整数序列化问题均修复，但都不能解释历史P1。详见
[残差锚点与动作顺序实验](D2P_RESIDUAL_ANCHOR_AND_ACTION_WORDS_20261026.md)。

2026-10-26最新补充：原32根×四分支×五档动作插值的640次主机CUDA物理执行
全部通过动作桥、模型路径和原后继重放检查，原两处反转仍在。另完成固定4096个
不同source episode索引（排除已用144个episode）的RGB三帧历史检索对照，32根、
两库各32候选，全部根/候选保留。排序一致率0.955078→0.956527，配对95%CI
[-0.005922,+0.009514]，不支持稳定修复。字面零与旧浮点近零仅改变另两根的
RGB，原反转根不变，不能归因于该残差。

此外修复了真正的阶段实现漏洞：P1 consumer不能仅信旧true标志、空完整性gate
或缺失outcome；P1→P2也必须所有预声明CI严格负而非触零即过。8个反例先复现，
现已通过，四份历史P0正例仍有效。神经/编译/执行函数AST不变。最新相关回归
403项通过，两个新增全根图已生成并目视检查。**这些修复不等于原首段排序已解决，
也不构成理论负结果。** 完整证据及两次新审计中断原因见下述RGB审计页。

## 原提案要求与当前证据

| 原要求 | 当前可定位证据 | 尚缺的闭环条件 |
|---|---|---|
| C1：冻结动力学生成带动作见证的程序 | PointMaze/Reacher 的 action-Jacobian、SVD、有界 DLS、实际 action witnesses 及原始权重哈希检查 | 在当前修复后的 solver/context 合同下，完成独立 preflight、校准与 outcome；不能借用旧 SHA 的通过回执 |
| C2：非贪心动作改变未来可达能力并带来真实收益 | 渲染策略修复后全408单元/22576条未来完成；4896拟合及1632两因素拟合独立复核。30个物理诊断候选均被RGB代理拒绝，真实RGB增益亦不支持对应能力排序 | 以模型可用信息处理RGB/物理任务度量合同错位，并独立验证controllability-gap任务族、非贪心、能力改变与真实收益；不能部署oracle候选或以测量完成替代机制成立 |
| C3：动力学决定程序、搜索深度和计算预算 | 临时 program dispatcher 已实现；实际 predict 样本与 VJP 行数已核账 | 当前仍逐根编译全 bank，不是由几何自适应分配预算；需要预声明的调度规则及完整等资源验证 |
| 实验 1：Executable Subgoal Gap | 有动作生成的中间状态及重观察执行接口 | arbitrary latent waypoint、已有 learned subgoal/proposal 和 dynamics-compiled program 在同一任务/底层 controller/资源合同下的真实实现偏差与闭环比较；本地实现或合成 fixture 不能替代 |
| 实验 2：Controllability-Changing Actions | Reacher 两段受益案例；PointMaze-SRA 受限机制；新八根全部64后继已做双尺度全坐标物理探针及独立448增益复核 | 探针现已完成，但普通全驱动任务没有稳定能力改变优势，不能把“已测量”写成机制成立；仍需合格任务族及 CEM/MPPI、GD、GN、GRASP、无组合 SRA、kinodynamic search 的强对照 |
| 实验 3：One Frozen Model, Many Tasks | 同一原始 Reacher 模型已有统一目标接口，另设全驱动与两种动作限制任务 | 多目标、horizon、动作限制下独立的真实 success/cost、耗时及预算结果；新任务 pilot 不等同于完整复制 |
| P0：可微程序生成及执行前提 | 历史 P0、source-native RGB 审计、后缀回放、精确状态修复与有限差分逻辑 | 每个新任务/新 chart 的 CUDA、FD、action scale、完整 replay 合同都须重新绑定；旧 passed 状态不可跨接口或模型复用 |
| P1：重配置 kill experiment | 已修复后缀、非线性代理、目标拓扑与严格渲染回放；完整八根全库归因现定位到能力度量合同错位，原选择仍未建立完整C2 | 当前不通过，不据此宣布理论失败。先以可部署信息处理代理/任务度量关系，再独立校准和outcome；不能调整margin、删除反转根、挑oracle程序或更换成功定义 |
| P2：多程序组合 | 已有两段 prefix--observe--suffix 原型 | 在 P1 前提下实现并验证组合搜索和生成动作见证，检验超过无组合谱规划的真实收益；两段 pilot 不能冒充完整组合能力 |
| P3：动态 horizon / 计算调度 | 旧 SRA fixed-tree horizon 分支有局部证据 | D2P 下由动力学几何决定的 horizon 与预算，及噪声/成本/收益消融；不能移用另一分支的 P3 通过 |
| P4：RGB 闭环多环境 | PointMaze-SRA 有限定 checkpoint、root 分布与固定预算的闭环证据；Reacher/PushT 有分支实验 | Reacher/PushT 的 D2P kill-experiment 与真实闭环仍需各自通过；PushT 接近、接触、推送和闭合分别评价 |
| P5：跨模型/跨任务 | 其他分支存在 teacher/adapter/repair 资产 | 预声明的独立冻结模型/架构复制；后训练 teacher、adapter 或 qpos readout 不得填补 training-free 主线 |

## 已排查与仍需区别的负结果来源

1. **已复现并修复的代码/协议问题**：目标事务、source history/root 支持、真实 successor
   观测、DLS 非线性接受规则、计算量低报、snapshot warm-start 与陈旧证据绑定。
2. **已验证的运行时策略缺陷**：无 witness 时的 neutral fallback 拖累全根结果。新 progress
   fallback 在新 32 根的同根分支比较中消除了这部分拖累，但没有证明重配置的增量效益。
3. **残留的当前合同预测失配**：新 32 根中两个已选 prefix 的 candidate-vs-zero 真实 RGB
   顺序反转。128 分支同根重放已精确复现 observed RGB、actions、cost；这不意味着所有
   可能代码问题被穷尽，只表示这几条被测实现路径当前有正向证据。
4. **诊断替代并非自动修复**：只用上一段已知观测的 unit innovation 将平均 latent L2 从
   1.155917 降到 1.094625，但 eligible 排序仍为 5/7，修正一个反转同时新增一个反转。
   没有采用它作为正式 selector，也没有拟合系数或借它打开 P1。
5. **任务前提未确立**：普通全驱动 Reacher 的 qpos 目标不自动构成提案要求的
   controllability-gap family。固定动作限制任务用于补查这一前提，不是把有利 outcome
   筛成新任务族。若前提未通过，P2/P3/P5 不能以“代码已写好”为由开放。
6. **已定位但未修复的排序差异**：新 32-root 代数分解中，两个反转根的响应余弦约
   0.996，但共同预测偏差与响应投影误差仍足以翻转 goal-cost margin；即便 future-zero
   oracle anchoring 也不能纠正两根。完整 10 维双尺度响应审计在 1,280 个真实干预中
   分开核对了自动微分、同模型 FD、实际 RGB FD，未发现新增的梯度实现错误证据。
   有限尺度/量化误差与预测响应误差仍需区别；这些诊断不能升级为理论失败或 P1 pass。

## 强基线与资源合同

**最新协议修正**：旧 `lookahead_cem` 搜索三段，却只执行第一段并改用公共单次 DLS，
不是完整 CEM-MPC。32 根固定首段因果对照已完成：保留原搜索 tail、无新增搜索时，
两种 CEM 输出均在 32/32 根改善；完整 tail 相对原 dispatcher 为 30 胜 / 2 负。
旧 “D2P 优于 CEM”不能再支持完整规划器优势。实际 RGB 重搜 CEM 后，dispatcher
同首段均距从 0.257405 降至 0.033826（31 胜 / 1 负），但多了 1280 predict 样本；
这修复/定位后缀执行限制，不修复首段非贪心排序，也不自动开放 P1。

当前新 32-root dispatcher 每根为 2062 个 predict 样本、24768 逻辑 VJP 行、约 16.36 秒；
探索性 CEM 对照为 1934 样本、192 VJP 行、约 0.19 秒。同库 greedy 为 142 样本、192 VJP
行、约 0.14 秒。不能把前向样本接近说成完整等预算，也不能将主要由 greedy 达到的 CEM
差距全部归因于 reconfiguration。

现有 gradient-and-restart comparator 明确仅为 GRASP-style，不是已审计的外部 GRASP
逐实现复现。当前合同尚无完整挂接的 learned-subgoal、kinodynamic 与上述所有强对照
回执。正式效益需要冻结资源分配、候选支持、目标和停止准则，并按 root 配对、区分校准和
outcome、控制多重比较；描述性 pilot CI 不替代这些条件。

## 本轮不可变证据入口

- `le-wm-main/results/reacher_d2p_program_dispatch_fresh32_20261025_cuda.json`
- `le-wm-main/results/reacher_d2p_program_dispatch_fresh32_result_audit_20261025.json`
- `le-wm-main/results/reacher_d2p_causal_innovation_fixed_branches32_20261025_cuda.json`
- `le-wm-main/results/reacher_d2p_actuator_limit_tasks_pilot8_20261025_cuda.protocol.json`
- `le-wm-main/results/reacher_d2p_actuator_limit_tasks_pilot8_20261025_cuda.json`
- `le-wm-main/results/reacher_d2p_actuator_suffix_ablation24_v2_20261025_cuda.json`
- `le-wm-main/results/reacher_d2p_margin_causal_decomposition32_20261025_cuda.json`
- `le-wm-main/results/reacher_d2p_physical_response32_20261025_cuda.json`
- `le-wm-main/results/reacher_d2p_source_render_crossover32_20261025_cuda.json`
- `le-wm-main/results/reacher_d2p_cem_incumbent32_20261025_cuda.json`
- `le-wm-main/results/reacher_d2p_cem_execution_contract32_20261025_cuda.json`
- `le-wm-main/results/reacher_d2p_iterated_suffix32_20261025_cuda.json`
- `le-wm-main/results/reacher_d2p_fine_suffix32_20261025_cuda.json`
- `le-wm-main/results/reacher_d2p_suffix_repair_fresh8_20261025_cuda.json`
- `le-wm-main/results/reacher_d2p_refined_selector40_20261025_cuda.json`
- `le-wm-main/results/reacher_d2p_full_bank_refined_fresh8_20261025_cuda.json`
- `le-wm-main/results/reacher_d2p_post_prefix_geometry8_20261025_cuda.json`
- `le-wm-main/results/reacher_d2p_common_geometry_fresh8_20261025_cuda.json`
- `le-wm-main/results/reacher_d2p_geometry_qualification_causal8_20261025_cuda.json`
- `le-wm-main/results/reacher_d2p_response_qp_convergence8_20261025_cuda.json`

动作限制 pilot 已完整完成：固定 8 个新根、8 个独立 source episode，与此前 104 个
episode 不重叠；三条件 24 个单元的动作、重放及有限差分 gate 全部通过，原权重未改。
full/weak-shoulder/weak-elbow 分别产生 0/2/2 个 witness；全根相对同库 greedy 的终态
距离均差为 0/-0.024062/-0.001586，仍是小样本 pilot，不是正式 P1。

后缀因果消融也已完整完成。24 个单元 × 3 个方法共 72 条完整 15-step witness，终态
qpos 距离全部精确复现原分段执行。保持同一 5-step prefix、仅将后 10 步置为 raw zero，
有 2 个 eligible 单元满足“被动运动较差、优化后缀后反超 greedy”；另一个 shoulder
案例虽最终获胜，后缀控制本身反而增加 qpos 距离，不能算作控制能力改善的正例。
这区分了惯性运动与控制贡献，没有确立物理可控性定理或独立效益。

首次后缀审计在输入预处理前因 renderer 的负 stride RGB 视图中断，协议保留；v2 只修复
数组连续性，图像取值与朝向不变，未调整 gate。该新脚本错误没有产生旧 P1 的排序结果，
不能把它归因为历史反转原因。

上述主机 CUDA 诊断已完成，未训练额外组件、未改变模型、未用 CPU 替代神经实验。
后缀重线性化审计也完整完成：原始首次动作/终态精确复现；固定首段后，dispatcher 的
1/2/4/8 次 SVD 平均终态距离为 0.257405/0.183632/0.133556/0.101150。8 次对原单次
29 胜 / 1 负 / 2 平，仍弱于 observed-CEM。该改动只定位后缀局部求解不足；新 solver
不替代原 P1 合同，非贪心首段排序未变，不能因此开放下游阶段。

更细步长救援也已完成：5/5 个 dispatcher coarse-stop 均存在模型下降且真实执行
改善，其余 27 根动作不变；fine-8 均距进一步降到 0.083214。同库 greedy 则降到
0.080737，其模型下降不保证所有物理根改善。该证据确认了粗网格过早停步的实现限制，
不确认重配置独有收益。新根 8-episode 三版本确认与真正 CEM-MPC 已完整完成；
与此前 112 个 episode 隔离、重新核验完整 93 GB source SHA，所有 gate 通过。
Dispatcher 原单次/coarse-8/fine-8 均距为 0.226112/0.118124/0.057419；fine-8 对
原单次为 8 胜 / 0 负。真正 CEM-MPC 为 0.032064，仍不能声称胜过强基线。唯一
重配置根在 greedy 后缀充分求解后失去相对优势，未建立非贪心重配置的独有收益。
因此更新为“后缀数值修复得到新根证据；P1/研究闭环仍未完成”，不是理论负结果。

选择器与后缀一致性的 40-root CUDA 因果审计也已完成：8 个旧 witness 有 4 个在
模型内精化后失去收益，这 4 个实际均不如 greedy。重检策略不使用真实 outcome，
全根对旧 dispatch+fine8 为 4 胜 / 0 负 / 36 平；对同库 greedy 的描述性 CI 仍跨零。
旧两个首段排序反转仍保留，不能称 P1 已修好。该审计只重检三个旧分支；完整候选库
的一致求解器和全新 8-episode 对照已完整完成，与此前120个episode隔离。
选出6个witness，首段真实RGB对zero非贪心6/6，qpos为3/6；全根对同库greedy
3胜/3负/2平，均差-0.011168、描述性CI[-0.077904,+0.049450]，不支持稳定收益。
对真正CEM-MPC为3胜/5负，均差-0.012038、CI亦跨零；不能用少量均值优势声称
强基线胜利。平均predict样本/VJP行为dispatcher7923.75/71808，greedy324/1536，
CEM1920/0，明确不等预算。原始权重、完整source SHA、执行身份和全根保留均通过。

后继物理几何诊断也完整完成：8根×3分支的24条完整动作见证终态和首段距离精确
回放；对10步后缀20个raw torque坐标，以±0.1和±0.25分别探测全部方向。两个
相同尺度的共同greedy目标残差下，仅2/6个witness改善有界二次控制增益，3/6改善
最小奇异值。因此“更优的优化后终点”并不自动确立“改变控制能力”；这是见证合同的
机制缺口，不是理论失败。物理qpos/Jacobian仅作评估，绝未进入规划器。

新的独立命名compiler在冻结模型内增加共同目标响应资格：保留完整fine8终点比较，
所有可能通过者与两个对照都在同一greedy原点目标残差下比较有界局部响应增益；
需严格优于两个对照，不用真实物理标签，也不改原首段margin。另8个新根与此前
128个episode隔离的主机CUDA比较已完整完成，所有完整性gate通过；无候选通过，8根
全部精确回退greedy，对greedy无增益，对CEM为3胜/5负、均差+0.056086、CI跨零。
不能冒充原P1修好。同根固定决定重放已完成：改变4个决定，对仅看终点策略3胜/1负/
4平、均差-0.127437、CI[-0.447880,+0.097605]；拒绝的202631005其实优于greedy。
另发现新1000步box-QP在latent上未充分收敛，投影残差约1e-3；新增active-set SPD
box-QP、目标误差证书与上下界比较，同矩阵CUDA审计已完整完成8根/22矩阵：旧分数
逐项精确重现，增益最大修正4.148158、目标duality gap上界最大4.285461e-13，所有
决定不变。该数值缺口不解释本批零放行，也不解释历史P1。零放行不证明物理机制不存在。

新的certified compiler已接入原先固定的三任务族（全驱动、弱肩、弱肘）。8个新来源
episode与此前136个不重叠，三任务使用同一根组、完整CEM、各自FD/动作限制和状态
回放，主机CUDA顺序矩阵24个单元已全部完成、全部gate通过，identity-joined reducer通过：
全驱动0 witness，弱肩1 witness（202633002）同时满足真实RGB/qpos对zero非贪心，
首段qpos距离0.966983也大于初始约0.919239；终态dispatch/greedy/CEM为
0.608673/0.634703/0.616283。弱肩全根对greedy均差-0.003254，CI[-0.009761,0]，
对CEM4胜/4负、均差+0.001238、CI跨零。单根不替代整体证据和实际控制几何。
弱肘0 witness，dispatch/greedy均距0.489182、CEM为0.532249；全驱动/弱肘均精确
回退greedy，相对CEM差异不能归因重配置。三任务物理几何均已完成，所有完整动作
和首段距离精确回放。唯一弱肩病例在两个尺度相对greedy改善共同目标控制响应，
但不如zero，故只有部分机制支持，不是P1通过。任务支持与响应坐标在
协议中显式区分，不把受限参数包误称为官方原始raw packet。
P2/P3/P5 仍不从此 Reacher 链开放。

前批相关代码逻辑回归 244 项通过；新增 crossover / CEM 保留审计后为 264 项，后缀
执行合同和迭代 SVD 后的合并回归为 283 项。
细步长救援加入后为 286 项，新根确认脚本加入后为 289 项，完整候选编译器加入后为
308项；代码测试通过不构成阶段通过。此前物理几何/共同目标响应模块的集中单测23项
通过（与完整候选测试存在重叠，不相加），扩大Reacher回归321项通过；新代码最新
测试和CUDA作业状态以新增审计段为准。新增数值证书、固定任务及identity-join reducer
后为342项通过；后续阶段门修补与AST作用域回归后的广覆盖为358项通过。
新发现旧P0→P1准入将触零CI误放行，现要求严格负区间并绑定六个实现gate；
修补只影响标量准入与main，不改变神经/执行函数。四份已检查历史P0正例原本满足
严格区间与完整性条件；此漏洞不解释旧两个首段反转，不改写历史回执。
测试通过不是研究阶段通过。完整研究目标和正式 Reacher
P1 仍未完成，不能把本轮新审计完成写成整个研究闭环。

详细代码因果与数值表见 `REACHER_D2P_SOURCE_NATIVE_RGB_AUDIT_20261025.md`；历史分支记录见
`SPECTRAL_REACHABILITY_ATLAS_RESEARCH_PLAN_20261023.md`。

## 2026-10-27 并行后续阶段的新增闭环边界

最新运行状态见`D2P_PARALLEL_CUDA_PROGRESS_20261027.md`，不以本文历史测试数判断进度。
P2配对组合四根/八分支真实执行80步与8次CUDA编码已经完成；模型Q改善4/4，
真实编码Q改善2/4、pose改善1/4，尚不能证明组合独有收益。独立160个suffix KKT
复算未发现求解残差/动作见证接线错误；仍不是穷尽所有代码错误或理论否定。
P4两臂各20步三轮反馈已完整执行与journal独立复算，actual反馈改善角度但pose
比imagined更差68.170945，双方任务success均false，不能把七维含速度距离下降
替代物体位置或任务收益。

P5环境版本不兼容已通过39.55MB隔离依赖实际解决；新鲜同seed两个独立原生环境
的零动作和非零动作各16时刻全公开状态/RGB精确重放均完成。原生official模型
接新鲜非零数据的主机CUDA v11也完成：actual对mean/negative更好，对reverse更差。
它接通了可审计物理输入→冻结模型factual预测链，不证明prospective控制或跨架构
机制成立。主线程修复失败证据丢失/attempt漏计/收尾遮蔽原错的代码问题，v11重跑
与v10成功输出完全相同，不能将该修复当作排序问题已解决。

后续P3 v4也已完成实际80步/8次CUDA编码，四对完整H2的dynamic−uniform pose
为0/+71.857102/+11.635172/0，仍无动态预算任务收益；模型到编码排序与编码到
任务指标均有不同失配，独立代码与算术审计继续。P5另三固定新seed的96帧/906
原生物理子步精确双重重放、v12三根CUDA全部完成；actual对zero/negative均更近
事实终态，对reverse仅1/3更近。它补齐多根输入→预测诊断，不等于前瞻控制。

剩余核心要求未缩减：独立P1非贪心能力改变与任务收益、与强方法真正预算可比的
P0/P2、P2组合增量、P3物理质量—成本优势、P4反馈机制收益、P5独立架构与三个
原方案核心实验。文献reported表可引用，不替代同协议本地因果比较。阶段门与整套
研究目标保持未完成。

### 后续实际运行更新（同日，以新报告为准）

P5前瞻C1已完成独立goal→HOST CUDA/JVP/ridge编译→三分支freeze→原生真实
执行全链；executor清理预算漏洞在任何该三臂physics前修复为新v2，重新绑定
协议与CUDA编译，不改旧sealed字节。独立物理saved-data审计542项全通过，
但三臂suffix物体均精确静止、coverage0、无接触，故不成立物体任务收益。
独立HOST CUDA40predict中央有限差分与JVP相对误差约3.64e−4/5.20e−4，
另91项saved-array复算通过，未发现导数/动作接线足以解释该结果的错误。

P3/P4新goal-only patch度量独立422项数组/goal mask审计通过，随后8计划实际
CUDA双重重放、8分支90步真实执行、8次CUDA终态编码完成。两对同H2的
新减旧pose=−57.8736070612/−3.0984134738；另两对H2/H3分层且不混入同预算
结论。八臂official/object-only success仍false。这支持继续检验目标度量假说，
不等于P1机制、完整一致compiler或独立泛化成立。

GRASP H1官方adapter机制子集实际CUDA八臂完成21648predict/action encoder、
7200backward；其原始计划4/8越界完整拒绝。新真实执行consumer完成其余
4臂60步，仅1根完整配对pose改善39.36598，不报告选择后的总体胜率。
详细路径与资源边界见同日并行进度文档；核心剩余要求和阶段门不缩减。

### Q 一致条件编译与比较协议复核（2026-10-27 后续）

完整 conditional full10/fixed-H2 新旧Q编译已真实HOST CUDA完成四根八臂，
不是score-only重排：528predict、528action-encoder、88fresh Jacobian、96SVD，
模型未变、错误为空。实际报告位于
`le-wm-main/results/p3_goal_patch_consistent_compiler_20261027_v1_cuda_main_v1/report.json`。
主线程另两个纯数组审计器实际重建所有bank/solve/cost，及112份context的严格
float32 normalized history内容。保存接线检查通过，但八臂selected均等于各自
stage-greedy，尚无选中非贪心前缀，独立新根数仍零、真实执行还需新consumer。
不能将本批无非贪心见证泛化为理论不可能；应核对根协议是否满足原P1前提。

独立fault-injection审查发现该v1的异常收尾/partial数组保存窗口、history verifier
内容漏检；旧seal不动，另写v2修复。实际成功运行未触发前两类异常，history
已补独立内容审计；不把这个事实写成所有代码问题被排除。

P2组合独立saved-data预算复核确认四根standalone预测次数为无组合88、组合168；
故即使同H2/同10真实控制步，也不是等预测预算，更非等总算力。
新predict-matched frontier准备中，禁止空跑padding或将论文reported表代替这个
方法内部的组合因果对照。P4独立审计确认仅一批两臂20步/三轮，而非两批执行；
step20真实latent未保存，正在从封存真实RGB准备独立CUDA补编码，不冒充新物理根。

### 上述后续的实际完成状态（同日，以新报告为准）

完整Q一致conditional程序已实际八臂80步执行与8次主机CUDA端点编码完成，
四对同H2新Q−旧Q pose均改善（−20.95/−59.17/−35.30/−67.74），但朝向三根
变差、八臂official/object-only success全false；本批未选中非贪心前缀，故仍非P1。
独立新v2的R1–R3代码修复也完成真实主机CUDA528预测/88J/96SVD重跑，完整
数值编译对象与v1八臂精确相同；修复留证漏洞不等于修复行为效果。

P4两张原始step20 RGB已实际主机CUDA编码，native192派生数组完整落盘、
模型未变、无预测/AD/新physics；后续auditor将独立接入该派生receipt，不能继续
把末端编码说成未运行，也不能追改旧v1当时缺失记录或把两臂计为两次新批次。
路径、hash和所有资源数见并行进度页最新节。完整研究闭环与原阶段门仍未满足。

### 2026-10-27 constructive校准续报（MAIN代码审计，非独立确认）

新增真实证据详见`D2P_PARALLEL_CUDA_PROGRESS_20261027.md`最新节：
共同probe v1实际92编码后0预测被exact gate拒绝；MAIN同PNG主机CUDA交叉诊断确认
默认SDP与MATH编码末位差异，旧task未固定SDP。这是已定位的数值协议代码问题，
不是模型/理论负结果。v2保留旧现场，增加4次原协议exact认证、92MATH正式编码和84预测，
已全部完成、权重未变；4原锚认证通过，MAIN独立中心secant复算差0。
局部方向部分吻合，但两根跨prefix能力投影幅度增减均与实测反向；G=D仍无非贪心证据。

gradient MPC已完成272预测/64标量反向→两根新真实执行→4次主机CUDA真实RGB编码链，
其中一根满足原生四维XY及wrapped角度成功判据，MAIN独立复算确认。
因此当前两根不能证明D2P超越普通动作梯度优化，且两方法不是等计算预算；
1/2不是独立泛化成功率。项目nominal MPPI完成144预测与两根真实执行0成功，
它不是官方density-corrected MPPI复现，尚待新真实RGB编码；不得借此补论文强基线缺口。

同轨迹overlay的F/Z/M条件消融另完成32编码/24预测，F对Z、M各7/8较低；
M仅屏蔽outgoing embedding，Z/M没有反事实物理标签。不能用这些结果认定训练overlay根因，
也不能说冻结模型完全忽略动作。原P1独立非贪心机制、强基线等预算、P2/P3增量及
三组多环境/跨模型闭环要求全部保留，当前仍未完成。

### 2026-10-27 MAIN后续：MPPI RGB已补齐，GRASP模型计划已真实生成

上节“nominal MPPI尚待新真实RGB编码”已被后续完成产物更新：两根4图主机CUDA
编码完成、模型未变，6项artifact与全部NPY meanQ由MAIN独立复核一致。
报告SHA `d730e5845b0d06edbbfb00a6dc1f6490ac4d5e6662ceef6abae3588fbeaf3dcf`。
0/2真实成功仍保留，不调参数/阈值替换，也不据此否定所有MPPI或补强基线忠实性缺口。

GRASP_H3_bounded新增实际主机CUDA两根3604预测/3604动作编码/1800标量反向，
模型未变、errors=[]、原consume通过，MAIN另核动作/范围/数组meanQ及6项artifact。
报告SHA `501651be13c6dd6b6082bd20513d464c0ffc8aed053bb7a40f1c4ee9cd9388c5`。
仅模型计划，不是自身真实控制或论文官方数值复现；真实执行适配器尚在实施。
按用户最新要求不设专门审计子智能体：MAIN负责代码审计、运行准入及结果核验，
并行子智能体负责独立实验/基线实现。完整研究目标与P1门仍未通过。

### 2026-10-27 MAIN再续：GRASP真实链完成，继续定位部署context问题

GRASP_H3_bounded在自己封存的两根10步命令上已完成主机真实执行与4张主机CUDA RGB编码。
物理SHA `bf7f15cf9d5aad4ca7923c4a116f6703c5e559b496c50488de55a4a94394f606`，
RGB SHA `c1a74f179fc367e45e65e8e38c429d4b8a416efa9f1dd74453e6aa838d1dfd77`。
全部initial_exact/full10/errors=[]、模型未变，MAIN独立成功判据和NPY meanQ复算通过。
0/2原生成功不能被物体终距或latent改善替换；不声称官方数值复现或等总算力胜负。

模型4半径响应新完成324predict/action_encoder，84原点完整context逐位重放，
SHA `5379546ede547fe72af2b6a7182960d51b8c822f351531175e48092594bb5d0b`。
MAIN独立16个中心secant数组差0；四半径的跨prefix模型幅度增减方向一致，
但新增半径无物理标签，不能据此排除物理接触非线性或确认理论负结果。

新增固定archive4/live4输入描述SHA
`8d8ec5b20938a619179234a16d9e8736aa93b340693ce9066edb589e33b85b20`。
live每根三个历史RGB完全重复，历史两packet确为真实zero warmup；archive W10则均变化。
这是真实部署context差异，不是已经证明的padding代码错误或checkpoint训练因果结论。
后续固定当前状态和outgoing factual动作的历史状态/动作消融正在实现，不用outcome选窗口。
原研究的C1/C2/C3、三个实验族及P1独立门均保留；当前仅完成更多真实评价链与根因证据，
没有把工程完成重新定义为完整研究闭环。

### 2026-10-27 MAIN历史消融续报：不能把静止历史差异直接当根因

原定固定archive-history F/S/A/SA新32预测已主机CUDA完成，8原F窗口完整输出逐位复现，
模型未变、errors=[]；报告SHA
`2d20dadc8ff67515f59ba3495731d4d90a93f49d3fbe57b1cd7d4f4c53fff783`。
MAIN核229artifact、全部输入干预、128实际module attempt/return及32NPY MSE复算差0。
F/S/A/SA平均MSE分别.003893315391/.004302927737/.004498737894/.003837776112；
同时制造静止旧状态/零历史动作的SA平均未恶化。因此“静止历史本身解释live失败”没有获得本实验支持。
消融均对相同factual label评分，没有新的合法物理轨迹或因果识别，不否定全部history影响。

真实非零warmup的新source/encoder/planner适配按预声明最小协议实施，明确旧zero-only消费者不能直接复用。
这会改变warmup终态和构造goal，仅是复用原placement recipe的校准测试，不是同root配对因果比较或独立确认。
不据结果事后选幅度/窗口/根；原P1非贪心与真实收益、强基线等预算和跨环境/模型闭环仍缺。

### 2026-10-27 MAIN非零历史续报：真实源与盲编码完成，规划尚待执行

新moving-history源两slot均通过两次独立重放，120public/1224physics/126render、errors=[]；
sourcepack SHA `c2d8d573f94884ffc081163f25f2bfa269f98ffe8155887704109219910ffec6`。
MAIN独立核3428事件哈希链、126PNG、两slot真实raw/三帧history与三phase动态重放一致。
新盲编码57tests后主机CUDA8encoder/8projector完成、模型未变，报告SHA
`e7fb554327b3bb5a96190f6291f8450b2b0723cd981817715cfb7e50aa266bf6`；
MAIN核12artifact SHA和全部NPY/latent/非零归一化历史接线，复算差0。
这仍是旧placement recipe派生的新root/goal校准；并非独立确认，也尚无本批GN/D2P真实控制结果。
旧zero-only consumer不可直接使用；新planner/真实warmup执行器由子智能体实现，
主线程独占代码审计、实验准入及结果核验，不专设代码审计子智能体。P1及完整研究闭环均未关闭。

### 2026-10-27 MAIN非零历史端到端续报：规划/物理链完成，P1未通过

新wrapper 41tests后主机CUDA实际288predict/288action_encoder完成，errors=[]、模型未变；
report SHA `cd8432c06288577d25d9d9b92c7eb3673de592499ee64cae93238156f3f19aa0`。
own consume重算全部实际非零history model-input/候选/AD/selection/replay；
GN78predict/root、D2P66/root，非等总预算。MAIN独立动作tanh/球/terminal meanQ复算差0。
真实执行器39tests后四initial_exact/full10、errors=[]，816physics/88PNG；
report SHA `a5322dc4f483e82b5ed9d7b3996cf8a63845261d65774f90a4ae341773016aaf`。
MAIN核2354事件链/88PNG与实际动作；GN/D2P均0/2原生成功。
D2P两根四维终距56.85/56.02较GN89.03/213.12小，但任务改变且复用recipe，不能外推泛化收益。
两个D2P winner6/7均为bank即时贪心，不满足非贪心机制前置条件。
非零历史不是本轮已证实的修复；真实RGB与固定factual successor保真诊断实现中。
完整机制、独立确认和强基线/资源公平等未解决条件不由工程链完成替代。

### 2026-10-27 MAIN factual successor诊断续报

固定两moving-history构造轨迹的真实主机CUDA2encoder/8predict已完成，29tests、
errors=[]、模型未变；报告SHA `81bd7359b40216aae3d224f5d97a381d6f30ac73e0d15dbaea7d152d591e311f`。
MAIN核67artifact与全部actual输入/8NPY MSE复算差0，0AD/物理/训练。
slot0/1 root首跳MSE .41361/.35439；真实prefix接入后terminal .10598/.14575，
完整两跳想象terminal .46263/.12655。全部优于各自persistence，但teacher forcing改善非一致。
不能由此声称静止history或滚动误差是唯一根因、模型不具动作响应或P1已通过。
后续真实选中轨迹RGB评价及固定prefix的观测反馈重编译作为独立校准推进；
原非贪心机制、真实能力改变、独立泛化及完整三实验族仍未闭环。

### 2026-10-27 MAIN目标轮廓协议干预：预测误差显著下降，尚非P1闭环

同body off→on→off物理对照固定archive episode1/6229原lag0动作，
10checkpoint完整snapshot/goal不变；RGB平均MAE从1.970554降到.411711。
随后实际主机CUDA三condition factual对照完成30编码/24预测、0AD/physics/training，
errors=[]、模型未变。H1/freshH2/teacher-forced三mode nativeoff→nativeon平均MSE
分别.02360528→.00806386、.03901473→.00963173、.06876959→.02032199，下降65.84/75.31/70.45%。
MAIN独立核522artifact/516数组、156module ledger/577事件和全部24MSE/预处理/动作接线。
报告SHA `d8cbd4b6e8ff1497421e6cdaf8644f4b39ed2cd2b695bb10cd15b57fbae121f0`；
独立分析`le-wm-main/results/p1_archive_target_overlay_fidelity_independent_analysis_main01.json`。
说明代码/显示协议贡献不能忽略；仍非checkpoint训练配置认证、规划排序修复、独立确认或P1通过。
默认绿色轮廓不是构造任务goal，不改变物理目标/阈值。后续以新协议固定同根/同动作检验规划相关性，
不覆盖旧封存结果。MAIN负责代码审计，子智能体只负责界限明确的新模块实现。

### 2026-10-27 MAIN moving配对观测已执行，待CUDA预测/排序评价

同原moving根/任务/动作的新source实际主机完成6full20/186PNG，report SHA
`85ca1df3de910628bfd6c5bd1a0a482f61880a2ee76bfd404732a77dd8c7666a`；
MAIN核3520事件/120实际命令/30同bodytriplet/四盲包，off全精确复现，on goal来自真实witness终态。
固定GN/D2P配对计划另完成4full10/112PNG，report SHA
`d40c32e8c59df7c5552937cf76ee43525ae2aae0b595340cd325662b8ea686c0`；
MAIN核2434事件/80命令/88旧off/12triplet及原终态指标，仍0/4native success。
生产阶段0神经调用；新CUDA consumer仍在实现/修真实事件接口，不能声称规划排序修复。
此为复用根的观测协议校准，P1及后续科学前置不由数据采集完成打开。

### 2026-10-27 MAIN续验：配对CUDA与三臂真实比较已完成

上节“待CUDA”已经被实际产物更新，不再作为当前状态。factual/plans两phase实际CUDA
分别20编码/16预测、32编码/24预测，模型未变、errors=[]；MAIN独立复算896artifact/
872数组、40误差、16目标Q与8排序。slot0的开放H2排序off/on都反转，teacher跨算法排序
off/on都一致；slot1均一致。观测轮廓并未修复slot0开放滚动排序，不能称P1已过。
此处发现并修复handoff SHA与底层CUDA报告SHA混用的工程准入错误，旧prepare保留，
新prepare02运行；没有把该拒绝当科学负结果。完整证据见
`le-wm-main/results/p1_moving_overlay_independent_recomputation_main02.json`。

gradient与CEM/本地nominalMPPI实际CUDA搜索已完成，随后预声明三臂同两根物理：
6full10/132PNG/1224内部physics，errors=[]但0/6 native success；主机CUDA再编码12真实
prefix5/terminal10，errors=[]、模型未变。MAIN核物理3530事件/132PNG/60实际命令，
独立复算梯度66候选/64更新与RGB12指标。证据
`le-wm-main/results/p1_moving_baselines_independent_main01.json`、
`le-wm-main/results/p1_moving_baseline_rgb_independent_main01.json`。
报告SHA/预算/终态Q见`docs/D2P_PARALLEL_CUDA_PROGRESS_20261027.md`最新续报。

这补齐了本批三臂的搜索→真实执行→真实RGB评分链，但不是原研究完整闭环：
旧两根、预算不同、nominalMPPI非官方复现；P1非贪心能力改变与泛化仍未成立，
无组合/GRASP/kino的moving完整执行比较、P2/P3/P5及多任务/跨模型证据仍有缺口。

### 2026-10-27：六基线与连续反馈的新鲜执行续报（覆盖上段对应工程缺口）

六基线（gradient/CEM/本地nominalMPPI/kino/本地GRASP bounded/无组合谱bank）已完整接齐原两根12模型计划、12物理full10与24真实RGB编码，原生成功0/12。MAIN已独立重核源报告/selection、24NPY及24均值；实际中文逐根MD/CSV已输出 `le-wm-main/results/p1_moving_baseline_reader_table_20261027_v1_main01/`，不是仅框架或mock。

连续反馈新driver已在主机CUDA与同一episode物理流程实际执行固定12分支O/I/R，完整24编码、40预测、8J/8SVD/2448physics/264PNG；模型未变，errors=[]。MAIN独立核8334条journal、264PNG、24NPY及先观测后编译后step6的顺序。四配对层R−I的真实终态latent Q皆下降，但R−O第一层上升，且native0/12；不构成P1非贪心/能力改变机制通过。报告SHA `990843e8bba15955db5163b9b5d77c1f8eecaa1870d3745b49d46cc22ca895e3`。

代码归因明确新增一项：analysis v1误要求5维原生状态而实际为7维，MAIN实际运行捕获，未发布分析结果且不计理论负结果。保留v1，新v2按冻结原生全7维distance、前4XY+angle成功判据最小修复，32项回归与实际同report分析均通过；结果 `le-wm-main/results/p1_moving_live_feedback_analysis_20261027_v2_main01.json` SHA `5df7dd90b3da010d1bafbb66198334272ec6470b7b0f7bc62991e4b0dae0b5d8`。未重跑/改写原神经与物理产物。

预算调度四策略实际主机CUDA520predict/2J/384grad/2SVD及8分支物理已完整结束；MAIN独立复算两根权重162.3240517/156.2867147，按预声明half-up整数分配均[10,10]，所以D/U/S/N无分配差异、赢家相同。这是该校准panel的干预区分度缺口，不是谱调度理论否定或应按结果改配额的理由；也不等于完整动态horizon/C3已验证。报告和计账详见CUDA_PROGRESS最新节。

仍未闭环：独立新根全历史排重coverage、P1原科学机制、强基线等总资源、P2组合增量、完整C3/P3动态时域与预算、跨任务/模型及适用P5证据。六worker继续轮转实现/历史authority/对比文献资产，MAIN亲自审计、主机CUDA执行与复算；不因代码或表格数量关闭原研究goal。

### 2026-10-27：teacher factual真实CUDA定位与预算续验

Teacher v1在任何模型调用前因prepare序列化超256KiB上限被拒绝；保留原版，新v2无损有界封装修复后实际主机CUDA完成12条自身factual轨迹预测，模型未变、errors=[]，无训练/AD/新增物理。报告 `le-wm-main/results/p1_moving_live_feedback_teacher_fidelity_cuda_20261027_v2_cuda_main01/report.json` SHA `7323347a9e74c8f1751550642503426d07ddce55549ddaef4331f3c362567128`。

MAIN独立复核124artifact、170事件和全部12条动作/目标/评分；四条R完整3步输出与原compiler new_suffix_replay逐元素一致。四配对层R−O预测/真实编码目标距离方向3/4反转（层0/2/3；只有层1同向）。这排除本次teacher与controller接线漂移解释，尚不排除全部代码/协议因素、度量对物理任务的偏离或开放滚动累积误差，不能推导理论不可行。仍只有两旧校准根，没有P1科学准入。

Scheduler真实16图CUDA与8行综合已接齐；四策略配额相同导致实际Q相同，不能作为动态预算负结果。六基线完整预算矩阵已实际生成，15对等总资源均未建立，旧结果不充当公平等预算比较。历史resolver480/480来源捕获与6/6物理身份桥已完成，但8旧来源reset上界和family覆盖缺口仍需保留；不把采集/代码测试通过当独立根认证。具体SHA、计账和新并行状态见CUDA_PROGRESS与DISPATCH末节。

### 2026-10-27：真正单步误差与开放滚动分开定位

MAIN完成新固定12episode×3forward的实际主机CUDA，32项实现测试通过；report SHA `e319035d9e13a6ae2dc4cbfa5cefb9eb2e6ccd49dbcf35408fda1704f5f34253`，36AE/predict/predictor/pred_proj、errors=[]、模型不变，无新encoder/AD/physics。ownconsume及不导入执行器的独立复算均通过352artifact/482事件/916输入绑定/144module entries；12个teacher结果与旧H1逐位相同。

同原两根四相关层，真实前缀条件的suffix H1预测排序3/4反转，替换为预测前缀的H2变为4/4反转。均值factual MSE由H1 .0742253192变为H2 .1916894369（非逐条单调）。这说明滚动可能加重误差，却不足以解释原本已经存在的H1排序反转；不能只用“三token误差积累”归因。完整独立算术报告 `le-wm-main/results/p1_moving_teacher_horizon_independent_MAIN01.json`，SHA `371c59c40bb67482257ba210f2ac5d1a71e78c2970a7d9fff8463d3a914130c7`。

仍需核对原LeWM cached rollout与当前手写packet接口、checkpoint训练normalizer来源及任务度量错配，未声称所有代码原因已排除。并行的新P2队列/动作见证图核心只有实现证据，不能取代独立P1或组合增量科学验证。原scope和全部未完成项保持不变。

### 2026-10-27：官方API实测等价，扩大实质实现并行

MAIN实际主机CUDA24次官方cached `LeWM.rollout`已完成（24AE/36predict/predictor/pred_proj，无训练/新增编码/AD/物理），report SHA `71b67ec61f9a054ebf4ed99ab24ca5899307a3075ce45611fa34fe95e0a5caef`。24/24完整输出及36个step的全部关键模块桥与旧packet路径逐位一致；ownconsume与独立460artifact/1273绑定/482事件/156ledger复算均通过，独立报告SHA `d339ce200acd664f664c1f5897de6d10402581e161fc17b3f3f7f63b198bb87a`。本批失配不能归因于这两条rollout路径数值不同；不是所有代码问题均排除或理论否定。

公开metadata实际刷新：官方超时，镜像commit `22b330c28c27ead4bfd1888615af1340e3fe9052`的权重LFS声明和config字节与本地一致，未取得绑定训练run的normalizer统计。report SHA `4beda578da61c972b8468171a74aa7462f9d3d0fad1110a22717ff3b0b2e221c`；训练合同unknown保留。

空闲工作位已复用为overlay完整重编译、真实执行、RGB CUDA评分、配对比较、预测预算合同和严格同预测调用数采样基线，MAIN继续亲自执行和验收。新实验实现/测试尚不能充当科学结果；仍为两个已知根，未通过独立P1或完整研究闭环。详见20261027 CUDA_PROGRESS与PARALLEL_DISPATCH末节。

### 2026-10-27 MAIN真实能力探针完成，P1仍未准入

上述overlay与采样实际CUDA、物理、真实RGB链已完成并验收，不能继续标“只有实现”。另完成固定40分支能力探针，真实report SHA `e075d3b75c7b5a16c47d8c76c846eaa2ac4d8ebd202abd18037a7242792b076a`；40/40完整、errors=[]，saved分析SHA `19661bb010662dcfc7c9397723ecb085080fe18a7de069bdaac1f7c59e0e7557`；MAIN独立32矩阵/27922事件/资源复算报告SHA `e4f668ce10f4389019f04d4fa507a6efdfa66158fc7a1609feeb74d23420d43d`。

新证据支持“prefix改变真实后续响应”，但不支持“总是改变为任务有利的响应”：root0的D2P固定探针没有正向object任务投影，root1则有。无符号谱/响应增强不可替代有符号任务收益。全四库40候选另行saved复算仅1库因线性与非线性对应评分产生winner差异，未执行新winner，不是已验证修复。源码符合声明的线性proxy，不能称新接线bug或归结全部负结果。

真实执行不是fresh确认；模型对响应方向是否正确、训练normalizer来源、独立根排重、强基线预算、完整P1、P2组合增量及C3/三实验族仍未完成。当前未新建子智能体；四侧边栏对话只有交接文件，尚无工具创建证据。详见[D2P_MOVING_CAPABILITY_PROBE_MAIN_20261027.md](D2P_MOVING_CAPABILITY_PROBE_MAIN_20261027.md)。原goal保持active。

### MAIN固定能力方向CUDA与完整任务轴复核已完成

新主机CUDA40编码/80预测（14.441633秒，320模块收据）report SHA `eac431fca369c977c57d38a32a600d1ef02ae2030b4112636d4c3df92bf143fe`；独立saved桥接与排序report SHA `f564c6771c1940dc3f6925999647569670a9dbbb4381bb6da5a37a5636a0c5bd`。8锚点真实teacher prefix均降低预测MSE；固定Q pairwise符号一致62/80→72/80，但仍有反转，两个旧根不能支持独立泛化。

完整原生位置目标包含agent+object，不能将object-only方向无进展直接当官方任务失败。新saved40分支任务轴report SHA `ee0fdadbd99de36681bd3dbf93a4cbcbd77b4e69dbe7b8a73b35a93d1bf39bc2`：32非zero对照中真实Q方向与联合位置改善一致30/32，teacher29/32、open24/32；原生成功0/40。raw-angle历史公式保留，wrapped诊断不改旧标签。

这缩小了误差来源范围，支持后续严格的观测反馈上下文刷新验证；尚无新反馈优化控制成功。原P1/freshness/强基线/P2/C3与三实验族仍未闭环。用户最新授权恢复子智能体，四工作位已续接强基线/独立根/P2/P3；此更新取代上文派发方式快照，不表示支线已交付，审计与CUDA仍由MAIN负责。详细证据见`D2P_FIXED_CAPABILITY_DIRECTION_RESULT_MAIN_20261027.md`。

### 2026-10-28 MAIN：实际24后缀修复、物理与验收已完成

v1 CUDA/NumPy tanh重建ULP差异导致235次token不等，保留失败且未放行物理；v2保存实际CUDA raw buffer，24case/312forward/936模块严格消费者通过，report SHA `fc620c9fdcbb5046c1d076c79cd0703da999275ec921de799f3747c0204d69ab`。

物理24分母全部执行、工程错误0，23完整后缀+1原生成功早停；实际report SHA `cc92465141b400659e024e052b05683153b4dc6c013b81bdf1e3e29355fd2663`。全saved journal/PNG/51前缀/raw动作bits消费者与原生指标复算通过，分析SHA `970bb7071d3bfb25ab5d01e5b4e87ebc07377b01e52c704799372739d2fa6c8f`。open/teacher_full/teacher_small原生成功分别0/8、1/8、0/8；联合位置误差相对原后缀均值变化-39.91193/-54.68591/+7.14731px。teacher相对open仍有3/8前缀变差，不能以单例宣称修复普适。详细证据和解释见[D2P_SUFFIX_FACTORIAL_PHYSICAL_RESULT_MAIN_20261028.md](D2P_SUFFIX_FACTORIAL_PHYSICAL_RESULT_MAIN_20261028.md)。

这取代“尚无新反馈优化成功”的旧快照；仍是两个known calibration roots，无独立P1或原C1/C2/C3全闭环。主线程审计/执行，五个子智能体已实际续接并陆续交付：P2新工程合同MAIN109测试与saved verify通过；P3冷启动新对照MAIN33测试及verify通过，其CUDA产物另验收。强基线与freshness还须按实际交付核实，作者报告值与本地结果不得混列。goal保持active，不训练、不降低门槛、不按结果选根。

### 2026-10-28 后续实际验收补充

P3冷启动512次主机CUDA已完成，report SHA `5bb52423c796713899a4947f9c0167ad9dfa32d1b336b7abf858d17dd45f8462`，严格consumer SHA `0403affb31fa137308502f5875a05960222c60540b9cf28b91012984310cd65c`；四策略各128次预测，候选/配额/1536模块完整验收，仍动作碰撞、modeldelta0，保留零结果。原C3同任务的程序类型/horizon决策仍未完整实验，不能以跨根路由替代。

四旧observed-prefix CEM64主机CUDA260调用也完成并验收，report SHA `83f9b6fa5ad991022af6bc38063c0fa6c72fca350578692b3c30f4a7e4854a6b`，exact consume SHA `b81abd4c2f6600bae69513228a9b6896e2a67f09c3e0ab46fc6901a4034f47d3`。物理adapter待交付，模型Q不能充当真实任务收益；该四prefix与最新moving8不是同一协议，不混比。238作者报告cells继续与本地结果分栏。

C2关键限制补核：最新moving四个D2P前缀均已改善官方联合位置，唯一成功分支首段179.73780→62.28799px。因此该1/8控制成功仍不是暂时退步后改变未来能力的机制见证。新根工作者新增1033文件metadata与原返回链，解决5个原子状态请求，但完整历史reset/加载闭包/scope仍缺，当前0认证fresh candidate；不以新seed或找不到旧记录追认freshness。

完整证据与当前并行任务见[D2P_PARALLEL_ACTUAL_VALIDATION_MAIN_20261028.md](D2P_PARALLEL_ACTUAL_VALIDATION_MAIN_20261028.md)。P1、独立根、强基线物理、P2/C3真实增量及原三实验族仍未全闭环，goal保持active。

### 2026-10-28 MAIN：CEM真实物理与P2实际派发账推进

上节“物理adapter待交付”的快照已被真实执行取代。四旧observed-prefix CEM物理完成：零工程失败、原生1/4成功，独立严格消费及native7/joint XY复算完成，四项joint XY均比历史后缀改善、平均−51.73718px。其v1 prepare容量错误（540309>524288 bytes）已保留并以最小v2工程修复真实重跑；不能将容量拒绝称理论负结果。证据见[D2P_OBSERVED_CEM_PHYSICAL_MAIN_20261028.md](D2P_OBSERVED_CEM_PHYSICAL_MAIN_20261028.md)。

另一八teacher anchor CEM主机CUDA520predict及严格持久回执完成；其主机物理也全部执行、零工程失败、原生1/8成功，全部51项prefix恢复及saved consume通过。实际report SHA `f359812367d8e901b171921552ee464cb3be399168e825bdca4c3fdde4c0a2aa`。仍是两known roots，不是独立8根；与谱teacher_full逐条配对分析待MAIN验收，不能混旧observed4分母。证据见[D2P_CEM8_PHYSICAL_RESULT_MAIN_20261028_v1.md](D2P_CEM8_PHYSICAL_RESULT_MAIN_20261028_v1.md)。成功D2P prefix本就立即改善，不能开放C2非贪心机制结论。

P2 runtime真实HOST CUDA四根八臂480候选终结，1328predict、4个192×20与160个192×10 Jacobian；source/model unchanged、无新物理。组合与无组合模型Q两胜两负，严格独立资源/数组consumer待验收，不称P2增量成立。证据见[D2P_P2_RUNTIME_CUDA_RESULT_MAIN_20261028_v1.md](D2P_P2_RUNTIME_CUDA_RESULT_MAIN_20261028_v1.md)。P3同任务runtime MAIN30tests及saved verify通过；H1/H3真实signal adapter在实现，不能以callback合同或跨根零差替代C3真实实验。

freshness MAIN核对38tests及固定request成功：16引用对应8来源10次已发现调用；0新增认证。原运行loaded/reset/scope材料仍不足，空schema通过不是历史真实性证明。不能以新seed追认freshness。P1、全资源强对照、独立泛化、原C1/C2/C3与三实验族完整科学闭环依旧未完成；goal保持active，不调科学阈值、不开失败门、不训练冻结模型。

### 2026-10-28：后续严格验收与真实 P3 动作生成

CEM8/teacher_full严格配对和MAIN独立算术均完成：各1/8原生成功、成功分支不同，平均joint XY为76.44600438/53.73472948px；两known roots、各自预算不同，不宣称等算力优势。32行基线长表与16条唯一费用记录已验收，作者论文值继续单列。详见 `D2P_CEM8_PAIRED_RESULT_MAIN_20261028_v1.md` 和 `D2P_BASELINE_COVERAGE_PAULI_20261028_v2.md`。

全十谱方向42候选v1因输出容量低估失败；未作科学负结果。最小v2仅修容量/失败落盘后实际1032predict、172J完成且strict consume通过。四项raw选择与旧轴库均不变、仍为stage-greedy；非贪心候选均不在(stage,own-reach)前沿。sum/mean归一没有发现接线错。限制仅适用此固定库；不证明连续动作空间或理论无解。详细来源见 `D2P_ALLMODE_CAPACITY_REPAIR_MAIN_20261028_v2.md`。

P2 runtime saved独立消费与MAIN验收完成，不再仅标“待consumer”；本轮复用接口及真实480条动作见证接缝再经88项联合回归，尚非独立程序库/任意深度组合或物理增量成立。

P3真实H1/H3局部信号采集、决策和严格消费已完成；本轮进一步实际HOST CUDA生成16候选、32双路径回放、64predict，严格消费与独立终态Q复算通过。report SHA `7f5f16ffe9d440a5cd2731e1e87c2325b1b39de9c18508ebdcb28409cbe858d3`；两根H3相对H1模型Q一好一坏，horizon/预测预算不同。它是真实root extension，不是等预算C3、child-local树或物理收益；详见 `D2P_P3_NATIVE_PROGRAM_RESULT_MAIN_20261028_v1.md`。

全方向与旧teacher物理的完整两consumer复用检查已实际执行：reused0/unknown4，原因是完整raw10不等，不能继承成功或失败。本轮复用回执SHA `b8cfcf8731bae5c6ad745cfc55f57cbe0c306536e07332c615b2bc1ec0795d2a`，unknown不补0。后续独立执行若产生报告必须另验收；当前不把准备或启动当成完成。

P1仍缺同根条件任务、所选真实非贪心能力改变、真实收益与独立确认全链；强对照完整资产/公平资源、P2/P3增量和原三实验族仍未闭环。五工作位继续独占并行，MAIN审计与CUDA执行，无训练、阈值放宽或神经CPU回退。goal保持active。

### 同轮全方向物理终态更新

上节四项unknown是旧teacher动作不能复用，不是物理失败标签。MAIN随后执行独立四项完整重放：errors=[]、40规划动作及40warmup、816内部physics、264显式render，完整身份/PNG/journal消费者与独立原生指标核验通过。实际report SHA `3ca4891a5f0823bede431e85442513a095b1e9bc8d49c87feefbb56663dca3b4`，持久严格回执SHA `c26b2dd15b69a1f3532a5e7fd88ad3f39acb7f84071e34101c5711673d8b4acc`。四项原生成功0/4、均满10步；联合XY终距56.84685/69.80531/56.01562/52.50351px，仍未达20px成功门槛。首段均立即改善，不是非贪心能力机制见证。详见 `D2P_ALLMODE_PHYSICAL_RESULT_MAIN_20261028_v1.md`；不将固定候选库负结果提升为理论否定，goal仍active。

## 2026-10-28 本轮并行后的实际验收

P3 common-budget主机CUDA与strict consume已完成，并持久化32候选独立Q复算；每臂32predict、384数学VJP方向，两个已知根最低Q仍一好一坏。fixed H1与adaptive H3执行长度不同，不宣称等真实时间或闭环收益。详情见 `D2P_P3_COMMON_BUDGET_RESULT_MAIN_20261028_v1.md`。

mixedmode222经MAIN60项测试后实际完成5352predict，报告SHA `d6166b98958af70cecb8529135396c77467b7c803bfe12a1a5de29c422c22ad2`；严格消费/独立复算已启动，尚未验收，不提前记P1通过。

后续终态更新：mixedmode222 strict consume与MAIN独立完整旧42复现/新222算术完成，分析SHA `0788833b58af36e49c4794be90def317688c3590c2d873f81cfc631f2226c830`。root0两个条件新选index51/raw改变，root1仍23/raw相同；四项仍stage-greedy。MAIN随后实际328次HOST CUDA中心差分，8上下文×10坐标×2epsilon，160全192维比较与独立journal/差分算术通过，最大相对L2误差0.0004381352。详见 `D2P_MIXEDMODE_AND_FD_RESULT_MAIN_20261028_v1.md`，不将局部导数数值正确等同真实模型准确或理论成立。

mixed物理adapter MAIN172项联合测试后实际prepare完成13,463,090 bytes，SHA `667a212745650520c7c852c1192d34e98960c260fafbbd021e1fabed47877049`，双完整consume认证路线为两项新执行+两项整行继承。主机物理已启动session99260，未在本段宣称终态。能力probe v1实际prepare另遇8MiB容量拒绝，无新物理；已保留错误并派发v2实现修复，不算科学负结果。

P3固定请求执行5步的新合同MAIN完整读源码/tests/doc，联合210tests通过；实际prepare SHA `428f043c731098c5f6620b29d3bf180fc6912fd69e90b29a2d6f6166454650d6`。新CUDA runner和固定5步物理consumer仍待实现验收，不能用准备合同称等真实时间对照已完成。

Hi-LeWM唯一官方整包GET终态失败：HTTP200、curl18提前断流，实际16634096 bytes/2299863924，部分文件及失败报告保留；不是403、不是已到手模型，不把资产失败记基线0成功。三臂评分合同MAIN读完源码/测试/文档并实际52tests通过，但仍缺真实三臂共同执行与完整配对资产，不能以该合同称E1闭环。

### 2026-10-28 MAIN：固定五步P3真实CUDA及完整闭包修复

mixed physical已实际终结并strict consume/独立原生算术完成：两条新执行root0 off/on相对旧全方向jointXY改善5.58731149/2.79775045px，root1整行历史继承，原生仍0/4；首段仍立即改善，不是非贪心机制。详见 `D2P_PARALLEL_DISPATCH_MAIN_20261028_v6.md`，取代上节运行中快照。

能力probe v2实际prepare约8.9MiB，SHA `e8e4554ea2448a8c4bcf4e916934800fa8c7afe3eda335174056a551620da3d5`；完整闭包确实超过旧8MiB上限，旧拒绝属于工程容量问题，不是模型/理论负结果。新40分支before/after执行器完整审读、含原RGB guard的70项联合回归通过（另2跳过），实际own prepare SHA `8226f8a77a7584c598ec3ce137926703a99029ec58160a1024c8c1b76630f394`。主机物理session38504运行，尚待终态，不提前计科学结论。

P3 fixed5实际HOST CUDA完成100predict，32候选/4赢家首包完整，errors=[]；report SHA `8067a743e3893b120ac8d57582c8552cb5b1dec182f7ca4bd0c6f70ffacaeeb5`。MAIN严格consume+全部候选/共同五步Q/稳定tie/raw5独立复算完成，持久回执SHA `a8770ed8242ed8d2f7f27eac0002d7f0bfabbd8ed5084c5d7d933f8793bc0388`。计入已付信号各臂33predict/sample及2J/384VJP；共同五步模型Q仍root0改善、root1变差，实际物理未执行，不能宣称等总资源闭环收益或完整child-local C3。详见 `D2P_FIXED5_CUDA_AND_PROBE_MAIN_20261028_v1.md`。

P2组合执行接线、P3物理adapter、P4RGB反馈输入和P5冻结复制前置已明确续派，独占分工见 `D2P_DOWNSTREAM_DISPATCH_MAIN_20261028_v7.md`。P1、独立泛化、强对照真实公平比较及原三实验族仍未完全闭环；goal保持active，不以工程产物重定义完成。

上述40分支物理随后实际完整终结：report SHA `57ac436ff6cd7fcffcda62e0a165782b719c0383404647f2f87c1fa97ab009c8`，40completed、errors=[]、700public/7160内部physics/2340render。MAIN严格physical/probe双重消费者已完成，持久回执SHA `d1efceb6aa0eb4e72a586bb0c1ba297ecb433f12e048e7a0c966450b71156537`。另由不导入producer/model/env的独立源码重算26322事件与700 lossless NPZ返回的原生奖励/终止/fullnative7 info join/资源，全部一致，回执SHA `4e90b88ffb135aeafdd63b68cda57b75f6315638bf1f6829a75822d8d07b622e`。这取代物理运行中快照；有符号任务能力分析仍待完成，前后状态/时间混杂仍在，不据此打开P1。
# 2026-10-28 后续 P 阶段执行补充（主线程）

本节终态补记：P2四根八臂真实物理与strict消费已完成（0/4 vs0/4 native），另MAIN独立80步native7/reward/stop复算通过；P3四臂各raw5真实执行完成，合同哈希与placement角度两项代码接线修复后完整strict消费已完成（未重跑封存物理）；P4三反馈8臂已完成，7臂20步、1臂11步原生成功早停，23编译/115predict、模型未变，strict及MAIN独立247PNG/151控制返回/10040事件审核通过。以上是已知根工程校准与局部实验结果，未打开P1或宣称P2/P3/P4机制成立。P5新的原生action-word接线准备时发现原manifest哈希方言用错，正在新版本修复，尚无新CUDA结果。完整路径、SHA与全部正负结果见后续执行v8；下列较早“启动/待消费”描述由本终态补记覆盖。

后续P工程并未因P1未闭合而全部暂停。最新主线程证据与并行分工见 [后续执行 v8](D2P_DOWNSTREAM_EXECUTION_MAIN_20261028_v8.md)。科学准入仍按原前置，不能把以下完成度等同整套研究成立。

- P3原实际prepare失败已由主线程定位为跨schema合同哈希方言错误：原默认空格JSON与runtime紧凑JSON混用。v2仅修原合同所属哈希函数，96项回归通过，真实prepare成功。已实际完成四臂各5步物理：errors=[]、60 public/616内部physics/204 render，报告SHA `5fdfc8cfea7f6b4fe78d3e19b59fe946aad73a537850c52a86f2d395055b5e45`。严格saved消费仍需核终态收据，不提前声称C3收益。
- 新同动作域beam对照：主机CUDA288predict已有严格模型消费；此次两根各raw10真实物理完成，0/2 native success。MAIN独立1186事件/44PNG/native7终距核验通过；报告SHA `01d5485208fe40a71a53ae20bb2920740894d6ef7c114c139a991f4707e5ebe7`。不是官方论文复现、不是等FLOPs、不是独立泛化或优于D2P。
- P4三次反馈边界新driver已完整审读并28项回归通过；冻结后MAIN02新prepare、主机CUDA+physics已实际启动，不能以启动当完成。旧MAIN01因源码准备后变化正确拒绝，未进入模型/物理，属于工程协议事件。
- P2组合执行与P5独立冻结模型前置继续由独占工作者推进，主线程审计未外包。Cube已有原生动作统计/P0的事实已更正；PointMaze JEPA原训练口径normalizer数值认证仍缺，禁止tail重拟合充作冻结统计。

## 2026-10-28 MAIN：后续P新终态与代码错误隔离

本节覆盖旧“P5尚无新CUDA”的快照。P5四终态 RGB v5 在主线程审读后实际HOST CUDA完成：
7.290963秒、4API/4sample/4frame、0 predictor/proprio/新物理，模型前后expected相等。
report SHA `9505f332fc6c1cbb670d9a77f46290f8d2822486b00d89006ea038359898bdc5`；
strict回执SHA `33f3f24735131fd8b569c1d0a87068d1b3f4e26dab31618a4afb90005cf8c927`。
MAIN另不调用producer/consumer直接重算四token的预测目标/真实目标/预测真实三组MSE通过。
zero真实特征目标MSE1.14708618而x_minus_x为1.89871782，但coverage分别0.32396671和0.37083639；
预测误差与表征任务错配须分轴，不把单known OOD根诊断当理论负结论或P5/C2收益。

本轮真实prepare暴露并修复authority顺序、Path/dict API、path→SHA闭包以及wrapper递归风险，
这些是代码/协议错误，未运行前不计科学负结果。项目解释器固定数值环境下89项联合回归通过；
系统解释器曾在旧contrasts严格算术复算失败，具体依赖差异仍待核，不调容差或改旧数组来掩盖。

P1 fixed40归因已独立verify：76个库内可比pair中12反转，4pair因缺终态unknown；
同词ZS19个可比pair无反转、1unknown，不宣称完整40/20分母率或独立根机制成立。
P2 Stage-C 8分支/116保存资产严格绑定并持久化，receipt SHA
`7297bb9b5788bf379a21dd791ee61cbe9e21c36026b625a0c24027f50284d7a4`，组合收益仍null。
P3新budget-native真实saved prepare→verify成功，SHA
`22f162fb493b6b83b30bddd66ec1b7f900f4bf36ea73c6249a5a97258002e50d`；
保留48槽，仍未授权新CUDA，下一缺口是完整lifecycle和交错fullarray消费者。
P4修正audit的SHA/schema后仍无合法freshness/history证书，不能升级construction NPZ。

完整原scope不变：独立P1机制与真实收益、强基线等资源、P2组合增量、P3计算调度优势、
合法独立根/多任务跨模型及三实验族仍未全部闭环。五实现位继续推进，MAIN负责审计/CUDA，goal保持active。
详见 [后续P验证 v12](D2P_DOWNSTREAM_NATIVE_VALIDATION_MAIN_20261028_v12.md)。

## 2026-10-28 后续P持续推进：同预测对照、预算物理交接与渲染根因

最新终态详见 [后续执行 v13](D2P_STATIC_RENDER_AND_MATCHED_PREDICTION_MAIN_20261028_v13.md)，
覆盖上文“P3尚未新CUDA/待完整lifecycle”的旧快照。P3新budget HOST CUDA与完整48槽数组消费已完成；
固定/自适应分别391/634预测，各46J/8832VJP，不能声称等实际计算。
后续strict_saved物理交接MAIN prepare/verify完成：41精确同槽同raw5复用、5付费拒绝、2未到达、
0新增物理槽；不将复用当新鲜或独立根。

P2新事前同144预测协议已完成MAIN主机CUDA与完整strict消费：4槽576预测、模型未变、errors=[]。
两个已知任务谱减beam模型Q为+0.10233790/−0.13377134，仅一好一坏模型目标；真实任务收益仍null。
新选择的物理adapter继续推进，不借旧物理标签宣称通过。正式FullD2P/组合增量/强基线全套仍缺。

P4真实异步extract/consume恢复8session的literal poll链，但10batch终态无法唯一归属，
历史认证仍不完整，未打开保留根。P5静态v3修复错误overlay来源门后，实际在yx原baseline出现3像素差，
xy精确复现；10新颜色条件未运行。原native renderer默认画碰撞点，而public snapshot不含contact cache，
MAIN独立saved差异与源码分析已完成，下一真实原生重放保留缓存，不降低逐byte门。
这些工程/协议失败与模型目标的一好一坏均不得提升成理论否定。

五个实现位继续独占推进P1全库机制分解、P2新物理接线、P3预算配对、P4作者报告协议登记、
P5真实重放渲染；MAIN亲自审计与执行。整套研究未闭环，goal保持active。

## 2026-10-28 后续P真实任务与重放更新（覆盖上文待执行快照）

详见 [物理任务与重放 v14](D2P_PHYSICAL_TASK_AND_REPLAY_MAIN_20261028_v14.md)。
P2新四分支物理已完成且MAIN完整consume/独立40次plan奖励与终止复算通过，0成功；
两已知根spectral−beam jointXY为−18.996306/−7.795297，但不能宣称P2组合机制或FullD2P胜出。
持久完整回执与算术报告SHA `6b2e0301614d163cffba3d8eb2f1e522ebf6304c86d8c589eda3fe0469bfd413`。
P3完整48槽/24配对MAIN verify通过，18可比pair的native7下降而jointXY上升，仍不同实际资源，C3未成立。
P1公共五词有限库40槽/160投影MAIN verify通过，仍只有两个已知根和观察性支持，不是能力扩张证明。
P5新原生重放不是旧contact三像素门失败：四prefix44帧全byte通过，xy frame13的RGB/state/proprio也相同，
仅coverage/reward末位差导致严格停止。实际Space.step434，未完成805，未运行10颜色/CUDA；
原shape set顺序/MultiPolygon面积数值假设正在精确验证，不当作理论负结果，也不改门补成功。

## 2026-10-29 MAIN：继续并行后续 P，补齐跨尺度断点

主线程已对封存 80 对中的全部 40 组跨尺度匹配完成独立复算，四项测试通过。
模型响应方向跨尺度 cosine 最小 0.999999653；物理 pose5 全 40 组均非零，
真实 RGB 编码小尺度有 21 组精确零、大尺度有 3 组精确零。
模型目标差分 40 组同号，真实编码目标差分有 3 组异号，不能忽略零响应分母。
产物 SHA `38ee261381dab7ba1897a70f921ba2b23f1363879627fddf9fb7808eeb1c6095`。
这尚不构成 renderer 全局根因证明、尺度部署选择或 P1 准入。
详细证据见 [跨尺度与后续执行](D2P_LOCAL_SCALE_AND_DOWNSTREAM_MAIN_20261029_v1.md)。
五个实现位继续独占推进三程序组合、完整 H3 提交、真实 RGB 高低层接线、
raster 输入归因与 archive 动作辨识；主线程承担源码审读和主机 CUDA 验收。
实现中状态不是已完成实验，P1/P2/P3 科学门槛保持不变。

本轮后续终态更新：P2新三程序H3已实际HOST CUDA完成2136predict/170J、完整strict
及MAIN独立168终态Q/840局部候选Q复算；两个known根模型Q改善，但四winner首段
模型Q均立即改善，非核心非贪心机制，且组合更多计算。P3完整H3 commitment也已
实际HOST CUDA＋physics终态：四臂public20、120公共/1216内部step、128RGB、
16encoder、完整strict回执`f7424d13342295e30fd432094913e2735b55f94155dcf3a171c313499d432721`。
每臂实际51predict/1152scalargrad，H3两臂中间付费观察不替换完整15步承诺。
jointXY均损害，物体/角度一好一坏，原生0/4，不提升为C3优势。
Hi四槽真实RGB→highCEM→FIRST→embedded低层固定面板已实际CUDA/完整strict；
仍无lowCEM/rawinverse认证/物理控制释放，不算完整官方baseline。archive动作辨识48次
CUDA与完整strict/MAIN独立算术完成，12factual full3逐位复现；P1仍未打开。
详细新产物SHA、所有正负分列及执行范围均见[本轮终态](D2P_LOCAL_SCALE_AND_DOWNSTREAM_MAIN_20261029_v1.md)。
P2 full15物理adapter已MAIN37测试通过，四臂完整15步实际终态、完整strict及MAIN独立60次原生奖励/终止/raw位串复算均通过。
report SHA `49b682d0ca326590c88ef69d74eecfdd9d5c19f21667e790a7c1458aac98e064`；
strict SHA `95c967c95c8ff00e0c2d4ab5c801afbab708095479c96986319dc103933208ac`。
两已知根组合−joint的jointXY变化为+16.094521/−22.658762，物体位置+7.421614/−12.756588，
角度一好一坏，成功0/4；更多计算、无freshness，不开放P2机制门。
八模块联合测试258通过/7失败均涉及跨模块冷导入状态，独立进程隔离回归正在执行；不记成全绿。
并行位新增P2等资源协议/P3三策略完整承诺对照，真实RGB与Hi低层CEM及全场景callback继续实现。

终态补记：MAIN02独立进程回归八模块265项全通过，封存测试SHA不变；同进程冷导入断言
的7失败仍保留，不伪装联合全绿。Hi低层CEM四槽真实HOST CUDA及完整strict已完成，
report`68d23f06b88114d3f3a3df227cefe7ca247cb399ba452defbf5456dbda9f32d0`，
receipt`96be2b84aadabc4ca23a8e84f2f54f889b86367d41d4d344e0d7d56b46e3b43e`。
归一化低层优化已从固定候选升级实际CEM，但仍缺官方inverse-normalizer认证，raw_actions=null、
physical_release=false；不得写成完整官方基线或物理收益。P2的12槽RGB在MAIN生产prepare中。

终态补记2：P2的12槽真实RGB主机CUDA完成、完整strict及MAIN独立NPY/物理算术通过，
strict SHA `26b65ef96371d8e09cd5f5601225aff901e87234c6f240ae6f817bc5534210ab`。
root0在cp5/15模型→编码排序反转，cp5另有编码→任务反向，root1cp5精确平局保留，
cp15编码/XY改善但角度损害。科学准入仍关闭；不归为唯一代码因或理论不可能。
新callback捕获器实际运行时MAIN确认日志重复newline的工程错误，已中断自建作业并保留partial，
旧v1不改、v2正在补真实writer→reader回归；889public/9070physics不记完整164重放成功。
P3三策略adapter/prepare已通过，尚缺六臂生产driver/strict；后续实现继续并行。

终态补记3（2026-10-29）：native callback v2已完成全部164分支并通过完整strict，
receipt SHA `49c9a4d38fa8d271bfdd6924f56d9be97683c57fa213437d59310edbc3abdc9f`。
11152捕获中固定80对的callback均不同，24对在整数图元前差异消失且512/224相同，
其余56对没有观测层差异消失；h=.001为21/40，h=.01为3/40。
仅认证新重放，不认证历史callback membership或P1唯一根因。双LF修复经真实文件回归与
完整生产消费验证，旧partial故障证据保留。P3六臂v2真实prepare已封存，主机CUDA/物理
正在执行；P2共同wall/resource上限prepare完成，runtime独占实现中。P4历史来源身份恢复
及P5同面板跨模型adapter继续并行；这些不是P1/P2/P3科学通过。

终态补记4：P3六臂v2主机CUDA/物理和自己的完整strict均已完成，receipt SHA
`aa5c34faabae5205dbeb7dd216fa5ff0631d16239eae25de06bf2b9a2084e94a`。
六臂均20步、0/6成功。两根adaptive皆选H3/4候选，fixedH3为H3/8候选，终态相同；
每根预测51 vs75，各1152 VJP；是已见根的调用量/控制结果诊断，不是等总算力收益，
更不打开P1/C3。P2共同上限runtime与P4来源身份/P5输入桥继续实际实现。

终态补记5（2026-10-29）：P2共同上限主机CUDA四臂及完整strict已完成，receipt SHA
`7b3eba575a9cf39a565c1956a26a937828509b313be7e4aecb9f95e7f0b67ac1`。
两个known roots均composition模型Q更低、1062 vs1902预测、各85J；共同cap不等于equal FLOPs。
封存动作原生物理及完整strict也完成，receipt SHA
`47ee2abeb1ee5805c47a204efc7afca07e5f98cfca8144591bfd59b995dc376a`；321PNG、99public、1006physics。
root1 joint第14步真实success早停，其余3槽15步未成功；早停不padding、不删分母。
MAIN独立逐步算术receipt `26af5d8e074cba61f01c3a522a253b4905b69b888f9deba42c06d37f20d57738`：
root0 step15、root1共同step14组合的jointXY/objectXY均更差，不能称物理优势或P2机制成立。

终态补记6：P5固定164 JEPA factual H1主机CUDA及完整strict完成，receipt SHA
`4f6bbed041f077818627e68157b0cfab99ddba2e38100827db10f46c4aa1b759`。
164完整预测先seal后解标签，656次实际top-module dispatch、模型不变，无训练/CPU神经fallback。
MAIN纠正此前时距误判：原actual-cp5 FD164也是H1，不是root-H2；JEPA额外proprio权限仍不允许
声称完全同输入公平比较或跨模型feature MSE直接排名。该工程/条件预测诊断不是P1/P5科学通过。
P4固定十次dispatch有界附件定位已完成，指定引用内未取得原attempt/state/loaded manifest；
不反复新增空wrapper、不把有限范围扩大成无限全历史义务。细证见当日LOCAL_SCALE文档及innerdocs附件结案。

终态补记7：P5独立open-loop H2主机CUDA及完整strict已完成，receipt SHA
`783b65f67899791b5697bfe0819ccfb3231d57212b9e3e85aefc02419459c0ea`。
328预测先封存后解标签，第二次输入严格来自第一次预测，不反馈真实15；模型不变。
H1/H2全部80对与native新重放精确身份连接也完成：24个真实特征零响应与24个量化相同
配对是同一集合，不只是数量相同；56个非零对仍存在方向失配，尚未归因完毕。
该结论只限新重放，不认证原历史callback membership。

后续执行补记：P2共同上限真实RGB编码已完成11/12可用槽，errors=[]、模型未变，
report SHA `282dcfa0e2ecfd0f46809bf1c21d7ba7db8586e9b05702d1801e9cf65237e498`；
root1 joint第15步因native success14合法缺失，严格消费与独立算术在进行，不能padding。
完整历史/joint30有界GRASP适配已86项回归并生成实际prepare，主机CUDA运行中，
尚无终态；每根2706 predict/900 backward不同于旧1902上限，不称等预算或官方复现。
P2物理接入、共同预算规格、P3有效调度与P5非零因素正在独占文件并行推进。
这些工程进展不改P1未准入、不等于后续科学闭环。

终态补记8：P2共同cap RGB完整strict已完成，receipt SHA
`ec10dcf054929ad50d45dc9b47a831b900563d59810eeeba8c8043cf9b737a60`；
MAIN独立NPY数值复算 SHA `f70988be555998dd44f216f38c1967ee4d1cf0b641f3cfe6acc0c06fc1ba02d9`。
12模型槽、11真实编码、6固定配对中5个真实可比；root0终点模型→实际编码反转，
root1终点缺失不替代。已知两根事后诊断不冒称泛化/P2机制。
GRASP适配真实CUDA完成5412 predict/1800 backward、模型不变，生产report SHA
`d92e24d9807a20f296811d4321b64cc87f2c8c73c04677a350f136debc1ce14a`。
strict发现空Adam状态`parameter*0`负零与consumer全正零的代码差异；实际首个virtual快照
185负零且0非零。冻结生产及结果保留，精确新consumer修复在进行，不作为科学负结果。

终态补记9：上述GRASP strict工程断点已定向修复，除空state负零外还修复参考中的
beta补数和CUDA RN-FMA相消复算；原容差未扩、原run未改。MAIN102项回归及完整strict通过，
receipt SHA `55c192a908835e6f8c05160c92c4d5a2d46a0ed81ce4985dbf1de3c3c2f4fe2e`。
两条封存动作的真实物理及完整strict也完成，receipt SHA
`17373c3d9878d2bfab9eebfa6ac16c11e8ebf2736d8b77c4d206b5a45cf4957e`。
两根都15步，root0成功、root1未成功，全部162PNG和30response验收；MAIN独立算术确认
成功不依赖负角度bug。这是已见根强基线适配证据，不是等预算/官方复现或本方法胜出。
P5非零因素全量分析及整份重算亦完成，80扰动前时序负控全部一致；固定56非零对仍有
19对H1/H2均反向，需后续固定全分母视觉/proprio输入干预，不能简单归为统一offset。
P3完整运行接入、三臂共同预算和GRASP RGB companion仍在并行，不改变P1未准入。

终态补记10：P2六臂共同资源box的主机CUDA与完整strict已完成，receipt SHA
`d3ee465f2fa6d31345f369f9514165c4817c51a29c3d00ffdd2d681e5829cc1f`。
各臂natural_completion，有完整raw15导出；GRASP模型Q低于两谱臂，但计算轴仍不相同，
不称等FLOPs或真实控制获胜。其六分支原生物理执行已取得实际prepare并运行中。
GRASP750适配六槽真实RGB与完整strict也完成，receipt SHA
`f93b9082e3b313eaefc4fe56ed0f0390c6e211a44514768dbf56e13d9f3ab22c`；
无producer数学导入的六槽独立MSE/native7复算及整份verify通过，summary SHA
`f038c1575df13bebe0727735d37bbe0459d4c88970f03e3ffed9a82a8a22f650`。
P5视觉×proprio完整历史2×2主机CUDA完成328新预测，全部seal后解共同label；
完整strict SHA `5d540a3bc6694501b74d97a52ded7aefd12ee2a70d9a46d75b1ddbb89325c456`。
独立全164/80数值及分层summary、整份verify均通过，SHA
`dd1bdbabaa3d40f91f3afce992be9a64b3f7892ab077d5e0701792f3b58a26b0`。
固定56非零方向的反向数AA/AP/PA/PP为25/25/21/22；只换视觉history有影响但不能消除，
不认证唯一代码根因、物理可达counterfactual或P1理论负结果。所有24零对和4baseline保留。
P3八臂/24节点完整运行链已MAIN审读，独立46项通过、实际prepare封存，
主机CUDA及真实闭环正在进行；未出终态的阶段不由其他分支替代签证。

终态补记11：P2六分支共同资源真实执行及完整strict已通过，receipt SHA
`ae5a62dbc158ac1a7da217a413987f5a88bafde357326559c2c384c016ada87c`。
149 public/1514 physics/483 PNG；root1 joint native success于14步提前停，故其cp15缺失，
其余五分支满15步未成功。继续完整18槽/三方法同时间模型→真实编码→物理核验，不补缺帧。
P3八臂/24节点实际515.4562秒且errors=[]，完整strict SHA
`a48d63d272a4da1b4a270fa3dee60008d1e5d59251146b5ddd8143ba14632b5f`。
独立逐步/资源/候选支持summary及整份verify均通过，SHA
`3235490cc586f1521197accfbcfcd12f583194609e91e45314b3f40c31c5b446`。
四个同bank反馈对照终点jointXY均下降，但root0物体XY反而上升，八臂均未成功；
不同资源轴且仅两known根，不认证C3/等计算收益/P3科学成立。
P5 latest-slot后续实际HOST CUDA完成492predict/1476frame，锁定actual5/10的字节，
只交叉15-slot，0encoder/physics，模型未变、工程errors=[]。完整strict SHA
`c636f8d9c6624317d4da4f3ddb127b4f20710f9a5c02c212872d0d70c9cab642`。
其原固定56非零对AA/AP/PA/PP仍25/25/21/22反向，其余24对null保留；
与整history相同的统计须进一步做全部数组/接线分析，不据此挑根、宣布唯一代码原因或修复。
P1科学门未打开；P4仍缺原始身份资产，无新资产时不重复扫描。

终态补记12：P5最新槽独立164/80 summary及完整verify已通过，SHA
`ff5958ef8a23ebd6a76b4748659e43008ebd932b8f4c39dbb817bb766b1b728f`。
随后跨whole/latest的完整原生IO比较及整份verify也通过，SHA
`bc7665c448a89fee4b89bd3bc150cd2016d7772035cb0a0588bd8014b47003e0`。
所有164分支前两槽visual/proprio原本相同，因此四cell×164完整输入与两类输出相同：
“整history”在此来源上实际也是latest-only，新重跑不构成独立contrast，不能扩大证据量。
绑定实际模型mask允许末槽关注前两槽，统计相同不能推断架构屏蔽。
P2六分支全部native标量独立复算及verify通过，SHA
`052b1fe3b7d6000e1c8971a277414e1e183ab7b419a15e0abb044526de162959`。
其17张实际RGB已完成HOST CUDA编码，完整strict正在接续，未借其他收据替代。
P3资源精化：八臂完整路径各4encode/6J/51predict，compile/solve/SVD/replay轴仍不同。
原包装器恢复及node时钟同基准已通过6项非神经fixture；目录扫描与fsync开销真实存在，
但未定量归因生产增长，不据此认证速度优势或宣布计时bug。P1科学准入仍未满足。

终态补记13：P2共同资源17张真实RGB完整strict及无producer-math的18槽/18pair独立
标量/误差项/物理各轴分析与整份verify均已成功。strict SHA
`ec38708895cefd85e9178936826a0903b427050849c97246674879e19f4a45c4`；
独立summary SHA `b3c1493d92816e0956f6e952a885505147879fdd243528d10f6098a82608b353`。
16可比对：model→真实编码13一致/3反转；真实编码→jointXY同为13/3但反转集合不同。
早停缺失两pair保持null，没有补终点；这些不是16独立根，不能认证泛化、统一偏移或唯一根因。
跨whole/latest全部数组的整份verify亦已成功，两轮实际相同输入的协议纠正保持有效。
本轮下游工程执行/严格认证已收尾；P1/P2/P3/P5科学准入仍未满足，研究总目标未完成。

并行推进补记14（2026-10-29）：MAIN新P3 shared-panel执行器与独立saved分析器
联合104项回归通过，生产prepare及完整verify完成；真实主机CUDA/物理八臂正在执行，
尚无终态严格认证，不能把启动或fixture当科学完成。每节点完整支付H1/H3两个面板，
仅路由原policy horizon字段；原width/depth/type未完整执行，unsupported保留拒绝，
不fallback、不认证完整C3/自然廉价fixed基线/等FLOPs。
P2六槽subgoal准备及verify已保存，SHA
`624c369764ba75776c9649de67a84f99a4004490808b626c68def405c512b788`；
原Hi权重/实现已存在，但prior到P2坐标/normalizer及三臂controller尚未接齐，
属于实现/来源前置，不是理论否定。直接latent与normalized Hi transition最小核心继续并行。
P5实际解析164份actual0原生IO指针并核branch0六份原NPY/归一化接线，catalog SHA
`1c49b5eba91619a6798a9d387c524b6e7d8658c196a9ac75f1ea9a65226c2678`。
这没有执行新神经模型；固定branch0 T1→T2 CUDA探针仍待代码审读/生产，
不以旧full3早期输出代替causal rollout，不将相同输入重跑扩大独立证据量。
当前运行证据与准确进度以D2P_LOCAL_SCALE_AND_DOWNSTREAM_MAIN_20261029_v1.md为入口。

终态补记15：新P3 shared-panel主机CUDA/真实物理已完成，八臂各满20步/132预测，
总1056预测、0拒绝/errors=[]。完整strict SHA
`d23ffa48bc954306bc27a15f924da8daafffc9b4bc76a91e59aa949ab1abbafd`；
独立summary+完整verify SHA
`4e89812a384a352840f41b84ef9a0ac1ea07622288949c6d5ff6c5795cc28224`；
激活/分叉summary+完整verify SHA
`c2e933932b2b6a924f565295b72e676849470c7ac21f98dbb21a1086acf9e2ab`。
fixed12节点全H1、adaptive12全H3，全部八配对第6动作/状态真实分叉；不是未激活，
也没有实现轨迹内horizon切换。四个同bank adaptive−fixed终点jointXY都更差，
root0 latent-Q却更好；完整不同物理轴保留。仅known2共享付费路由的不利证据，
不认证C3/理论一般失败或唯一代码根因，八臂原生success均false。
P5 branch0真实T1→T2 HOSTCUDA及保存因果接线认证已通过，strict SHA
`1e376b1f06ca3b04ad71822b76178308dc4ad3d700641c97726b1c7f94c5d346`。
两predict/三token-frame、无模型变化/encoder/label/physics；只证明该输入短启动可执行，
不是164全滚动、预测准确或P5科学机制成立。
Hi normalized transition源码与whole/segmented接口已MAIN审读、39项RAM测试和源核验通过，
仍待真实CUDA数值/导数验证。P1科学准入与研究总闭环均未完成。

终态补记16（2026-10-29）：MAIN完成动作见证48槽逐raw字节对齐，结果SHA
`7b0031e127848ccacc474699ae8e98d7aaa9b3cb633f00d740b1ee201d4d5911`。
只有12个fixedH1同动作终点；全部24个H3槽无完整同动作见证，四条adaptive b5在control11变动作。
不同动作轨迹的终点差异不得当作纯H3模型误差。固定四分支完整15动作物理见证另行实现，未提前签通过。

Hi真实CUDA probe v1失败report SHA
`44ce2355c454f690016ad73d1565659881da796c354c1799044c7ed11512c02d`。
MAIN从逐模块requires_grad追踪到旧可信loader内部inference_mode截断动作图；3个low前向已付、
首次VJP失败、其余route未启动。此为新可微接缝误用inference-only合同的代码问题，不是理论负结果，
也不反推历史P1均由此造成。旧CEM/loader/模型/结果冻结，新facade与probe v2修复接线后另行验证。

P5真正全164 causal rollout已完成HOSTCUDA及完整saved consume：report SHA
`1b014fc5c2e9868ee5e08e0dc05b602b5e9708a342c9c842698183f3430297c4`，strict SHA
`3a20eb780028ddd40a88d11c59b8ada9dcbc0c7147bf1df9aa9acde332aa26fd`。
656predict/1476token-frame全attempted=returned=validated，38.0707秒、errors/source_errors=[]；
164共同warmup输入/输出逐字节检查均一致，所有预测先封存后才读取既有H2共同标签。
这是新增真实因果滚动及完整协议证据，不是历史改善、同预算算法优胜或P5科学准入。
共同标签旧路径重评分分析并行实现中。P1和总体闭环仍未完成。

终态补记17（2026-10-29）：上项P5共同标签比较已实际analyze及完整verify，
SHA `bbafa5ec58dde0f0a12d01aded1f2d49a6a869bfe1c73e0d638eb12a6c57ff24`。
short20−H2_20 meanMSE=+0.018059734；不称总体改善，不选唯一有利stratum。
保留H1/H2 goal-bind时间与标签batch接缝及80对null响应，详见局部尺度/后续P主记录。

父节点真实CUDA树report SHA `75f4c162aaa75223dfd8523381e7614c8a35637afd52278df8a80d6affa34e40`，
完整consume SHA `622d1cdaf8d1ed69c4e78f8d1f5c7c6114285db6121f0021fbbaf6573a8a76ad`；
新增saved汇总与整个verify SHA `00ebd93b13353dfc54d45263575bf0d16a642d8985e773281aea265b94d81467`。
各原144分母保留；root0 adaptive在child R8停下不发布plan，属于尚未支持的生成器工程前置，
不是理论否定。root1 adaptive与fixed计算量不同，尚无等预算/真实horizon切换C3证明。

四条原b5 H3动作完整物理执行已完成，report SHA
`c8fa06be8eaaf6f7515e6e9550c900c71e04f2141d745699b33d1585eb21a2ce`，完整consume SHA
`48a1a5ecadf7043e3162ca104d8f43b0d59baf88655a076f675c9f96d697b9f7`。
新commit-vs-MPC native指标复算/完整verify SHA
`5f1234f9fcf2c217e66d193fff287f1fe545045daf68afe67c01a574283d16c6`：四主对照cp20 native7均改善，
但jointXY两改善两损害、全部success=false；不混为P1/P3成功。真正同动作RGB编码尚待新实验。

Hi可微接缝已另版修复inference-only误用与IR tuple/list ABI，真实CUDA probe v3 SHA
`ff4d92b73d5f11bb40953acb0b6e90c60bd97e6c991894066b51ab716e1e60c1`，
三路前向/30动作梯度逐位相同及固定FD通过、原425项state未变。
新增saved consumer SHA `ce69411a511820a0b3b11b05cb5bb0959e54ae65a0edc9e50cd0e790d721c0c9`。
明确披露原manifest缺失两个nonexecuted loader test/doc metadata的before/after封存，
仅认证完整数值/导数与已有执行source，不能追补声称整个metadata原合同已通过。
Hi/P2权重逐key配对、原scaler provenance及跨实现计算仍未闭合。P1与总体科学闭环未完成。

终态补记18（2026-10-29）：H3同动作RGB编码已真正完成HOST CUDA与完整consume，
report SHA `c3195ea6ba35898bfb0eaef8f9dd69e7e2f4e01834c7a2a43c6c73302352c539`，
strict SHA `671e48b0e953e6adeb8e07381ecaa965572afa8801d29d961c344eb82a3e505e`。
16encoder/16projector全返回、0新predict/AD/physics，模型未变、errors=[]，四cp5锚点逐位复现。
现在有四固定H3完整动作轨迹的cp10/15/20真实编码，不再拿MPC换动作后终点替代。
十二同动作Q误差正负混合，root0 endpoints在cp15/20为负，其他为正；
向量误差与目标方向交叉项可相消，不支持统一常数偏移归因。三时刻非独立根，
不能由此证明P1机制或P3等预算收益。MAIN独立只读解全部16保存数组并复算四anchor/十二Q恒等式。
下一分解器及P2目录/P3共同cap与R evaluator/P5四格历史干预实现并行推进，尚未冒称生产完成。

终态补记19（2026-10-29）：上项saved误差分解器已实际完成，33项回归通过，
产物SHA `45bea5540cec65e0b2afc5cb4756e3388d572ecd6752a17312aa628d3bfcd2e2`；
4anchor/12forecast/12temporal全量，继承完整encoder receipt，不重复上游strict。
Hi/P2目录导出已MAIN HOSTCUDA实际执行，41项回归通过，report SHA
`5605d9d7966a38ac146a84f987a09099dae33da5a2864110a69da06fd1970650`；
完整consume SHA `9a5406bd076ccfd8a931d41943a64a1f905fa07a6a42711d419db923bcd7abd7`，
五组join SHA `1523183330336bd3e66884a15fa96e7a15161abd8a343d1f50dba7a0427494b8`。
五组逐key角色/shape/dtype/persistence/None/alias/原值SHA全部一致、布局差异0；
runtime333含显式None，不是333非None参数。两遍目录和原aggregate hash稳定，0forward/训练。
这消除了“现有两个checkpoint的五组tensor目录未知”前置，尚未验证跨实现RGB/history3函数、
完整config及原scaler来源；不把相等目录推成P2三臂部署或P1科学通过。计算parity另线实现中。

终态补记20（2026-10-29）：P5真实早期视觉×proprio四格干预已完成HOSTCUDA和完整saved消费。
prepare SHA `c8376a96a8320d4f147bf5e526954fbe8e31d37db565428ad82916dcc9d0f9c0`；
report SHA `4fc86a7fdc69fbe95003e417ee38e3a4d430360a8eb5bd0c89f73ee845926d0f`；
完整consume SHA `d9461f7a21fa13454eca1f4371488431cf833a08ee758990c2dcb999b342600e`。
656新predict/1968tokenframe、43.00644秒、errors/source_errors=[]；164个RR全nativeIO逐位复现旧short。
所有预测先封存后读取标签；全164/80及四层保留。AA/AR/RA/RR meanMSE分别
0.5695298113/0.5697998422/0.5645169505/0.5645417704，视觉效应−0.0051354663、
proprio效应+0.0001474254、交互−0.0002452110。joint四层正负混合，不选择最优cell/层。
这证明固定latest/action下早期history有真实模型内部影响，排除“早期槽完全无效”的简单解释，
不证明物理因果、单帧根因或控制收益。R8 RAM评估核另有MAIN69项测试通过，尚未接生产。
P1与完整P3/C3科学闭环仍未完成。

终态补记21（2026-10-29）：P2跨runtime计算与P3等上限已取得新真实证据。
Hi/P2 computation parity MAIN35项回归后HOST CUDA19.87878秒，完整786数组；report SHA
`12fe43cc4a796fffc18b98fcc867283f8534c22110d794729652ac112aedead8`，完整consume SHA
`3555696c02329062084d791c0c40958084d7352e6f7680938bade0068afe9480`。
known2各104个RGB/history3/三槽/30动作梯度比较均bitwise一致；AE累计history的布局差异
如实另列，未通过coerce掩盖。原scaler/raw部署/三臂统一controller仍缺，不据此开放P2或P1。
P3同27轴上限HOST CUDA235.81499秒，report SHA
`8c65d3c5c54ba447b279fd0ad938edfab6eb42f7e1b3dbb4a1624987153cc58d`，无运行错误；
四槽predict/J为114/18、28/4、114/18、278/26，原R unsupported仍关闭该槽plan，非R执行负结果。
原reader因79913556byte单行超过继承64MiB拒绝；MAIN不改封存源或数据，新建bounded reader，
11项测试后完整原数学/资源/计时/来源核验成功，receipt SHA
`2ca340e108cd0c82fea6f9378a95b47f51aad6c49599f9f6cf113d8b32b3cb82`。
该修复保留原1GiB总输出cap/256MiB reserve、全576分母和完整hashchain，无CUDA重跑。
P5独立不导入producer标量分析及整份verify也已完成，SHA
`ba881dc05522683fe44491a04e028c27dd66e20ae1bb4cd65be9a8cb22ee212e`，28项回归通过。
原生R公共接缝72项MAIN回归通过，生产bank生成器按新事前spectral8变体合同继续；
下一P5 temporal两角+RR事前固定492新predict，尚未执行。不得将工程准备写成科学通过。

终态补记22（2026-10-29，更新补记21的temporal待运行）：MAIN完成60项回归及真实HOST CUDA，
P5固定temporal新两角+RR bridge全492predict/1476帧，40.31847秒，errors/source_errors=[]。
report SHA `f046c617ba2eeb6fcbd79799238e8ccbd00ca7ecd6b5414f2021844550672747`；
完整consume SHA `86149cf64a0e5c423ee99f23f98dde9ce06ccf8ced837f70c65648e3518e51ea`。
slot5/slot10平均MSE效应分别+0.0004319723/−0.0054200132，但四层方向不一致；
全164/80与所有四角/零响应保留，不挑角、不称控制修复或独立确认。
P3同动作前置resolver48项MAIN回归通过，真实输入另捕获132MiB原report超过128MiB通用cap；
MAIN独占compat reader恢复仅该角色的原256MiB reserve（4项回归通过），保留旧source/结果。
actual prepare SHA `d51467dabc0fb2625016ff21229f33b04fde93b966e4a5c8e77408d9cef675c8`，
八槽旧R完整三边IO与A真实history接缝全部ready；固定8次新A预测runner仍待实现/执行。
这两类reader ABI失败按代码问题修复，不当作模型负结果；P1与研究闭环继续未完成。

终态补记23（2026-10-29）：P3 fixed8同动作actual-history诊断已完成，不再只是ready。
最终40项MAIN回归通过；首次封存重建因交接窗口tests/doc变化被预测前拦截，保留旧prepare，
冻结后新MAIN02 SHA `3ade105397985c56cb878a9677a79170ef80b67e7286671f4021ae6277a4146f`。
HOST CUDA 11.75593秒、8新predict/24tokens、0新R/encoder/AD/physics，errors=[]；report SHA
`cf038db843d2daa51b426e12aac46558d907d65762147ef465d8a5d82381b251`。
完整own consume SHA `0cea92f6ff26d5075e679a835a91e564d75830145e1cd109173a6d183f04d3a9`。
全8槽向量MSE由滚动history平均0.1810596635降至真实history0.0244807613，但目标Q绝对误差
仅5/8改善、3/8恶化，不能推为排序修复或物理受益；known2仍非独立确认，P1/P3/C3不开放。
P3 equal-cap comparison v2完成MAIN42tests及真实analyze/verify，SHA
`23746d89f38d668cefc5dafcc01601721a416a65c496b4bbc0a9c16e51a77d20`；全4臂27轴actual计数
新旧差0，三个released plans动作逐位一致，另一臂仍8个unsupported R请求，不能冒称R执行负结果。
P5 temporal独立算术工具MAIN39tests与真实analyze/整份verify完成，SHA
`9552571dc77ff819d54143dcdf402b39c08e684945325d3448ee39c3d01768cb`，全164×4/80标量相等。
R spectral8 bank另59项MAIN回归通过；R-aware HOST orchestrator、history3统一tracker真实接通、
commit5物理前置与fresh-cohort准备继续独占文件并行，不将工程交付记为科学门通过。

终态补记24（2026-10-29）：MAIN对同动作history八槽的四个同root/time排序对进行独立标量分解，
12项回归及实际analyze/verify通过，产物SHA
`1c7bafae7d57bb146f40128f96ded293009644a70a97eb03faa03b3615955a62`。
真实history四对均与真实编码Q同号，滚动history仅2/4；只说明known2描述性排序，不证明泛化/物理收益。
R运行拒绝另定位静态协议冲突：旧tree replay cap2与完整R组18不相容；保留旧合同，
新增独立四组诊断协议，不把未执行R视为模型负结果，生产仍待完成。
actual-cp5 builder MAIN31tests及真实prepare/verify完成，SHA
`ce6670d84f6f0f712bdfe5e1b7471be65b0441338c18010f79b3967ef28397c7`；两根八分支真实5/10/15历史
明确区别原0/5/10。P2共同history3 tracker MAIN49tests后HOST CUDA28.91847秒，24API/108samples，
report SHA `ab89ea3c8c7198ce001280a93c73c312c5454f552b4452232e04a6776adf6884`，完整consume SHA
`080fe1173895621ea4c0565671509a267ba456135fedfac313ddac7894f98138`；errors=[]，两根returned mean
whole/segmented付费重放完整，模型425项不变。该小预算normalized smoke不是正式强baseline或三proposal比较。
P5 fresh前置MAIN18tests及实际resolve/verify，SHA
`d453e4cf70518839844916566fa274959c515d05310ae090fb5699e589cf3133`，认证fresh=0、P5N=null，
仍缺历史覆盖和事前新cohort协议，不将旧seed槽位重复宣称fresh。P1与整套闭环继续未完成。

终态补记25（2026-10-29）：P3原control0完整R四组HOST CUDA124.39967秒，196predict/40J/72replay，
report SHA `2c9218f6dfbcf4d24f9a93c9f0ef9dfad2a97f7496adce83bbe0e73cbe8dc354`，完整consume SHA
`c7cedef24438bc6ca95ff2b5c6ec813feb9caee5edc8b47346394d66d8be521a`；全部zero+8真执行、errors=[]，
四组no_witness，不再是预算派发拒绝。MAIN独立14tests及标量analyze/verify，SHA
`e1fa1a527e0d5a025cfb3ba5f6fac35900d1d68467e6127a039e9ffe908a113c`；32候选无资格交集，
reach分解为immediate+nominal_drift−bounded_linear_gain，不把任务残差变化冒称纯能力变化。
该known2固定bank不等于所有候选无解、物理R负结果或理论不成立。新actual-cp5模型交付/物理守卫另立事前合同。
LeWM官方CEM300×30×known2完成HOST CUDA11.27327秒，report SHA
`ab3313d24b6efb3539bb1b1a70417f7aa41df1602157465d1da5cd1d26e8b965`，完整consume SHA
`85af90a95df1481f1bb2491f1db0adfd307be50d0d7a1f636d3e4b0a712e5ada`。normalized动作域不等同有界基线，
returned mean尚未评估，不借elite平均填报、不冒称同协议公平排名。
P2四个target的真实配对及整份verify完成，SHA
`82b36fa3030f8ebb2da37bdc91ba3e8d582f129d6bc9e5c9f8e774b363ef3f62`；固定六槽保留arbitrary两missing，
prior时距unknown、无witness privilege、不称完整三臂比较。physical cp5 authority桥实际输出SHA
`9c47743431166ddb2564bf5dea5d5335e794d98ec0873a8b90417e632eefc97d`仍BLOCKED，不能从saved连接签新物理执行。

终态补记26（2026-10-29）：P2保存程序链接器MAIN33tests及实际全库analyze/verify完成，SHA
`82f73b810e5cd34b12907f11d0d063914578263ae4e41163e9581ac3ce3f5567`；168原链/336程序capsule齐全，
不等于新组合收益。P3 actual-cp5 MAIN37tests及实际HOST CUDA129.26231秒、完整consume完成，
report SHA `deb69d78579b4cdc0ffb3ae4c671dd71f9ecd9b9e02729484085cb3d0c0dd19d`，receipt SHA
`8d55a55e108aca1695a550861d125a0b0baac2b225cba914d0af4b47d30159a5`；196predict/40J/72replay，
四槽全no_witness、errors=[]，不造fallback或执行物理，不据固定bank否定理论。
P2四目标controller MAIN49tests及HOST CUDA51.31324秒完成，48API/216samples，report SHA
`b1358eb3cec42cdddb30b0d138c2bc2e68773f195aa985977bf9c090527382cb`，完整consume SHA
`34e0491ca05a1b381db3541997c6215c44f9927761d6123a559e604160f0be4a`；4returned/2missing，
无raw inverse/physics、prior时距unknown，不称公平三臂结果。官方CEM实际返回mean的新独立评分
MAIN联合61tests及HOST CUDA7.72938秒完成，8predict/8proj/6AE/20tokens，report SHA
`e7332cc12f75a8081405bfeed3da9db038ecab00d9234ebbb24f387c182dbebc`，完整consume SHA
`622754ccbd87a860598040d65af35572bb0acab0ee7be435e3a2604cf6564ee3`；两根whole/segmented未来全元素
实际bitexact，mean192分别0.1560053343/0.0620638260，不拿elite均值替代、不作跨动作域排名。
FIRST5物理driver MAIN44项RAM回归通过，但实际prepare暴露`C3`/`full_C3_demonstrated`字段接线错误，
未生成prepare/未派发物理；保留v1、另修v2及真实ABI回归。该事件为工程失败，不归理论负结果。
本轮R无witness，修复也不授权造fallback；P1及后续科学门仍未开启。

本补记续：actual-cp5保存拒绝分解MAIN联合62tests、实际analyze/整份verify成功，SHA
`e12280f17807fc0a50d0800a878ba188b8d9382c1af55d8df5839401fd0d00ff`；复用MAIN数学，36项完整。
20个nongreedy中6个线性gain提升仍不足抵偿残差与漂移，32候选无资格交集，不推全空间无解。
physical v2真实13flags接缝修复MAIN117tests/73.47秒及实际prepare/整份verify通过，新prepare SHA
`1da9a251fa7d535c0236a08a707b7e9d2bb735c4ae57eeda6d946aae05d90385`；原v1错误和封存均保留，
四槽仍null/no_witness、无env/physics。代码修复闭合不等于科学门打开。

终态补记27（2026-10-29）：P2 delivered-mean raw-domain MAIN35tests、真实prepare/整份verify完成，
SHA `9df31325d7d2bf149b60db0f6d21ce40e6284060c4bb3af29c1699e1ee056c56`；固定4槽均域内，
roundtrip分别4/8/6/10元素非bitexact，另2槽missing。原mean成本不能证明实际部署tokens成本；
下一HOST实际token replay仅授权实现，尚无新模型/物理结果。官方CEM同类工具MAIN联合94tests、
实际analyze/verify完成，SHA `a1c7abc70fab0cffcb2ab23831b1c9a7a1d0b07a237df02adff4fbcd2924652a`；
root0原raw第5步Y=-1.3065606356越界，root1域内但需token replay，完整两根保留，不clip、不挑根。
只沿P2工程normalizer，不认证Hi原scaler；以上部署边界不能算理论负结果或公平baseline排名。

终态补记28（2026-10-29）：P2实际部署token replay完成MAIN全文审读、42tests/2.40秒、
实际prepare/verify及HOST CUDA26.84481871秒，report SHA
`6f1f153863d7e30b8297a03b7b10c5d5297d16cf0c5f2cefdf301b3e78772719`；完整consume SHA
`a8fe4dd9c8c90ed14a5bbecdcb226ec4d7626c786fe4db1223194f7e3f6d87b1`。
四returned/零blocked/两missing，errors=[]；CUDA inverse raw字节身份、whole/segmented完整路径均exact，
实际tokens相对旧mean仍4/8/6/10元素不同，但本批CUDA和NumPy roundtrip一致。新增24predict、
24AE/24lowproj/96AEtokens，425state不变，0新encoder/AD/physics。此项补齐的是工程部署token证据，
不是Hi训练scaler认证、物理任务收益、三臂公平比较或P2科学准入。基于真实receipt推进独占四槽
full15物理adapter；首个plan动作之前必须有持久化真实history/goal guard，不用终态检查代替。
P3新goal-conditioned bank runner已冻结进入MAIN验收；有界CEM与直接latent几何对照协议获准实现，
均属探索性新方法、不改旧封存/门槛，不预写未来结果。完整研究目标仍未闭合。

终态补记29（2026-10-29）：P3 goal-conditioned actual-cp5 MAIN53tests/147.27秒、实际prepare/verify、
HOST CUDA130.288628429秒及整份saved consume均完成。report SHA
`9bedec46c5c0e74b73c145935280b2efe400f617f8bbe7b72205fedb60a20d4f`；receipt SHA
`b50268a2018e6da1fbc46a4d19d701cd4a9a575cfaf581807cf29c01861d7ba8`。四组bank全complete、
全no_witness、errors=[]，196predict/40J/72replay/7680VJP directions，无新physics。
MAIN独立标量曲率分解7tests/真实analyze+wholeverify完成，结果SHA
`0e0108514e229556cbd737ebd6e44dd7c19ff3dd791f59c61a4a302b65111dbb`；四组nongreedy3/0/6/7、
better_reach4/8/2/1、交集0，32候选10个一阶项与实际immediate变化反号。有限幅度非线性与
候选生成目标不足有具体定位，但不是J错误证明、物理负结果或一般理论否定；不放宽原gate。
直接latent交付预审发现inclusive计时和消费超时校验缺口，已要求冻结前修复，未据工程错误
产生科学结论。后续基线/目标/物理三实现仍推进；P1及完整三实验研究闭环均未完成。

终态补记30（2026-10-29）：P2四槽 commoncontroller full15 主线程53tests/22.50秒、真实
prepare/verify、主机原生物理执行和整份 saved consume 已完成。report SHA
`3164f9fb3808e6d579ae9aada8306427b0acbf9a657eda2f2e7e46f68f4f5076`，receipt SHA
`bf976c4f28f3b4634d2d78594779cc74933bf8870d78aef2db8cb182a49342b3`，errors=[]。
4 constructor/close、100 public steps、1016 physics steps、324 render 均返回；首计划动作前有
durable history guard。两根 program 的终态 joint_xy_error 均低于 prior，但 native7_distance
一根改善一根变差，四分支 native_success 均false，不能称稳定收益/公平比较/P2机制成立。
arbitrary companion 修复wall接缝后，MAIN87tests、真实纯数学 instantiate/complete consume完成；
receipt SHA `1e47155d480c23a0bda9cae50e633f0fe7ef56c992d3f489a31e255e43286ffc`。
两FIRST均交付，0神经/physics，不改旧missing。MAIN批准六槽controller全新job及固定四组
二维I/R割线方向搜索独占实施，不放宽gate；有界CEM135tests和prepare/verify通过，HOST CUDA
随后13.417531817秒完成，并已整份consume：report SHA
`f23014b2ee8a742ec097ec42f52ba5d326b73ea16cb27499af3399f5d7d0231a`，receipt SHA
`f4ebc1653a9bb69b70c280dc97dd1096447c2f98d30e749a0ecc40b987564c33`。
两根各9000候选，返回mean实际raw10一次normalize后独立whole/segmented逐位一致，
128predictor API/36008samples；无新physics。H2不同P2 H3，不直接排名或称官方原版/等预算。
五线并行独占文件，GPU生产仍MAIN串行，P4/P5 fresh前置及总闭环未完成。

终态补记31（2026-10-29）：共同controller物理saved独立数值比较，MAIN全文审读、30tests/
0.49秒及实际analyze→wholeverify完成，结果SHA
`d0b67d36ec9df31dba2aee81afcf4e0ffb387f950482baae1d0ac3d07570edaa`。
全30行/10配对，无新NN/physics/fullstrict，继承完整原consumer认证。root1/cp15 native7²
差值+7839.81767465分解为位置−5454.91028770、raw角−0.08954140、速度+13294.81750375。
root0/cp15则位置改善抵销速度损害后native7²−5290.69629578；完整较早cp同列，不择时刻。
这是具体指标断点，不是统一代码根因或理论负结果；原native success/角度合同不改。
MAIN另批准新H3有界CEM仅实现/RAM，将补同实际任务/时距，不冒称与P2球域和预算相同。

终态补记32（2026-10-29）：六目标共同controller完成MAIN76tests/10.25秒、真实prepare/verify、
HOST CUDA和完整saved consume；report SHA
`b81f12ee3602835bc14fd2e22d7e3644b8376f7ba8b0b265948bfb18429b6934`，receipt SHA
`592062bded7d6eb1716594b21fac4b5ef6255cf93d5285c2361f0ec2686c0103`。
同一新job新dispatch6/6returned、72API/324samples/1296AEtokens，425state/source完整验收；
没有旧4加新2拼接。暂未六槽raw部署/物理/任务收益，不改旧missing，不升级公平或P2机制。
P3二维I/R割线MAIN63tests/40.74秒、真实prepare完成，verify进行中，未有本版模型结果。
full30两个完整程序postcontext迁移协议获准独占实现/RAM，新时域/支持/成本另列。
P4/P5接点清单完成，fresh0前置未改变；五线继续推进、MAIN串行GPU，研究未闭环。

### 补记33：reach-secant模型资格有进展，尚非真实机制通过

覆盖补记32的待运行状态：四组固定16候选HOST CUDA完成138.322738048秒，
report SHA `191684824c95b39fcaf4cd2683ad3f3f7cd071009cdaaa076014bd9f67df91e7`，
整份consume SHA `d96f97dd076894901edea43f6223cf8cb32da4dc630968fea8fae7834b143151`。
80predict/16J/32replay/3072VJP全部付费，无错误；原严格gate不变，
root0/H1、root0/H3、root1/H1、root1/H3的资格数4/0/0/0。
MAIN标准库标量复算全部16项，9tests通过并完成真实analyze/whole verify；
`le-wm-main/results/main_p3_secant_gate_scalar_20261029_v1_MAIN01.json`
SHA `aa943b9a2c0947c6dca1f60d672df786ed76bb527ca7f85da54c3eac6df585c5`。
该adapter复用既有独立math.fsum核，不冒称第二套独立数学实现。
候选生成改变能改变模型gate资格，旧有限bank的无见证不能证明理论无解；
本版尚无delivery动作见证/真实物理/新根确认，不升级P1或C3调度收益。
下一环节协议独占推进，禁止按结果挑根、放宽gate或复写封存结果。

### 补记34：H3有界CEM与六目标预测双距离验收

MAIN214项回归、prepare/verify、HOST CUDA12.019269205秒及整份consume完成；
H3有界CEM report SHA `b3e239367a482a9ff561e69f7a09657540cac690eecd2adcba08d948c210d6a1`，
receipt SHA `d0e215618c367049c09eed7e93accb90615973f4f8cb322fd19359fef1f7e7a4`。
两根各9000候选，192predict API/54012samples/162036context tokens，
全部raw15落盘重读与整段/三段回放验证；无AD/新encoder/physics/训练。
同P2任务H3但支持、归约和预算不同，非官方原样复现或公平排名。
六目标双距离比较器46项MAIN回归及真实build/whole verify完成；
结果SHA `eb4e9acc5be07d4c9381d15a932acc14de2454d4db5aecb72f17a9b65fa447b7`。
全24行/24对，root0 prior cp15 tracking MSE0.0674412074，
但共同task_goal MSE1.3938647025、相对初始增加0.2396774183。
跟踪不同子目标的loss不直接排名，预测目标改善不当物理收益；
六槽新部署token重放与新物理比较仍需后续验收，不升级P1/P2/P3科学门。

### 补记35：六槽实际部署token主机CUDA重放已完成

覆盖补记34的待运行状态：MAIN全文审读、95tests（3.54秒）、真实domain及完整verify完成。
domain SHA `95c552c8c8e6edd480a4f782733c0367329a1b7b403408067e58eeb3c29f7c4c`，
6/6checked、blocked=[]；prepare SHA `9f367dbfdf91dc69ff4ca79562868a8bf16074bff55b91699f5894356a1dc5df`。
HOST CUDA全六槽实际完成、整份consume验收：report SHA
`c3733a752d1e9b80929edf9726bc0eba2ad1c2ae39b6f05550652ff7b3b74dbd`，receipt SHA
`6e5fa7c57721cd4765430c7379d501d19978bdb44eebe96dce94f7bc68d9a3db`。
全部6returned、0blocked/missing，36新predict API/36samples、144AE tokens/108context tokens，
至report fsync/hash墙时39.485927388秒，completion ack不在该边界内。
f32逆变换后实际再归一化token整段/分段重放全量认证；不是旧4加新2，不新增物理或训练。
原normalized-job与actual-token预测有轻微差异，真实执行误差须绑定后者。
secant动作见证与六槽物理比较两新协议已全文审读并批准独占实现/RAM；
执行和科学认证分离，P1/P2/P3及fresh前置保持原状态，研究目标仍active。

### 补记36：P2 full30两个完整程序实际迁移完成，科学机制未升级

MAIN四文件全文审读、39tests（6.69秒）、真实prepare/verify和HOST CUDA完成。
report SHA `99f2b5edceae839bc3b0387ac93ef1d866a3fb74ec869ec7a30ceeaf4211ae00`，
整份consume SHA `af2890ae4d330f69e8b01c7a144786ee454b08091e6d79ef63fa1b42e0bf6c5e`。
4槽完整、228predict/4个192×60 raw-J均返回，errors=[]；全whole30/两段15及同动作初值context精确核。
B本次输入是本次A返回完整context，未回填B旧root输出；每段15才算一个program，不按packet冒名。
root0 transfer/joint模型Q为0.8149349656/0.6037821767；root1均0.2164527070。
新H6时域与旧full15不同，joint支持/计算更强，known2非fresh；这仅验证迁移及联合优化工程链，
不证明组合优于强baseline、实际收益、P1/P2机制或等预算优势。full30新物理协议在准备。

### 补记37：目标速度语义及后续阶段准备

PushT固定环境审计确认：状态末两维为推杆速度，success检查位置和角度，七维reward距离还计相对目标速度；两根目标速度范数分别191.44856和188.55359，并非零。单张目标图渲染不直接读取速度，多帧history仍可能携带运动信息。十个真实终点状态与snapshot及目标一致；独立审计9项测试通过，结果SHA `8526bc6f1a914958858add0a39ea5190201da744455cea25f692a4fee7ec6927`。固定原轨迹的后验零目标速度敏感性使根0三对七维差值变号，但不是新执行或可事后替换的正式指标，也不足以解释全部P1失败。原科学门不变。

P2共同资源准入46项测试和完整六槽RAM schedule 6项通过；E3数值准备32项、E1合同21项通过并核验真实输入。它们分别仍缺真实HOST计量、完整生产fit及learned两槽来源认证，均不能当成科学闭环。C3冻结版全套测试session 38763尚无终态，不以旧快照结果冒充最终版。

### 补记38：E3 control5 主机CUDA及反馈接线边界

补记37中的C3最终版session已正常终态，25 passed/1710.61s；仅验证fixture运输与重建，不证明真实在线调度收益。E3 fixed16 control5数值节点已全部主机CUDA执行并各自saved consume：2176 predict、512 scalar reverse、0新物理、16/16科学admission=false。峰值allocated 142489600/reserved 144703488字节；known2而非fresh，尚缺真实控制与control10/15反馈，不能称三节点闭环。

补记38（2026-10-29）：P2 v1 主机六槽四个谱槽因 `AD_return` 真实事件约2009字节超过1400字节上限而失败，且顶层漏报槽内错误，独立 consumer 拒绝；不能将其解释为理论负结果。v2 修复事件限额并在原存储上限内重证、将槽内错误上报，36项测试通过；真实 CUDA 六槽自然完成并完整 saved 验证，receipt SHA `bcaba6c3c4cbdd05dafc475ad45ad859930efd3587fdb8e8adba47f7da7b7432`。joint/composition/GRASP 的实际模型调用与微分资源不同，共同上限不是同 FLOPs，GRASP 原路径 CUDA 等价仍待实测，尚无该轮物理闭环。

E1 六槽共同控制器真实 CUDA 与 saved 验证完成，receipt SHA `1ca8e949cc4b219b9b03adb3d34ee859e924e0078f9f81c519be2c03ed77be3d`，每槽144控制预测另加4 replay，整段与分段状态逐位一致；使用工程统一归一化，不是官方 Hi 复现，六槽物理执行仍待完成。C3 compact HOST 最终12槽fixture长测7项通过、真实来源 prepare/verify完成，真实主机 CUDA 当前在运行，尚无完整结果。

P1 既有两根80对中，76可比对的36个观测编码/native7距离排序反向里，34个在纯算术去除速度平方项后不再反向（不是物理干预）；40槽原raw动作均无精确边界触碰。非零目标 agent velocity 源于固定 witness 末态完整state7，native success 不使用速度、native reward距离使用速度。该目标语义差异为需前瞻验证的协议问题线索，不足以改原 gate 或判断唯一根因；P1、新根与完整研究闭环仍未完成。

16格各自33候选的winner均严格优于本格零初始化模型Q，但零初始化属于候选库，不能据此声称强基线或物理成功。MAIN发现未生产的反馈fixture漏校验prior raw5与既有history最后token；v2已加f32位级一致性和错接/signed-zero反例，联合10项通过。该修补针对潜在接线错误，未证明先前物理或负结果受其影响；完整连续physical/encoder与独立consumer仍待实现。

### 补记39：实际六槽物理、C3和GRASP核验（2026-10-29）

本节覆盖补记38中的待运行描述。P2 六槽 v1 物理 prepare 继承四槽分母 4，未被执行；v2 改为 6 并修正继承标签，预算和科学门不变，73项联合回归通过。v2 真实物理 saved receipt `le-wm-main/results/agent_p2_intersection_physical_20261029_v2_consume_MAIN01.json` SHA `e96462847db675dc0b0c5376067df3654edd40805d6e374ffa3f8f1d5224e6c4`；五槽满15步，root1 joint 在第14步原生成功停止；固定同时间分析保留第15步缺值，不能与满程端点混排。joint、composition 和 GRASP 实际模型资源不同，两个根已知，不是等算力泛化或 P2 机制证明。

E1 六槽 native 物理已完成并 saved 验证，receipt SHA `6644156f48a79a30f5118d9c56c3c3a06c3bc818e44edb55e6fe8bf4c29d38d4`；六槽各10步，success 0/6。共同控制器不是官方 Hi-LeWM 复现；不同中间目标的质量不能直接排名。C3 主机 CUDA 12槽及 saved consumer 已完成，receipt SHA `cdd61fc20ff4b819e64092f226cbe84ce3777edfc2a2c8472322a01ec3a0601f`；K1 六槽被预算预检拒绝，K17 六槽树完成，但不代表真实任务收益、同实际成本或 C3 科学通过。

GRASP v2 的 saved consumer 先因 CUDA `torch.linspace` 和 NumPy 的第二系数相差 1 ULP 而误拒：固定原报告精确重建后已通过。进一步在 v3 把两侧实际 CUDA fraction 落盘，真实 CUDA 与 saved consume 完成，receipt SHA `5040e084b549212941623b25afb3d1e0b0de37e489c5a473514c3b5a701bbe26`；固定输入、三packet、损失与记录梯度逐字节一致，不等于独立梯度重算、全部优化轨迹等价或等 FLOPs。P4 三反馈8臂同时间分析 SHA `aeaff75972afbb8400f8b3c458cacd6c45d143b6890f8449693f58e08e99916a`，早停后的比较留 null，费用未知项不填零。

P1 原目标语义六槽24位置有23个可用，14个可用位置的七维距离平方中速度误差占比过半；这些位置不是独立样本。该描述不能改原指标、证明唯一失败原因或替代严格非贪心机制与新根检验。当前仍无认证 fresh 根，E3 连续真实物理→RGB→编码→后续控制及强基线等预算、P1机制和跨模型/环境尚未闭环。细节、测试与范围见 `docs/D2P_PARALLEL_CHECKPOINT_MAIN_20261029_v3.md`。

### 补记40：P5 接线落盘与后续生产链（2026-10-29）

MAIN 独立审读 P5 固定164分支、80对的保存结果接线及持久化入口，联合28项测试通过；对真实封存输入写出的结果 SHA `25cb6846263027b1ceedd39b9060c3158c6cc6a095d323b012be30728e7511c3`，verify 从相同 join 重建整份字节通过，不重跑上游 full strict。仅两个已知根、认证 fresh=0；不同模型的特征误差不可直接排序，P1/P5 admission 均关闭。详见 `docs/D2P_P5_FIXED164_MAIN_AUDIT_20261029.md`。

C3 K2 共同预算是观察 K1 无法交付后提出的探索性后续设计，RAM 8项测试通过；E3 1800秒 native meter 和授权 episode 接线的11项测试通过。这些尚不是主机 CUDA 或连续物理生产结果，不证明 C3 收益或 E3 完整闭环。E1 与 P2 新 RGB 桥仍在补充来源认证、CUDA runner 和 saved consumer，未启动生产；不得把 RAM 接缝或其他批次 RGB 结果代替当前轨迹。

### 补记41：P1 多因素与 P5 固定80对分解

MAIN 对已有 P1 同动作对分解进行复核：76 可比较对中，实际 RGB 距离相对推杆位置误差反向2对，相对物体位置反向22对（另13对平局），相对物体环绕角反向33对（另13对平局）；模型到实际 RGB 排序反向12对。native7 分解测试10项通过；具体来源与原生速度、角度语义见 `docs/D2P_P1_MULTIFACTOR_MAIN_20261029.md`。这支持进一步核查观测表示对物体和推杆的敏感性，而不是仅删除目标速度项；不是因果干预或对模型能力的独立确认。

另一批 P5 固定80对保存分解经 MAIN 审读、12项测试及真实保存全字节重算验证；结果 `le-wm-main/results/agent_p5_fixed80_error_decomposition_20261029_v1_MAIN01.json` SHA `b74240a0c9a10f7c9b938f799b91be090d61a2f027f522c33c1743047582ddfc`。实测 LeWM 编码 goal Q 与 native 距离方向同35、反21、单方平局24；与物体环绕角同22、反29、双方平局11、单方平局18。JEPA 仅在同模型、同时间共同标签内比较误差；不与 LeWM 的不同空间做数值差。结果后提出的描述性分析仍为两个已知根、零新神经/物理、未重跑上游 full strict，P1/P5 科学准入保持 false。

### 补记42：P2 intersection 实际 RGB 编码（2026-10-29）

覆盖补记40中 P2 RGB 桥待运行状态。MAIN 审读、13项测试、真实 prepare/verify 后执行主机 CUDA；23个真实位置编码、固定24分母，早停的 root1 joint cp15 保持缺失。独立 saved consume 及重建 verify 完成，receipt SHA `3797fb394ad1a693272bcf21d70ea59c9c8c344ad4aab35be3462ee298ad3087`；只新增23 encoder/23 projector，不新增模型预测、物理或 AD。相同根/时间可比的16对中模型 Q 与实际编码 Q 严格反向3对；2 known roots、非独立样本、资源不匹配，不能证明 P2 优势或物理成功。cp0 是原观测而非预测。细节见 `docs/D2P_P2_INTERSECTION_RGB_MAIN_20261029.md`；E1 RGB、E3完整闭环和科学准入仍待完成。

### 补记43：E1 实际 RGB 桥（2026-10-29）

覆盖补记42中 E1 RGB 待完成状态。MAIN 审读、49项联合测试和真实来源重建后，主机 CUDA 实际编码六槽×cp0/5/10 全18个真实帧；18 encoder/18 projector，0新 predictor/AD/物理。saved consume SHA `86ee030e2b7a50e0412e41d9c16ae6a10c80a1b6beaf89831f04e943db2f33b5`，二次完整重建一致；root RGB/latent 精确一致。自身中间目标 tracking、共同任务目标与预测误差分栏，不能拿不同中间目标成本直接排名。两个已知根、原6槽物理成功0，不证明 E1 机制；E3 连续闭环与新根、P1/P2及平等预算仍未完成。细节见 `docs/D2P_E1_SIX_SLOT_RGB_MAIN_20261029.md`。

### 补记44：P2 配对误差独立复算

MAIN 审读并运行固定18对同根同时间分解，10项测试通过，真实保存结果 SHA `65da16658a05683599bd1f876b6af26ce100ac44c56652510050e9662294414d`，整份重算verify通过。早停后两对缺失、16对实际可比；模型Q→实际编码Q反向3对、实际编码Q→native7距离反向11对。逐帧模型误差平方与交叉项分解最大浮点残差1.11e-16，重算native距离与返回reward最大残差1.42e-14；这排除该批特定数值重建问题，不排除所有代码问题，更不证明理论失败。只继承旧strict、已知两根；细节与三轴口径见 `docs/D2P_P2_INTERSECTION_RGB_MAIN_20261029.md`。

### 补记45：P1 非线性候选、P2 新对照与 P3 实际物理（2026-10-29）

P1 四槽八臂 160 个已知前缀经同预算八步非线性后缀优化，122 个阶段上坡，8 个非线性正则化后续优于同臂优化基准，但没有上坡候选同时满足后续和原 winner 终态比较；另两包联合梯度探索 144 候选中 70 上坡、12 终态改善、交集 0，原 Vreach 未评估。两个结果均已 saved consumer 核算，不是新鲜根、物理收益或理论否定。有限差分在固定16方向小步与保存 Jacobian 相符，大幅度线性近似偏离；详见 `docs/D2P_P1_NONLINEAR_REACH_DIAGNOSTIC_MAIN_20261029.md`。

P2 新双根 H15 无组合谱候选真实主机 CUDA 和两根物理执行完成，物理报告 SHA `aecf685083df2160df0d11929b2bd67f3366f2dbcc59bb111d879c1a78c9d185`，两根均满15步且 native success false；同根、同时间配对和真实资源公允比较仍待验收。P3 K17 两已知根六槽 first5 物理执行与 saved engineering consumer 完成，306 帧及6 guard 重算、6槽 coverage true，但 `native_provenance_verified=false`、`complete_receipt_verified=false`、`scientific_pass=false`；相同物理接口调用数不等于相同规划计算。P3 配对已核验，根1 staticH3 与 adaptive 前5步动作相同，根间终点 reward 差异方向不稳。

E3 精简观测来源闭包下单个已知 cell 的主机 CUDA 实际反馈与原 inspect 完成，来源闭包运行后复验通过；`full_saved_strict=false` 和 `scientific_admission=false`，仍不代表完整三组实验或 fresh 泛化。P4/P5 历史来源与 coverage 缺口保持，所有核心科学阶段及完整研究目标仍未完成。

### 补记45：E3 生产入口的 reset 记录缺口

并行实现的连续 launcher/consumer 仍在开发，尚未通过 MAIN 生产验收。原 native meter 的 reset 记录覆盖计数与区域，但未记录两次 reset 的 seed、placement 参数和 sample_state 使用的 block 顶点；已有终态不能替代输入来源证明。当前要求新调用边界实时保存并重建这些输入，覆盖失败、早停和 cleanup，不对已有日志补造；此缺口属于协议／可复现性问题，不能作为模型或理论负结果。E3 的真实连续物理→RGB→编码→后续控制链、跨任务效能及科学准入仍未完成。

