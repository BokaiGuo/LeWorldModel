# 真实观察条件后缀：下一项可区分干预

## 最新：真实中点重编译已实际 CUDA／完整验收／MAIN 独立复算

以下覆盖后文“仍在实现”的旧准备态，不把模型下降当真实控制收益。
生产 v2 source SHA `e66bac861fce455ab6bb7a0cf12ba1644258c6c2e9cb6f2905973ac146e33567`，
MAIN 完整审读三件套、独立联合 **133 passed /22.98秒** 后冻结运行。
实际 prepare SHA `9aa3d63db85857b80c8a614a00a5a7b22c77c26e688e149970a83825c0d74411`；
report SHA `3de58e380acbef66269e69150cab7bca312604b7df88a0f1ae5663764d691b45`；
完整 strict SHA `09d617dbece12ce3ff2ca7246390e433454b1d8ac29042ab07eadd2be75238ec`。
路径 stem 为 `agent_nonlinear_suffix_observed_recompile_20261028_v2`，分别为
`prepare_MAIN01.json`、`cuda_MAIN01/report.json`、`consume_MAIN01.json`。

实际主机 CUDA 50.717秒（含完整来源核验），四槽全部 sealed、errors=[]、
模型全部参数/buffer 未变；恰 **72 predict /32 scalar reverse**，没有新编码、
物理、训练。每槽 iterate0 精确复现原真实 history 下同后缀预测，全部九个
score 和八个梯度均核 native journal、数组和资源，最终再独立 winner replay。

| root/condition | observed 原后缀 Q | 重编译后 Q | 新减旧模型 Q |
|---|---:|---:|---:|
| 0/off | 0.260187407 | 0.243705433 | −0.016481974 |
| 0/on | 0.325940316 | 0.290808163 | −0.035132153 |
| 1/off | 0.227339461 | 0.213957574 | −0.013381888 |
| 1/on | 0.325386654 | 0.311411350 | −0.013975304 |

MAIN 新独立工具 `main_observed_suffix_effect_20261028_v1` 完整重建 upstream
strict 并比对持久 receipt 后，用保存原数组重算全36个Q、stable选择及动作
bits；十项纯算术回归通过。结果 `main_observed_suffix_effect_20261028_v1_MAIN01.json`
SHA `c99c5ff5fa3c11544568d8beebe9073d7cca7d2246ef5b786fa8f04c15217904`。
全部选择 iterate8、prefix5 原位不变、suffix5 实际改变。

尚不能宣称修复控制：新物理 adapter 审读中发现 cp5 身份只在执行完成后
核验，MAIN 要求增加 **step6 dispatch 前**的 exact prefix51/cp5 守卫及事件
先后验收，并加入 mismatch 时绝不执行 suffix 的回归。这是新驱动的执行
准入接缝修复，不是已证明的历史 P1 负结果根因；目前尚未生成新物理结果。
另固定4预测的 held-old2 hybrid 历史干预已完成审读及独立联合134回归，
prepare 已通过并提交主机 CUDA；未完整消费前不作漂移影响结论。
仍仅两个 known roots、四相关条件，P1/P2/P3 科学准入与 freshness 保持 false。

MAIN 随后在实际保存 ABI 接缝回归中又捕获 **1 failed /34 passed**：新物理
adapter 用自身带空格 JSON canonical hash 验证 compiler 的紧凑 JSON hash，
导致同一合法 plan、goal 和 observed context 被错误拒绝。逐字段诊断均为
digest 不同，而 task binding、parent selection 和 midpoint 身份相同；已要求
仅用 producer 自身 canonical digest 认证 producer 字段，own execution plan
继续自身编码，并补双 codec 回归。尚未物理 prepare，不修改任何封存 compiler
或历史结果。这也是新接缝工程失败，不应算作研究方法的负向实验。

该接缝现已最小修好：own execution-plan/proof 仍用 mixed 编码，只有
producer 的 selection/goal/observed context 用 compiler 自身编码核验。
causal cp5 guard 也已加入，检查失败时无 step6 intent、无后缀 callback。
MAIN审读增量并独立联合 **53 passed /35.44秒**（44 adapter+9独立物理算术）
后冻结，source SHA `246739a509c8727bb9381d0c5df3c9649ab6dd9b484ec8f4b8f9bb5891e45a65`。
实际完整 prepare 已成功（SHA `6fb3da89aef4096a80686e4e5facbbcb10d24ca733fc576bffa505c80a8f88a2`），
四条新 native 执行已提交，尚待最终报告/完整strict，不从启动推断控制结果。

### 新动作真实执行及完整物理验收已结束（覆盖提交态）

四条均从原 root/warmup 新环境完整运行，cp5_guard 全部先于 suffix intent，
完整 prefix51/cp5 状态/图像精确相同。实际80 public/816 physics/264PNG、
9.759秒（native窗口，不含前置来源核验），errors=[]、原生成功0/4。
report SHA `5a05a79996d49a5e90dae7002cc09d07d04291029e74d988e9b18211bdff9a26`；
完整消费 SHA `268074c84ca2aa0b3b1ae3354894021e0e4e06590fda6cb4cde89e4bb0cc0b6a`。
stem `agent_observed_suffix_physical_20261028_v1`，没有继承行或新独立根。

MAIN `main_observed_suffix_physical_contrast_20261028_v1` 再完整重建该strict，
比对持久receipt后独立重算state7与原success/reward轴；九项算术回归通过。
实际结果SHA `bcb8660a80de940fd84a16e4bddac30ba6a28316ead519902b7762b59bcb69f3`。
同一prefix/同一原native goal的旧suffix与新suffix变化如下（负数为改善）：

| 条件 | jointXY | objectXY | angle | native7 distance |
|---|---:|---:|---:|---:|
| 0/off | −5.223962 | +3.386678 | +0.011205 | +8.269342 |
| 0/on | +2.525803 | +8.178895 | +0.111109 | +17.121762 |
| 1/off | −10.423138 | −5.547733 | +0.061120 | −39.982317 |
| 1/on | −10.964688 | −4.629724 | +0.120351 | −21.310291 |

jointXY三优一劣，angle四条损害、成功仍0/4。真实反馈重优化不是统一修复，
不能删掉不利的角度/速度指标或改成功阈值。新终态RGB桥已冻结，MAIN审读并
提交prepare；尚待主机CUDA/完整strict以区分预测→实际编码与编码→物理两层。
八文件联合 **320 passed /70.13秒**；测试是工程证据，不等于P1通过。

### 新四终态 RGB：主机 CUDA 与完整严格消费也已完成

新桥三件套 MAIN 最终增量审读、source/tests/doc SHA核对及独立联合
**133 passed /34.19秒**（68桥+59原桥+6 MAIN分解）后prepare。
source SHA `4c84f0d735981ec0a592d1df90ecaf6215390953f05ec7cd1dd438cb47b66427`；
实际 prepare SHA `9478a9e944b831b558b9148c3858f5242bd488948841923138ab10952df365b5`；
report SHA `deb1e01a819d3a78dcf9aa4d9b9968821a17d56406bf4ae8d2c7ad7ee5e3890a`；
完整 strict SHA `4a4eddd0b846a93c8a912eaefdde0fd17a24e8d354002d19464fea44bf24045c`。
stem `agent_observed_suffix_terminal_rgb_20261028_v1`。实际52.842秒（含完整
来源核验），四张新cp10图各一次encoder/projector，predict/AD/新物理0、
errors=[]、模型前后完全相同。原PNG→独立预处理→257hidden→CLS→192
完整数组及事件/guard/resource/SHA核验，没有复用旧latent冒充新编码。

| 条件 | 新后缀预测Q | 新后缀实际编码Q | 新预测endpoint MSE |
|---|---:|---:|---:|
| 0/off | 0.243705433 | 0.116424091 | 0.303192293 |
| 0/on | 0.290808163 | 0.140537806 | 0.364587949 |
| 1/off | 0.213957574 | 0.125614146 | 0.101045976 |
| 1/on | 0.311411350 | 0.201225598 | 0.099519170 |

MAIN新三层工具正在完整重建上述strict并独立读取旧新原NPY；比较的旧预测
固定为**同实际cp5 history下原suffix**，不使用旧imagined-context基线。
所有任务原生轴、误差cross term、反向项均保留；新分解未完成前不签统一根因。

该 MAIN 完整重建与独立原NPY分解现已完成，实际结果
`main_observed_suffix_rgb_contrast_20261028_v1_MAIN01.json` SHA
`55ee1bef2ef77b16fe2153a4d2eecfa494c909da3b82ef2c514bfd56eac24d22`。
六项纯算术回归验证带符号恒等式及task/cp5绑定；原新两端编码属于各自
真正raw10/step10，完整物理指标亦再次独立复算，没有按结果筛掉任一条件。

| 条件 | 同observed history预测Q变化 | 实际编码Q变化 | jointXY变化 | angle变化 |
|---|---:|---:|---:|---:|
| 0/off | −0.016481974 | +0.023567695 | −5.223962 | +0.011205 |
| 0/on | −0.035132153 | +0.065948660 | +2.525803 | +0.111109 |
| 1/off | −0.013381888 | −0.027493995 | −10.423138 | +0.061120 |
| 1/on | −0.013975304 | −0.037741813 | −10.964688 | +0.120351 |

**两类剩余断点都仍在**：root0/off/on的模型Q下降反而实际编码Q上升，
说明完整真实history条件下仍有动作响应排序失配；root1/off/on则模型与
实际编码同向改善，但角度均损害，说明只修预测还不足以保证原任务改善。
root0/off同时有编码变差、jointXY变好，进一步说明不能把编码与任务轴互换。
四对response-error MSE为 .008569830/.033548713/.008574909/.015584077；
四对带符号项均相消、均不符合精确动作无关偏移，不能拟合统一offset绕过。
向量端点MSE也不是任务代价误差或排序可靠性的替代量。

最终七文件联合 **241 passed /69.07秒**，`git diff --check`通过。
本轮有真实代码接缝修复、主机CUDA重编译、全新物理及真实编码闭环；
但没有找到一个能统一解释并修复所有P1负结果的单一代码根因，亦没有证明
理论不可能。历史checkpoint训练normalizer身份、任务表示充分性、候选与时域
覆盖、需要重配置的prospective任务设计及独立fresh确认等仍须分别推进。
不把已知根上额外优化或上述工程通过倒灌为P1/C2成立，研究总目标保持active。

### 历史重编码漂移：固定四预测的实际干预也已完整完成

上文“提交中”现已结束。held-old2 hybrid 是诊断而非真实完整反馈历史：
只保留原 root 后两张编码，末张仍用同一个实际 cp5 编码，raw suffix、
incoming/outgoing history、goal/Q 和精度全部不变。新模型调用恰4predict，
无 encoder/AD/物理/训练；四槽全返回，模型前后未变、errors=[]。

`agent_nonlinear_suffix_history_control_cuda_20261028_v1` 的实际 prepare SHA
`48913eea272b26466a7acebbc28f022b63ee2858c0b00f3aabbe743e8b7459b6`；
report SHA `e620e43bc1e283ea93160032819c48e771e1d7b0e2f4782b9aff67e03ab64b15`；
全量 strict SHA `9d8941633b07cc0ba0eb43eb8cd4c93352594850b1d7fb2575ad6c4149772253`。

MAIN另独立读取准备/回执的192维数值，重算四组误差、响应平方项及带符号
cross term；恒等式的最大算术残差4.09e−17。hybrid减完整observed history：

| 条件 | endpoint MSE变化 | goalQ绝对误差变化 | endpoint响应MSE |
|---|---:|---:|---:|
| 0/off | +2.84636e−8 | +1.59663e−8 | 2.03069e−14 |
| 0/on | −1.19696e−8 | −1.61396e−8 | 2.67689e−14 |
| 1/off | −2.65994e−8 | −1.87682e−8 | 1.51784e−14 |
| 1/on | −1.17141e−8 | −3.46458e−8 | 1.75804e−14 |

本固定预测下，旧两帧重编码漂移**确实会改变输出**，但实际测得的MSE/
goalQ误差变化仅约1e−8，不能解释此前约0.01–0.17的条件预测差异。
这是干预量级证据，不是“输入看起来小所以忽略”的推断；不外推其它模型、
更长rollout或原80对，不将GPU差异唯一归因batch布局，不改比较容差。

上一轮补齐精确历史RGB复用与四对分解，292回归通过；本轮保持完整研究目标。
P1/P2/P3科学准入尚未成立。normalizer当前常数一致，但历史训练身份仍未认证，
不得重算子集统计或按结果换常数冒充历史认证。

## 已有168候选的覆盖（不是新CUDA）

MAIN核对原完整strict后分析 `main_nonlinear_suffix_effect_20261028_v1_MAIN01.json`，
SHA `86651769dbd4357675f02b7b72f854bf8d18abd74a93d71ad7c2651e7db4b449`：

| root/condition | root Q | 先退步prefix数/42 | 最佳先退步终态Q | 全库最佳终态Q |
|---|---:|---:|---:|---:|
| 0/off | 1.154187284 | 20 | 0.506967218 | 0.386614428 |
| 0/on | 1.147924157 | 27 | 0.436930855 | 0.374633970 |
| 1/off | 1.388804530 | 18 | 0.345204901 | 0.135648054 |
| 1/on | 1.556512648 | 18 | 0.331780075 | 0.174545877 |

不能称库里没有任何先退步且模型终态受益程序；缺的是它们相对直接进展
最优程序的优势，模型Q也不等于实际任务收益。不能强制选先退步程序再用
其类别冒充选择器发现非贪心优势。两个known根四条件，无独立确认。

## 预先固定的区分实验

1. 固定新四winner的raw10/task/Q/goal及实际物理，只将cp5 imagined history
   替换为真实最近三帧和对齐动作history，预测原后五步。主指标为相对同一个
   actual cp10编码的endpoint MSE，次指标为预测goal Q绝对误差；不宣称控制
   收益，不拿条件不匹配的旧baseline声称排序修复。
2. 上述真实观察ABI冻结后，另版保留原prefix5，真实history上重优化后缀：
   原suffix为iterate0，固定8步/eta1/原域/原Q/stablebest9。新动作另需完整
   物理和终态RGB，不训练模型、不从模型下降直接宣称收益。
3. 新scalar-gradient以四槽原generationwinner的iterate0为固定锚点，十坐标
   正负与三预定步长共240请求FD，无新AD。域外/折叠保留缺项，不clamp或删。
   以前另一actual-midpoint编译器的矩阵FD不能直接认证这条新scalar路径。

MAIN第一项独立分析source/tests/doc已实现，14项单元测试通过；结果前协议
排他封存于 `le-wm-main/results/main_nonlinear_suffix_midpoint_effect_20261028_v1_protocol_MAIN01.json`，
SHA `d421793a24ff9c38789f9baa419ff183ecbb36d2bd0082884f313ef763461a18`。
包含三文件SHA，status为`analysis_protocol_only_no_actual_results`，非真实产物。

三个现有worker分别独占实现生产输入、重优化和FD，MAIN负责审读/验证/派发。
本文记录时生产源尚未冻结，没有新CUDA或物理效果；单元fixture不是实验。
神经计算只主机CUDA，不降级CPU、不训练/拟合偏移、不改根或成功阈值。
没有任何新GPU作业可以据此声称存活，也未重启旧作业。

## 真实中点主机CUDA与完整验收已完成（覆盖上文准备态）

MAIN完整审读生产source/tests/doc，修复的是RAM fixture漏资源字段，不改
生产校验；最终五文件联合119 passed /13.39秒。生产源冻结SHA
`b1371a0de4d805bac1b4b64cfb625b7d327a3772d61d68c092804430f94a7ece`。
实际prepare SHA `6dd4ef991ebf2947388ae7c139a23776592b200e11032fdc106be281ae9cc5e8`；
实际report SHA `1639745d33424a28e39a36924b4fe161a00aed2f5a6ae035352c898a9c8e88a0`；
全strict SHA `127c650eace2c41e43dc0361aa13009981acfc8e9509dfff7a1f924283f87521`。
路径stem `agent_nonlinear_suffix_midpoint_rgb_20261028_v1`，prepare_MAIN01.json、
cuda_MAIN01/report.json、consume_MAIN01.json。46.213秒含完整verify，
4encoder/projector（12图）及4predict全部返回，模型前后未变，errors=[]，
新增AD/物理/训练0，不是执行收益。

MAIN另从原full3 NPY独立重算，并再次重建整个strict，结果
`main_nonlinear_suffix_midpoint_actual_analysis_20261028_v1_MAIN01.json` SHA
`14a2da806b0df7d90ac64c12edfb24839c262a028049c726438c20c4c09f374e`：

| root/condition | 原imagined endpoint MSE | 真实history后MSE | 原goalQ绝对误差 | 真实history后Q绝对误差 |
|---|---:|---:|---:|---:|
| 0/off | 0.449716220 | 0.281968457 | 0.293758032 | 0.167331010 |
| 0/on | 0.334035675 | 0.263282632 | 0.300044825 | 0.251351170 |
| 1/off | 0.162147192 | 0.113830020 | 0.017460087 | 0.074231321 |
| 1/on | 0.072497078 | 0.086395903 | 0.064421535 | 0.086419243 |

endpoint三优一劣，goalQ预测误差两优两劣；不能统一归因imagined前缀。
尤其root1/off的向量预测更近却goalQ更不准，符合原带符号项相消的事实，
不能删除交叉项拟合共同偏移。四物理动作均未改变，不存在新控制收益。

相同旧观察的前两latent重编码不逐bit相同：MSE 3.89e−14至1.06e−13，
最大坐标差9.54e−7；新中点与imagined latent MSE .0561/.0663/.1822/.0975。
不因数值小宣称对输出无因果影响。已派独占四predict held-old2混合history
诊断，明确不是真实整history或生产反馈输入，不改原实验。MAIN主分析和
独立history补充检查联合26测试通过，原事前协议三文件SHA全部保持。

240请求标量FD也已实际主机CUDA结束（43.815秒、errors=[]），完整strict
消费进行中，尚不从其未验收结果宣布梯度正确。实际report SHA
`f0e648b9057db5ccea1e91f7edbd5f34b2ab7cfb38a59b1f566ba4f616e64c63`，
prepare SHA `a39785c864eae2c8cdc100f93ef34744eef77c6796c5efaf27d3b8923605b6db`。
生产FD源/tests/doc最终审读、独立79 passed /8.56秒；signed-zero完整原生
接线检查保留，未改旧封存源码。真实观察重优化v2仍在实现，未有新控制结果。

## 标量路径240请求：实际完整strict与独立复算已完成

上文“strict进行中”已结束：实际完整消费SHA
`71486805c02bad9cdb977ee833019dc410f3a64e06d0950140215b5e145aaf45`。
全240signed请求/120对均合法返回，新增predict240、reverse/encoder/物理0。
MAIN独立解码全部GPU native float32 loss字节及SHA，复算120差分及12整向量，
与strict全部数值相同；另完整重建upstream consume，结果
`main_nonlinear_suffix_scalar_fd_recompute_20261028_v1_MAIN01.json` SHA
`7e8a250900022ee17af3b5d6c7eb0594e15fff76e19834efeceba0e140587820`。
独立算术9项测试通过，非GPU reduction重新执行。

| 固定名义h | 四条件relative L2范围 | 四条件cosine范围 |
|---|---:|---:|
| .01 | 0.000057779–0.000153310 | 0.999999990–0.999999998 |
| .001 | 0.000297011–0.000598861 | 0.999999823–0.999999956 |
| .0001 | 0.002481444–0.011851084 | 0.999929924–0.999996951 |

所有h均保留，不挑best-h或新增通过阈值。小h时标量float32减法消去误差
可能更显著，但本实验没有单独测出每个误差项的因果来源。此四固定锚点
未发现粗大符号/转置/尺度接线错误，不推广到全部迭代、其它checkpoint
或真实物理；AD_correctness_certified与P1科学门仍false。
