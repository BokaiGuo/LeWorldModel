# P1 锚点/响应分离与 Hi-LeWM 参考实现修复

## 研究目标与本轮边界

原方案的C1动作见证、C2真实非贪心收益、C3动力学驱动调度与三组核心实验
仍是完整目标。上一轮为实质进展（新增可复算分栏与代码），不是研究完成。
本轮继续区分工程错误与受测模型/目标的失配，不改科学门、不训练模型，
不以未知来源的统计替换训练normalizer。P1与完整强基线/泛化仍未闭环。

## P1：即使知道正确共同终态偏移，也不能消除全部反转

新独占工具 `le-wm-main/experiments/spectral_reachability/main_raw_anchor_factorial_20261028_v1.py`。
读取原164探针已完成的模型/真实终态和完整strict封存，两个已知根×renderer，
两h×10坐标均保留。9项非神经测试通过。没有新神经/物理、拟合、动作选择或
独立确认；这是feature-space替换的精确代数，不是模拟器因果干预。

以同一slot的原baseline为锚点，设模型/真实锚点为p0/a0，探针终态为p±/a±。
四个二次目标secant均使用原goal192和实际float32动作跨度：

| 路线 | 用于计算目标的两个终态 | 意义 |
|---|---|---|
| MM | p± | 原模型 |
| AM | a0+(p±−p0) | 仅oracle共同锚点替换，动作响应仍是模型的 |
| MA | p0+(a±−a0) | 仅oracle实际动作响应替换，锚点仍是模型的 |
| AA | a± | 原真实编码 |

完整交互项保留并用独立内积公式复核；不能把主效应/交互删除后叫修复。

| 路线相对AA | 同向 | 反向 | 真实目标零响应 |
|---|---:|---:|---:|
| MM | 22 | 34 | 24 |
| AM | 35 | 21 | 24 |
| MA | 31 | 25 | 24 |

原34反转中，oracle锚点替换后仍19反转；另引入2个原本不反转的反转。
这使“只减一个统一偏移就能根治”在本固定面板上得到直接反例。
偏移确实贡献部分误差，但它与动作响应失配共存；不能反过来证明理论不可能。
24真实零响应始终不计方向正确，两个h分别列出，不能挑有效h发布。
所有oracle量只用于回顾诊断，不送入planner，也不把35/56同向称新控制收益。

实际结果：`le-wm-main/results/main_raw_anchor_factorial_20261028_v1_MAIN01.json`，SHA256
`7074aed0feb474e594831bfbc1e4f6bcbd9378e67c77c18f7966d116d852e71b`。
该工具绑定原paired80、model prepare/report/strict与RGB full strict的字节，
并逐对重算原MM/AA scalar精确相同；不冒称再次重建全部上游。

## Hi-LeWM：主机 CUDA 确认参考实现布局失配

MAIN完整阅读已有v2 source/tests/doc并实际34测试通过（7.18s），然后执行
固定B2/S2/T5、相同seed3输入的五路布局对照。未调用encoder/macro encoder、
AD或物理，未训练模型。RTX5060Ti/CUDA13.0/Torch2.12.0+cu130，
deterministic开启、matmul/cuDNN TF32关闭，真实模型参数/buffer字节前后不变。
40 high predictor调用（owned10/reference30）、25独立投影诊断、61顶层dispatch，
这些计量嵌套不相加当统一compute。完整执行2.579s（模型内报告wall）。

| 新封存产物 | SHA256 |
|---|---|
| v2_prepare_MAIN01/manifest.json | eadbb1d11383a46b02866a1966588b482dccbb45d6684a55c812d81925782c72 |
| v2_cuda_MAIN01/report.json | 8f19346c99aba5c3cc5dd6efec878356938c24bcb53023444c37791f94ff6771 |
| v2_consume_MAIN01/receipt.json | 0c2b3055c91cd01101774511e8dff24693ec46caa3a91a7ef43b495eeba0a4d9 |

共同前缀为 `le-wm-main/results/agent_hilewm_trusted_cuda_validation_20261028_`。
完整consumer已实际通过，随后MAIN不用producer的summary函数，直接从全部
五步25组原NPY独立复算same-actions/source/legacy/contiguous/restrided/projection
六组布尔对照，精确重现结果。

- 两条autonomous旧路线分别精确重现v1的owned与reference，旧失败没有消失。
- 原源码逐步reshape→append→stack的参考路线，与owned完整五步逐位一致。
- 旧prefix-slice→reshape路线有3473个不同float32元素，最大1.4901161e−7。
- 同值/同历史干预的首步，差异已出现在macro_to_condition投影；其后共同历史
  四步没有差异。首步legacy连续化回归source，反向恢复legacy stride/offset又
  重现legacy输出。原生完整与手工source路线精确一致。

所以**本固定case的参考实现数值失败来自布局不忠实**，不是“模型理论不成立”。
这只是源码转写的一致性修复证据，不解释P1约.01–.17的控制误差，更不认证
官方pickle的历史bytecode或完整Hi-LeWM基线。

保存回执曾误用old.write_new到庞大results根目录，被32MiB bundle cap拒绝，
没有产生回执文件；改在独占新consume目录保存完整回执。没有改cap、原报告或
任何科学数值，此接缝失败不算神经/模型负结果。

## 四项修复验收已经完成：不是完整基线放行

MAIN完整审读冻结v3源/测试/文档，独立36测试通过（13.82s），随后实际主机
CUDA完成两项direct、五步rollout及cost四项原固定case。全部owned/reference
输出逐位一致，errors=[]，报告wall为3.205441534秒；完整saved consumer通过。
参考rollout使用源码逐步stack布局，不改owned模型路线，不放宽精度阈值。
cost还独立核terminal-goal平方和的float32规约误差界；该误差界不替代两路
输出逐位一致条件。22次high predictor调用及8项顶层dispatch属嵌套计量。

| 新封存产物 | SHA256 |
|---|---|
| v3_prepare_MAIN01/manifest.json | 4632b5b692a9ef1ceb42c72ba3bd15f004dd934f8c022dfde9b81b365ea88c06 |
| v3_cuda_MAIN01/report.json | c6fd37e6e4540a82bb1bcb168a34cb4d222c81cf82f9cce88b1b44f557e27c20 |
| v3_consume_MAIN01/receipt.json | 6151e93c40fc1c5b053b81f8ac73a8dca7644a8a037492686a50a2718e24ddc6 |

共同前缀仍为 `le-wm-main/results/agent_hilewm_trusted_cuda_validation_20261028_`。
高/低层配对、宏编码、像素前处理与实际CEM/真实闭环，仍不因这些高层固定
case通过而自动通过。official_parity、baseline_runnable、scientific_admission
与旧v1失败release均保持false。

## 真实高层模型与安装CEM的联调已完成

MAIN完整审读新三件套，独立39测试通过（1.51s），v3四case实际通过后才
授权执行。新runner没有替代root.get_cost：通过HighBatchBridge直接调用冻结
owned高层模型，B5/B2、chunk2、8samples、3iterations、topk3、H2/block2固定。
原CEM不clip候选，Box仅接口元数据；合成输入仅用于integration，不是任务根。

主机CUDA完整12次cost调用、168候选行、48次high predictor调用/672样本行/
1680 context tokens，0 encoder/宏编码/AD/physics，模型425 tensor值前后不变。
完整consumer通过，实际wall4.213590208秒；没有放宽神经输出一致性条件。
MAIN另直接读原NPY独立复算全部168个terminal-goal代价，float32规约相对
float64平方和的绝对误差最大0.019583728；在本合成大数值输入上处于预定
gamma194误差界内，不能混用为P1目标误差或神经预测误差。

| 新封存产物（前缀 `le-wm-main/results/agent_hilewm_real_high_cem_20261028_v1_`） | SHA256 |
|---|---|
| prepare_MAIN01/manifest.json | 0a50badc0942a20deeed5ed28bcba32ecb7a8d5f15b8708884564052a1fe07f9 |
| cuda_MAIN01/report.json | 8fe58985700dcdefa79bcfd56f19f1c857781a926a31eebe5a220f83b0e97801 |
| consume_MAIN01.json | 7dfb4b49d552a6248da8b603e027a95c541c5db95af4fab777828ed412886e43 |

该步补齐真实神经cost接线，不是完整Hi-LeWM baseline、官方原pickle parity、
真实宏动作执行、等预算收益或P1机制证据，相关科学放行均false。

## 本轮尚未完成

P1动作响应与任务度量的部署修复、独立根、强基线公平比较、P2组合收益、
P3等预算调度收益、P4完整多轮与P5跨模型仍需各自范围的实际证据。
