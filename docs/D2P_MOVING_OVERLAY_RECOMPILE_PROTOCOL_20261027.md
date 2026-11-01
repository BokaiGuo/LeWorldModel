# Moving overlay完整重编译与真实执行：预声明

本轮是检验输入渲染条件是否影响从冻结模型生成的新控制动作，不是再给旧动作换评分。方法、源码、动作发布、真实执行与评价分开。不得用真实终态挑根、挑臂、改动作、改阈值或选择有利指标。

## 固定来源与比较对象

来源为已完成paired source：`le-wm-main/results/p1_moving_target_overlay_source_20261027_v1_main01/report.json`，外部SHA `85ca1df3de910628bfd6c5bd1a0a482f61880a2ee76bfd404732a77dd8c7666a`。只使用其中原两个known calibration roots的四个真实blind NPZ；不是独立新根，不授予P1 admission或泛化结论。

固定顺序为root0/off/GN、root0/off/D2P、root0/on/GN、root0/on/D2P，随后root1相同四项。八臂完整分母，不按新旧结果择优、不换根、不补失败臂。

off/on仅来自原生renderer `with_target`切换。环境默认goal_pose轮廓不是构造的task goal；本轮不修改goal_pose、物理状态、相机、颜色或目标来使图像好看。每个condition的goal RGB均来自相应condition下同一真实witness终态，不能拿off目标配on历史，也不能用模型生成目标图。

## 规划合同

每condition独立编码真实历史0/5/10和goal，共四组十六图；不复用off embedding作为on输入，不将旧计划作为新优化结果。原encoder/predictor/projector参数、normalizer、identity192目标度量、H2和每packet五条raw动作均不改变。不拟合Q，不训练任何模块，无CPU神经降级。

各slot/condition使用原冻结GN/D2P内核重新优化。GN每组78predict，D2P每组66predict，共576predict；ADforward包含在预测账内，API与内部模块计量不相加。导数、SVD、solve、encoding和墙钟单列，不能称GN与D2P等FLOPs。所有实际输入/模块输出、候选、选择与动作seal必须保存；失败不重试填满配额。

主线程先审读最终source/tests/doc并确认源码冻结，再prepare、verify、主机CUDA执行。只有全八计划合法封存并通过saved consumer，才允许交给物理执行器；后续真实标签不回流修改本批计划。

## 物理与评分合同

每臂重新构造并计量环境，按原seed/placement/reset/warmup10重建同一root，真实root历史和goal身份必须与对应来源匹配；随后原样执行sealed raw10。不得中途restore/reset、改动作尺度、补造successor或借别臂终态。控制时with_target保持原环境设置，on图可经局部off/on/off真实render得到，切换前后物理/goal/renderer守卫不变。

实际constructor、reset、physics、public step、state setter、goal setter/bind、render及失败attempt/return持久化。合法native early-stop与工程失败区分，保留缺失checkpoint。真实状态是七维；不再用五维mock接口假设。native success/reward/termination沿原环境，wrapped-angle、objectXY等只作明确标注的诊断，不事后替代native成功定义。

真实RGB evaluator固定最多24个checkpoint图和四个condition目标图，即28encoder/28projector，无新predict、控制、AD或训练。每图先保存实际preprocess/encoder/projector证据；缺失不补图、不写0。无此新CUDA产物时，saved physical comparison的latent Q必须为未评/null，不能用旧embedding或物理距离冒充。

比较保留每root、每arm的on−off差异、真实prefix/terminal指标、native成功及全部缺失/失败。n=2仅校准描述，不提供独立显著性、P1通过、模型已修复或理论否定的主张。特别是“目标距离下降”不等于“真实未来控制能力改变”；C2仍需单独真实能力探针、非贪心与强基线证据。

## 执行状态

本预声明写入时新八计划重编译及其物理/RGB结果尚未完成。官方cached rollout等价实验已实际结束，不能拿它代替本轮控制实验。后续实际prepare/report及SHA记入CUDA_PROGRESS，不追改本预声明的比较对象和评估定义。
