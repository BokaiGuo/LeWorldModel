# Constructive GRASP mechanism / real-H3 bounded baseline v1

只新增本 stem 源码、tests、innerdoc，不编辑任何 sealed/source/results 或共享文档。MAIN 独占实际 prepare/CUDA/physics/RGB；作者仅运行非 Module 解析张量代数与 RAM 测试。没有训练模型或读取 witness/native outcomes。新目标创建尝试被 API 拒绝：旧 blocked goal 仍 unfinished，未假标 complete。

## 一手来源与忠实性边界

先读外层 `docs/D2P_BASELINE_FAITHFULNESS_AUDIT_20261026.md`：其 restart-Adam proxy 不等于 GRASP。此后项目已取得并封存小型官方源码；不能将10月26日“未取得”误当成当前仍缺源码。

本实现参考并校验以下本地官方文本，不执行/import远端代码，不下载：

- `results/p2_grasp_official_contract_20261027_v1/sources/grasp_pseudocode.py.source.txt`：仓库 `michael-psenka/grasp` commit `64f988aebf04ad9f674f9821bc6d2b9e22879105`。
- 同目录 `lewm__grasp_solver.py.source.txt`：官方 README 链接的 `michael-psenka/le-wm-grasp` commit `94be3c42aa2f0867c6cb1e38bda498b293b6921e`。
- 同目录 `lewm__config__eval__solver__grasp.yaml.source.txt`：固定默认750 joint、lr_s=.1、lr_a=.005、state_noise=.005、sync_interval50、sync_steps10、sync_lr=.05。
- 两仓库 commit/tree、源码/config 的 Git blob SHA1、全部8份小文本 SHA256、原 MIT LICENSE 都在 prepare 中绑定；缺失或污染则失败，绝不降级普通GD。

官方 LeWM 原文映射：line421中间虚状态插值，428联合Adam，458状态输入detach，467 soft dynamics+dense goal，474虚状态Gaussian noise，477周期sync，483–498 fresh action-only serial sync/replace actions/reset联合Adam且不刷新虚状态。确切逐行文本与完整原始MIT notice保存在上述封存目录，LICENSE包含 `Copyright (c) 2026 michael-psenka`，本实现不移除该notice。

本项目的旧 `p2_grasp_reference_core_20261027_v1` 明确是 **H1-reset/无界normalized action** 子集，不能直接接现有H3并称官方等价。当前要求同时保留真实H3与合法logit球，故本实现明确命名 `GRASP_H3_bounded`：保留所有上述关键GRASP机制，明示扩展历史和动作坐标，不声称官方H1/DINO或论文表格逐数值复现。`official_equivalence=False`，`paper_results_code_commit=unknown`。官方训练权重不是本实验必需资产：按任务要求使用本项目同一冻结LeWM；因此也不能声称复现官方checkpoint的论文成绩。

## 数学与执行合同

唯一输入是原两份 approved blind export 的原task/goal/Q/history/checkpoint/normalizer，不读sourcepack、certificate、witness或物理报告。固定source_slot0后1，每根仅一个 `GRASP_H3_bounded` arm，无额外独立根；history为真实观察0/5/10，未来horizon为两包5步。seed=`20261027+101*source_slot`，局部设备torch.Generator，不污染全局RNG。

原状态 `Z=[z0,z5,z10]`、动作历史 `H=[h0,h1]`；虚状态 `v` 是完整192维的未来中间边界，初始化 `(z10+goal)/2`。动作变量 `x[2,10]` 初始零，raw为逐float32 NumPy tanh；原任务normalizer将raw五步展平为normalized token。无 `/100`、无双重normalization、无inverse或raw clipping。

lifted edge0输入原始 `(Z,H)`；edge1输入 `[z5,z10,stopgrad(v)]` 及 `[h1,token(raw0)]`。全部状态输入detach，但目标侧v仍可求导，历史中的当前第一条动作token仍可求导。令两条预测为p0/p1：

`L_dyn = mean((p0-v)^2)+mean((p1-goal)^2)`

`L_goal = mean((p0-goal)^2)+mean((p1-goal)^2)`，联合优化 `L_dyn+L_goal`。末edge goal计两次是官方公式，不擅改权重。loss为float32，交付terminal_cost使用与旧task相同的float64数组own-Q计算。

每次joint Adam后仅给v加固定std Gaussian；两条动作packet分别投影回原cap=.75sqrt5 logit球，沿用 sealed screen.projected_logits 的float32 inward rounding。实际raw forward严格NumPy float32 tanh；对logit的首阶链使用 `raw_constant+(x-stopgrad(x))*(1-raw_constant^2)`，与既有GN解析链一致，只做一次，不以torch tanh的潜在末位差异换掉真实commands。不实现二阶导数。

sync在0-based `(k+1)%50==0 and k>0` 触发：从原真实H3开始的完整autoregressive两包，保留整个状态链梯度；fresh action-only Adam10步；每步同一投影，结束后只替换动作并重建joint Adam，v不刷新。无best-ever/line-search/重启/noise action、无outcome选plan。最终最后一次更新动作先durable冻结，再从原task根独立执行一次完整H2验证并输出全部10commands。

明示的本地变体：H1→H3；无界normalized action→bounded logits/tanh/raw/normalizer；zero normalized→zero raw初始化；global noise RNG→local generator；显式Adam foreach=False/fused=False；新增最终真实H3模型verification。保留官方软动力学/虚状态/noise/sync机制，但这些差异禁止“官方逐bit等价”称谓。

## 资源、封存和失败

每根：750×2 lifted=1500；15×10×2 serial sync=300；独立verification2；共 **1802 predict +1802 action_encoder，900 scalar backward**。两根总3604各、1800 backward。predict/action_encoder均处理3个context tokens，单根各5406tokens。模型encoder/projector0、新physics0；历史blind acquisition单列，不是全pipeline免费。无Jacobian/SVD；scalar backward不等于192行VJP。**不匹配D2P66、不声称等总算力/时间/显存**。

prepare/verify绑定本producer/tests/doc、既有封存blind/GN ABI闭包、官方资产、torch Adam/optimizer源码、固定两份export和模型SHA；不依赖正在编辑的新stem。prepare只小manifest，不复制checkpoint。run MAIN-only显式cuda:0、MATH SDP、TF32关闭、模型eval参数冻结，原checkpoint tensorhash前后一致才释放。梯度只优化虚状态/动作，不训练模型；无CPU神经fallback。CLI无可修改超参，不能从新outcome追调。

新Store独立容量64MiB、200MiB reserve、256KiB final预留，启动需264MiB空闲；不能修改旧Store全局凑新容量。每transition先fsync请求（完整H3输入/history/raw），实际module attempt/return另计；返回fsync完整context。joint保存loss分量、虚状态/动作梯度、unprojected/projected logits、noise及更新v；sync同样保存梯度/动作与reset。gzip每event Z_SYNC_FLUSH/fsync，complete需要完整footer。超cap/floor保留partial失败，不扩容重试。

两根完整分母预置，失败保留首错/secondary、module未完成attempt及已封存现场；未开始根不替补、不挑成功根。final model/source/journal/hash/report尽力分别保存。硬kill或I/O失效不能保证最终报告，已有durable journal只作partial，不能假称完成。

consume是producer的消费接口（不是新增独立auditor）：要求外部actual report SHA，重建prepare/全部artifacts；核两根身份、实际module/backward预算、每phase两包及完整H3/normalization/action chronology、fixed schedule、final freeze和own-Q。它不重新执行模型、证明真实控制效果或替代MAIN的物理准入。

## MAIN命令与物理接线

cwd `/home/gbk/code/LeWorldModel/le-wm-main`；所有输出必须不存在。作者不执行下列实际run。

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_grasp_baseline_20261027_v1 prepare --output /home/gbk/code/LeWorldModel/le-wm-main/results/p1_constructive_grasp_baseline_20261027_v1_prepare_main01.json
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_grasp_baseline_20261027_v1 verify --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/p1_constructive_grasp_baseline_20261027_v1_prepare_main01.json --prepare-sha "$MAIN_PREPARE_SHA256"
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_grasp_baseline_20261027_v1 run --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/p1_constructive_grasp_baseline_20261027_v1_prepare_main01.json --prepare-sha "$MAIN_PREPARE_SHA256" --output /home/gbk/code/LeWorldModel/le-wm-main/results/p1_constructive_grasp_baseline_20261027_v1_cuda_main01
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_grasp_baseline_20261027_v1 consume --output /home/gbk/code/LeWorldModel/le-wm-main/results/p1_constructive_grasp_baseline_20261027_v1_cuda_main01 --report-sha "$MAIN_REPORT_SHA256"
```

MAIN在consume完整成功后固定新selection manifest，仅一个arm、两根；既有physical consumer支持显式schema/status和pointer，无需改变sealed物理代码或读取证书给控制器。root_slot映射由MAIN evaluator维护，planner只知道source_slot。

| physical selection字段 | 本actual report映射（i=0/1） |
| --- | --- |
| compiler_report/Compiler SHA | 新actual report绝对路径/外部MAIN SHA |
| compiler_schema | `p1_constructive_grasp_baseline_20261027_v1.report` |
| compiler_status | `grasp_model_plans_sealed_not_admitted` |
| source_slot_pointer | `/rows/i/source_slot` |
| arm_pointer | `/rows/i/arm`（固定GRASP_H3_bounded） |
| commands_pointer | `/rows/i/commands`（完整10×2 raw） |
| blind_sha_pointer | `/rows/i/blind_sha256` |

selection_digest必须沿用physical/source的默认JSON空格序列化SHA，不是本producer compact canonical。sourcepack/root/goal权限只在MAIN evaluator侧，禁止将certificate witness变候选。物理必须新输出逐步执行本plan；本脚本没有reset或physics入口，不继承旧endpoint。RGB另算实际资源。

## 测试与当前证据

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_constructive_grasp_baseline_20261027_v1.py
```

解析fixture是非torch.nn.Module的显式加法/乘法，没有神经权重或CPU神经forward；RAM gzip无大tmp。覆盖官方SHA/Git blob、软公式/stopgrad梯度反例、full-chain sync、真实H3和normalized history、local RNG/noise、project/tanh链、joint reset/全预算/失败和消费篡改。

本轮实际离线验证 **40 passed in105.85s**。其中完整默认750 joint/15sync×10、两根producer→consume均在RAM执行：3604次解析transition、1800次scalar backward，无Module/模型。RAM全产物28,466,697bytes（约27.15MiB），不保证真实模型压缩率；实际超64MiB仍须保留partial失败，不放宽gate。只读build通过，固定两份真实export、122个bindings、prepare预计24,854bytes；本轮未写prepare/真实results、未运行CUDA/物理。

当前最小缺口不是官方小源码：它们已存在且可绑定。剩余是MAIN review/freeze后实际CUDA的兼容性、显存/资源/数值验证，随后新selection→两条自己的物理10步→新RGB；公开normalizer的checkpoint训练谱系仍未认证，论文结果commit/checkpoint/配置未闭合，真实global freshness未知。解析tests通过不证明这些已通过，更不证明机制/效能或P1/P2准入。

experiment-code主SKILL与其实际位于 `/home/gbk/.codex/skills/experiment-code/references/` 的两份refs已完整读取。技能用于预声明单run、固定种子、日志与故障验证；按用户限制不执行训练/真实run/图表或“非零准确率”模板，不以负结果修改方法。
