# Moving固定两根六基线 saved assembler v3

实现交付，不是独立审计agent；MAIN负责实际综合、独立复算与科学准入。本worker仅RAM fixtures和CLI help，不执行实际assemble、prepare、神经、物理、训练或网络。使用experiment-code固定输入/预算/失败记录合同；按用户范围不生成训练、图表或额外文件。v1/v2及所有producer/results保持sealed不变。个人goal仅实现交付，非P1完成。

## 固定输入与六基线

固定原moving slot0/1，非独立确认。原encoder report SHA `e7fb554327b3bb5a96190f6291f8450b2b0723cd981817715cfb7e50aa266bf6`，sourcepack authority `c2d8d573f94884ffc081163f25f2bfa269f98ffe8155887704109219910ffec6`；原task SHA分别 `a42bfbb1ea692782355e2572a5df2cd048b72cf3e6c41fee40eb12503674959b`、`101add21a7a40591167a6ab8e8b22c3c62cb1fd700c741fd36cd2073e2d5df74`。原非零history及task goal/Q不变，不读取后续outcome决定纳入算法/根，不混旧zero或target-on。

六基线顺序：moving_gradient_open_loop、CEM、kernel_faithful_mppi_nominal、fixed_H2_kinodynamic_beam16x4、GRASP_H3_bounded、joint_H2_spectral_packet_mapped。前四复用冻结v2的ownconsume与多cohort接线，原三臂6分支/12图和kino2分支/4图保持独立账本，不重计原三臂。若MAIN事先在existing.cohort纳入原GN/D2P两方法，仍须v2严格legacy证明，此时总8算法/16模型行；不是按结果新增/删减方法。六个baseline本身始终预声明。

新增两个provider独立白名单和自有schema，不修改v1.PROVIDERS/v2 globals，不冒用旧zero loader。仅接受各自已声明actual invocation SHA：

| provider | source SHA | report SHA |
|---|---|---|
| grasp | 52d3157574e4e2726f6d88b43bd7e5dc804ce79730b65512888e63e0dd7c7106 | 7593940c1503416f42ce87a5d8135e8512763121e027c0ef98dbe614bfb31f91 |
| no_composition | 8c709bed26d88e093b0293482be07cfc34702fb095b2fe23a22edadbbb985963 | ec39956ca0c90cc5dbf955b1a65bf300b68d166627a748a82314a2176a481c34 |

report路径分别是Python根下 `results/p1_moving_grasp_baseline_cuda_20261027_v1_cuda_main01/report.json` 和 `results/p1_moving_no_composition_baseline_cuda_20261027_v1_cuda_main01/report.json`。实际模型证据存在不等于物理成功；本轮后续extended物理已完成，作为单独cohort接入，不按结果改模型选择。

## 实际ownconsume接线

complete模型source绑定report与原样ownconsume JSON绝对路径/外部SHA，先验证冻结source/module身份，再执行其纯saved `consume(output,report_sha)` 与JSON全内容相等；不调用prepare/run/optimizer/model。新增两consume均返回rows，不把SRA report原本没有的arm字段强加到输入。GRASP返回含whole_run_report_sha256的plan；SRA同样包含该字段，但其plan没有selection_uses_physics，不能伪造补键。各自sealed ownconsumer已核所有历史/候选/选择source closure；v3再核task/export/blind/contract、10×2 literal commands、原goal与prediction native192终态MSE、完整两根及预算。GRASP额外有原predict_API receipts；不能与native predictor模块计数相加。

新source state可complete/partial/missing，固定两个provider必须显式保留且不得重复或逆序。partial仅展示原reported attempts/returns，不调用完整ownconsume，不发布modelQ/完成预算；缺失或只完成一根不缩小分母。表中null保留missing/partial/unmeasured与原因，不转0。run绑定manifest、自身三文件、v1/v2源、task/loader、各producer source以及新provider prepare的全部bindings_sha256（含模型/原始输入），并合并既有physical/RGB saved闭包，输出前再次核SHA。

## 真实预算、计算域和成本边界

以下是两根合计；资源互不折算、不声称等总算力：

| 基线 | predict/AE | ADforward子集（已在predict内） | 反向/J/SVD/solve |
|---|---:|---:|---|
| gradient | 272 | 128 | 64 scalar backward；无J/SVD |
| CEM | 268 | 0 | 无AD/J/SVD |
| nominal MPPI | 144 | 0 | 无AD/J/SVD |
| kino | 168 | 0 | 无AD/J/SVD；pred_projector另meter168 |
| GRASP bounded | 3604 | 3600 | 1800 scalar loss.backward；无J/SVD |
| 无组合谱SRA | 176 | 4 | 2J、384 scalar VJP方向、4SVD、2bounded solve |

GRASP每根1500 lifted+300 serial sync+2 fresh verification=1802预测；H3指3历史帧，未来是H2/10步，不是15步截断。其标量loss backward不等价192输出Jacobian方向，不把900×192虚增成本。该bounded moving扩展不是完整官方GRASP复现，reported异协议成绩不作为本地对照。

SRA每根nominal2+AD2+40候选×2+whole2+segmented2=88预测；原40槽完整fullH2非线性终态择优，没有按H1挑选再补零。192×20 J为完整两packet图；joint谱bank后独立packet logit球投影改变原joint支持域，并非精确双约束ridge。原未归一化ridge damping .002，mean192等价 .002/192，不能声称mean目标damping .002。其vjp_directions是实际meter方向数；没有独立grad API hook/backward hook则保留unmeasured，不能伪报0或和Jacobian重复相加。完整CPU bank/投影/评分、ownconsume重算的SVD/solve工作保留在原contract，属saved验证开销而非新planner神经成本。

kino每根16首段+beam4×16后段=80 planning，另4 replay=84；全库枚举不是自适应预算优势。constant-packet离散bank与D2P逐时刻logitball不同域。nominal MPPI本地非官方。forward count不是FLOPs/总算力，ADforward不重复加，backward/grad/VJP/Jacobian可能描述同一导数工作。

共享继承8encoder/8projector只记一次；每算法standalone视图各8/8，不相加成共享账。实际评价原三臂12、kino4单列；新增extended两臂RGB若完整，另记8encoder/projector，未提供则evaluation_encoding仍missing。model run、physical run、RGB run分别记账；latency保留真实invocation scope，GRASP有limits.elapsed_seconds，SRA未单独记录则unmeasured，不均摊或推估。JSON保留每根、summary、完整合同/计数、来源及不等价警告，MD简表不排名。

## 下游合同与当前确切缺口

v3的existing是完整v2 manifest对象，不读取一个未经自身接线的comparison.json就当成认证输入。MAIN已实际完成v2四算法综合：comparison.json SHA `a0fb977445b79d72ebb062a78c454debabfa0e6560eb0f5957e7a8676f85424c`，8model/8physical/16RGB、0/8native；这里只记录MAIN交接，不将此文字写成新实际综合结论。可将MAIN原manifest原样复制到v3.existing，不改其seal或重复物理。

模板已接实际kino RGB report `results/p1_moving_baseline_rgb_cuda_20261027_v1_kino_cuda_main01/report.json` SHA `71dafeeebd1d2bd2567028e65800f4f27f479c72b066da3d2ace6a35ef8e7869`、prepareSHA `12ad9280e68625e9f8a9850a72fa1e769fe3f1014bd541b1934d0dcfe592767e`。actualQ从完整4个NPY重算，不硬编码用户消息数值。

**在v3冻结前MAIN提供了实际extended physical及新RGB adapter合同，现按独立4分支cohort接入。** 不使用/放宽旧physical/RGB白名单。物理source锁 `a0f7348ebe8c6e061b5d63a1ba4a8ecb064a26e7f7a1ef4be57fe8b07270d799`；report `results/p1_moving_extended_baseline_physical_20261027_v1_main01/report.json` SHA `f05ce7e2d773bb046c6c8ba905c9e86c367c7e19ad09174d4509e1823a297038`；prepare `9819cd6857e043492d27a239728fb3eee6d6ed5542546796467bf0ae83425fd7`。MAIN报告4full10/0native、88PNG/80public/816physics；另外environment_construct/close各4计入完整ledger。模板直接填写这条实际链；这是物理真实0，不迁移给缺失证据。

复用新 `p1_moving_extended_baseline_rgb_cuda_20261027_v1` 的纯saved build/verify/compare，源码锁 `da384d180254858238a50c0ecc602313204947060041b45597d43e6195f4ef9a`（交付前只读取得当前源SHA，MAIN须与Jason最终freeze一致）。build在未编码时也可验证actual physical全部journal/PNG/原非零历史/goal/原commands、source_prepare_id、selected_plan/root_sha256和生命周期计数；不调用prepare/execute，不修改旧authority。模型ownconsume完整plan默认spaces digest与physical own_plan_sha256、compiler_report_sha256双绑定，再经v2纯attach_physical映射。新schema/status由新adapter验证，不能伪装成旧report。

若RGB已actual，verify重建其原saved prepare，核整个模型前后SHA/contract/inputs/4分支8图分母。读取全部8个float32[192] NPY及8个原PNG副本，逐artifact SHA、严格18 artifact+report census、每图7事件（含raw_image_saved且发生在encoder/projector dispatch之前），复算native192 actualQ、同臂prefix5→terminal10原像素MSE、pixel mean及Q change。不能套旧6事件/无PNG副本协议，不能跨臂配prefix或用modelQ替actualQ。实际RGB尚未提供时不猜路径/SHA，保留missing。

manifest.new_downstream自有结构为 `{"arms":["GRASP_H3_bounded","joint_H2_spectral_packet_mapped"],"physical":{...},"rgb":{...}}`，固定两臂两根，不接受单臂或删根。physical complete含state/report/prepare_sha256/source_sha256；RGB complete含state/report/prepare_sha256；missing仅state/reason；partial为complete字段加reason但不调用完整saved验证，仅保留reported计数且不发布指标。RGB complete必须对应complete physical。没有提供actual或partial的指标保持strict null，不填0。以后别的authority仍须NEW明确合同，不能任意动态import扩展。

## MAIN manifest与命令

Python根 `/home/gbk/code/LeWorldModel/le-wm-main`。template只输出stdout，MAIN用apply_patch保存NEW manifest：顶层schema/purpose/existing/additional_sources/new_downstream。existing可原样采用 `results/p1_moving_baseline_comparison_20261027_v2_manifest_main01.json`（SHA `e2dcb6023a8c9f93c7e244febf5fa68bd2a99b1e7efb3aaf6c6d586a8f1a6a31`）；该manifest已有sampling/gradient/kino真实consume路径/SHA，不再重造。

将additional_sources两个missing条目改complete。以下为GRASP**结构示例**；consume尚无MAIN实际保存路径/SHA，显式占位，不可直接当真实值：

```json
{
  "provider": "grasp",
  "state": "complete",
  "arms": ["GRASP_H3_bounded"],
  "report": {
    "path": "/home/gbk/code/LeWorldModel/le-wm-main/results/p1_moving_grasp_baseline_cuda_20261027_v1_cuda_main01/report.json",
    "sha256": "7593940c1503416f42ce87a5d8135e8512763121e027c0ef98dbe614bfb31f91"
  },
  "consume": {"path": "<MAIN实际绝对JSON路径>", "sha256": "<MAIN实际64hexSHA>"}
}
```

SRA条目provider=`no_composition`、arms=`["joint_H2_spectral_packet_mapped"]`，report改本doc列出的实际路径/SHA。两consume只能保存各自CLI输出的原样严格JSON，不能把日志、wrapper或额外元数据混入。new_downstream.physical模板已填actual；RGB未actual保留missing，有actual则填其新adapter report/prepare完整SHA，不等RGB也可汇总物理+模型。MAIN可执行原producer的consume子命令生成两JSON；本worker不执行。

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_moving_baseline_comparison_20261027_v3 template
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_moving_baseline_comparison_20261027_v3.py
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_moving_baseline_comparison_20261027_v3 assemble --manifest /home/gbk/code/LeWorldModel/le-wm-main/results/p1_moving_baseline_comparison_20261027_v3_manifest_main01.json --manifest-sha <MAIN实际64hexSHA> --output /home/gbk/code/LeWorldModel/le-wm-main/results/p1_moving_baseline_comparison_20261027_v3_main01
```

NEW输出目录comparison.json/md合计≤1MiB，逐写保留200MiB，不覆盖旧结果。全模型齐全而下游仍缺时status为model_complete_downstream_partial_not_admitted；所有模型/物理/RGB齐全才为comparison_recorded_not_admitted；model_layer_complete单列。P1/P2、官方复现、等总算力/搜索域、独立确认均false；native=0只来自已实际执行并严密接线的各自物理cohort，不迁移到缺失臂。

验证：56项RAM fixtures通过，覆盖六模型/12行、原8物理/16RGB、独立extended4物理/8RGB全分母、全部预算与AD不重复计费、缺失/partial、schema/history任务错接、所有两根/commands/模型cost/预算/来源绑定、新adapter隔离、7事件/PNG/NPY数值与坏配对拒绝、pure consume loader隔离、prepare全闭包继承、v1/v2源SHA、cap及中文表。无tmp/result/实际综合写入；MAIN负责真实saved运行及复算。最终SHA见交付消息。
