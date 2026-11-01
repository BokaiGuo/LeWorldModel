# Moving 固定两根多cohort结果 assembler v2

仅实现交付，MAIN负责真实assemble、审读、独立复算与科学准入；本worker仅RAM fixtures和CLI help，无实际综合、prepare、模型、物理、训练或网络。v1及所有producer/source/results不变。使用experiment-code的实验合同与结果记录要求；按本任务限定仅三个新文件，不生成训练/图像或额外审计支线。

## 冻结合同

默认三臂顺序：`moving_gradient_open_loop`、`CEM`、`kernel_faithful_mppi_nominal`。每臂原moving slot0、1，不依据结果筛根。原三臂physical/RGB是一个固定6分支/12图cohort。可预声明添加独立kino cohort，顺序为上述三臂后追加 `fixed_H2_kinodynamic_beam16x4`，kino自己固定2分支/4图；不重放原三臂，不将其120 public steps或12编码再次相加。可再追加原 `line_search_GN`、`full10_conditional_D2P` 两臂，但必须同时通过旧moving专用saved adapter的严格source/task/原commands证明，不能只凭同名臂纳入。没有GRASP依赖。

固定sourcepack `c2d8d573f94884ffc081163f25f2bfa269f98ffe8155887704109219910ffec6`；原encoder report `e7fb554327b3bb5a96190f6291f8450b2b0723cd981817715cfb7e50aa266bf6`，8encoder/8projector。复用冻结v1 public loader验证原非零raw history和action normalization，两task SHA固定：

- slot0 `a42bfbb1ea692782355e2572a5df2cd048b72cf3e6c41fee40eb12503674959b`
- slot1 `101add21a7a40591167a6ab8e8b22c3c62cb1fd700c741fd36cd2073e2d5df74`

拒绝旧zero任务、target-on、换goal、换model selection。每source须显式 `complete/partial/missing`。complete绑定actual report及原样ownconsume JSON的绝对路径和外部SHA；调用已有纯saved ownconsume与保存JSON全内容比对。不能用wrapper虚构consume字段。partial保留原reported counters/status但不发布modelQ或完整预算；missing绝不变0。未完成两根不缩小分母算均值。

## 实际下游接口，不猜新schema

复用Jason冻结 `p1_moving_baseline_rgb_cuda_20261027_v1.build/verify/compare` 的纯saved部分。build从physical execution_binding、prepare/selection、journal全链和PNG重建所有固定分支/前后端点，验证全部原literal commands、动态历史、goal/source/task；不调用physical execute或prepare。RGB verify读取其原prepared并重建同一saved绑定，不调用encoder。随后读取全部native float32[192] NPY，复算actualQ；每图核完整6事件（image attempt、encoder attempt/return、projector attempt/return、saved receipt），report artifact SHA与严格文件census。source SHA锁 `bd174c810c22579862fd2707e24de58ed9fafd45327543a1c1f7da24397f0865`。

physical自己的default-spaces JSON digest不是v1 compact/newline digest：own_plan_sha256严格用 `sha256(json.dumps(plan,sort_keys=True,allow_nan=False).encode())`。与选中ownconsume完整plan（含commands）及compiler_report_sha256双绑定。旧GN/D2P则使用其专有planner_selection_sha256和planner_sha，并锁其adapter源码。外部source/model/prepare/task/test/doc及实际下游bindings最终再核SHA。

模板已固定原三臂实际输入：

| 输入 | report路径（相对Python根） | report SHA |
|---|---|---|
| 原三臂physical | results/p1_moving_baseline_physical_20261027_v1_main01/report.json | 6a9f4cf602a328cf5cb0fbe82296fc989a109a6859287c2efb703bebf44511db |
| 原三臂RGB | results/p1_moving_baseline_rgb_cuda_20261027_v1_cuda_main01/report.json | ab549f1d1a75c88b64a1c8c8697b004f8e6bf84d603b6e350406cf439fa305cd |

对应prepare SHA分别 `2e7304f11420ed65b6f3f039dd49522fee042e30bf6e97c82db791addd3fa50e`、`24a20378230ce64052f81c72cd91755309e426f7fa8bf2cd5f390761c826a0f0`。physical源码 `19b367debdfcc47032de8828dd0ab7124171c553d6c3f7b0a225ca86820cc3e0`。MAIN已报告6/6 full10、0/6 native；assembler只从绑定actual receipts计算，不把该文字当数据。

## 多cohort manifest

先从template stdout取得自有manifest（MAIN用apply_patch保存NEW JSON）。`sources`里sampling和gradient默认missing：MAIN必须填实际report及原样ownconsume JSON的完整外部SHA，才可和complete physical连接。sampling source arms按其provider顺序 `CEM,kernel_faithful_mppi_nominal`；展示顺序与provider原顺序可不同。

添加kino时必须同时：顶层 `cohort` 追加 `fixed_H2_kinodynamic_beam16x4`；`sources`追加provider=`kino`及同名arms的模型来源；`additional_cohorts`追加下例。不是把kino塞入原三臂physical，也不是重计原6分支。每cohort必须原两根完整分母，无single-root或重复臂入口。模型report已知路径 `results/p1_moving_kinodynamic_baseline_cuda_20261027_v1_cuda_main01/report.json`，SHA `8d2a8d6bc71eac53194c6001647c2d90736f292f9dfdcf5a2c838a58d36d2507`；consume路径/SHA由MAIN填真实保存值。

以下additional_cohorts元素可直接使用已完成kino physical；RGB尚待MAIN提供实际封存值，不猜路径/数值：

```json
{
  "arms": ["fixed_H2_kinodynamic_beam16x4"],
  "physical": {
    "state": "complete",
    "report": {
      "path": "/home/gbk/code/LeWorldModel/le-wm-main/results/p1_moving_baseline_physical_20261027_v1_kino_main01/report.json",
      "sha256": "2db7370214bf62810c21ad8778c1d58bc397dbe8f08c9f5ef1db435468abc655"
    },
    "prepare_sha256": "4f03963d6f3b5915072cecf3464d7c94cdf9d58ac76f9bfc8b0ecae717b21fbf",
    "source_sha256": "19b367debdfcc47032de8828dd0ab7124171c553d6c3f7b0a225ca86820cc3e0"
  },
  "rgb": {"state": "missing", "reason": "MAIN独立kino RGB尚未提供封存report/prepare SHA"}
}
```

kino selection SHA `a64d36fdfa33a7e96d362d675036df6826d4f40d9654e4be338a61a4fb7cda1d`，由physical execution_binding/prepare间接严密绑定，无需额外伪造顶层selection字段。MAIN报告kino2/2 full10、0/2 native、40 public/408 physics/44 explicit renders。RGB后续改成 `state:complete,report:{path,sha256},prepare_sha256`。若物理或RGB partial，使用相应complete字段加reason和state=partial；partial不跑完整saved verifier，不把未发布数据当actual。缺失物理时RGB不得complete。顶层legacy默认missing，不追加旧两臂即可。

## 数值、资源、时间与输出边界

每行分别保留modelQ、native success布尔、full10、actual root/prefix5/terminal10 Q及来源。完整10步不等于native成功，actualQ改善也不等于成功。实际native=false才记0；missing/partial保持value:null+reason。actual root Q来自原observed task编码，仅在物理原root RGB精确匹配后放行；其后两个Q来自各自实际NPY，不用modelQ代填。

模型资源继承v1：每根gradient136predict含64ADforward，32 reverse sweeps；CEM134，nominalMPPI72；kino84含80 planning和4 replay。kino两根168predict/actionencoder/predprojector，不把80全库搜索说自适应优势。GN/D2P可选时保留原Jacobian/grad API、数学反向方向数、SVD，未meter字段unknown；ADforward是predict子集，不再相加。算法完整信息合同、动作域、bank/seeds/选择规则保留。nominal MPPI本地非官方；kino离散constant-packet域与D2P逐时刻logitball域不同；不equalcompute、不equal搜索域、不SOTA排名。

共享原8encoder/8projector只记一次；每算法standalone视图各列继承8/8（不能相加），actual评价每两根另4encoder/4projector。原cohort12、kino4分别计账，不复用/去重评价图。物理reset/set_state/render/public/内部physics全账与模型账分列。latency保留原invocation范围，不除根数/臂数；RGB未独立meter walltime保持unmeasured而非0。跨cohort展示不代表同总算力。

写NEW输出目录 `comparison.json`（严格JSON）和 `comparison.md`（中文表），合计≤1MiB，每次写前留200MiB。不覆盖旧目录，所有输入封存失配fail closed。没有任何生产模型/物理调用，也不改变已有outcome/选择。known moving2，非独立确认，P1/P2仍false；不是理论失败结论。

## MAIN命令与实现验证

在 `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_moving_baseline_comparison_20261027_v2 template
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_moving_baseline_comparison_20261027_v2.py
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_moving_baseline_comparison_20261027_v2 assemble --manifest /home/gbk/code/LeWorldModel/le-wm-main/results/p1_moving_baseline_comparison_20261027_v2_manifest_main01.json --manifest-sha <MAIN实际manifest64hexSHA> --output /home/gbk/code/LeWorldModel/le-wm-main/results/p1_moving_baseline_comparison_20261027_v2_main01
```

40项RAM测试覆盖三臂actual/missing/partial、模型/物理/RGB逐字段错接拒绝、NPY metrics及事件账、旧GN/D2P特殊合同、多cohort四算法8行/16图、原6分支不重复、kino缺失不阻塞原三臂、两根不可筛选、重复/重叠cohort拒绝、固定payload cap及v1源SHA。测试不生成tmp/results。实际saved综合仅MAIN执行，交付SHA见消息。
