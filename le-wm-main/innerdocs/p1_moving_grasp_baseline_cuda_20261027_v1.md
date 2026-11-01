# Moving2 GRASP_H3_bounded adapter v1

仅新增本stem source/test/innerdoc，不改gradient或其他sealed三文件/结果/共享文档。MAIN独占审计、准入、实际HOST CUDA/物理及独立复算。实现方仅RAM/non-Module解析算术测试，不prepare或执行模型/物理。固定现存moving两根，不增加独立样本、不选root、不训练/Q拟合；P1=false。

## 最小变更与严格范围

完整复用sealed `p1_constructive_grasp_baseline_20261027_v1` 的Config、Dynamics、optimize、choice、journal_check和Store，核心源码SHA `ae9d2dd59bf613bc10c81858cb767dad1d92dcf9b7b7d103bedb018e0bd1fa28`。不重写优化或换算法。仅新增moving loader/来源封存、own prepare/report/model_plan/consume和实际native输入/API对账。复用已冻结moving gradient的saved-only绑定与Limits，不调用其优化器；该helper源SHA `3224d6a094e74b3845c2c695c6c85fe29c6522cd4a88c91c57f2cc03d127a70e`。不改module globals、不调用原zero-only GRASP build/base.load_export，不冒用旧schema。

输入严格为原moving encoder自己`load_task`的两合法非零history任务。封存原task00 SHA `a42bfbb1ea692782355e2572a5df2cd048b72cf3e6c41fee40eb12503674959b`、task01 SHA `101add21a7a40591167a6ab8e8b22c3c62cb1fd700c741fd36cd2073e2d5df74`，encoder report SHA `e7fb554327b3bb5a96190f6291f8450b2b0723cd981817715cfb7e50aa266bf6`。sourcepack SHA `c2d8d573f94884ffc081163f25f2bfa269f98ffe8155887704109219910ffec6`只做文件SHA provenance，不解析witness/hidden state/native outcome，也不读planner或physical结果。原task goal/Q/normalizer不变。未来zero logits初始化不等于用zero替代实际rawhistory；后者严格是float32五次[.1,0]+五次[0,.1]。逐包raw NumPy float32 tanh、原public normalize和投影语义不变，不clip/重复缩放。训练normalizer谱系仍unknown。

## H3并不是未来15步

H3是原生context的3个历史latent（observed0/5/10），不是未来3包。原核心字段明确`horizon=2`：未来想象2包×5=10步，输出10条raw command；并非想象15步、实际截断10步。virtual中间状态对应第一未来包终点。新脚本完全沿用此合同，physical执行=0，后续若MAIN接物理应明确执行这10条，不继承旧endpoint。不得与其他horizon/合同的reported成绩直接比较。

## 原算法与固定预算

原Config：750 joint Adam；lr_s=.1、lr_a=.005、state_noise=.005；sync_interval50、sync_steps10、sync_lr=.05。local device generator seed=20261027+101*slot。virtual初始为(z10+goal)/2，未来logits零初始化。soft dynamics+每edge goal项、所有lifted state输入stopgrad、目标侧virtual梯度和第一动作token的live history梯度保持原样；每次Adam后只有virtual Gaussian noise，动作逐包logit球投影。每50步fresh action-only serial fullH2 Adam10，随后替换动作、reset joint Adam，不刷新virtual。无重启、早停、best-ever/outcome选择、额外proposal或学习。最终最后动作先durable冻结，再一次fresh serial H2验证，不新增其他重放规则。

每root：750×2=1500 lifted预测、15×10×2=300 sync预测、2次verification，共1802 predict/action_encoder各、900 scalar backward；两root3604各/1800 backward。每root各5406context tokens；predict meter为predictor.forward，另记录model.predict API，二者逐一对应而不相加为新预测。backward是实际scalar loss.backward的meter attempt/return，不乘192、不另计VJP。new encoder/projector/physics/reset/render均0；原8encoder/8projector成本继承单列，不称全pipeline免费。无Jacobian/SVD，新优化变量仅virtual/action，不训练frozen模型。

原GRASP_H3_bounded本身就是history-preserving bounded-coordinate机制扩展：官方H1/DINO、动作坐标/初始化/RNG/数值等不完全相同，`official_equivalence=false`、paper结果commit绑定unknown；既有8个官方小文本SHA/Git blob与MIT notice、原torch Adam源码均纳入prepare。不是完整官方复现，不声称与D2P或原zero-task实验等总算力/跨合同reported比较。

## 工程与失败合同

沿用原64MiB总run输出、200MiB floor、256KiB final reserve；prepare≤256KiB，源closure只存path/SHA不复制大文件。新增工程cap：wall600s、allocated3221225472B、reserved4294967296B，首次Limits就是实际output.parent disk_usage；触cap保留partial，不减750步/预算或扩cap补结果。实际硬kill/I/O失效无法保证final report，已有durable journal仍不能准入。

CUDA:0 only，与原encoder host GPU名/torch/CUDA版本绑定；原192模型hash前后一致、eval/frozen，MATH SDP、TF32 off，autocast disabled。每transition完整H3/动作history/raw请求及actual module attempt/return和返回context逐次fsync；新增actual predictor输入latents/tokens与model.predict API attempt/return，全部1802次/根核对lifted、sync与verification。保存每joint的loss分量/梯度/noise/投影前后动作/virtual、sync梯度与reset。失败保留首错、secondary、固定两root not_started/failed_partial，不替根、不retry。own consumer只接受本schema且完整两根、预算、源模型/出口SHA、实际输入与原journal closure；不是独立审计agent或科学准入证明。

## MAIN命令（实现方不执行）

cwd `/home/gbk/code/LeWorldModel/le-wm-main`，所有输出必须NEW；PREPARE_SHA及ACTUAL_REPORT_SHA替换为MAIN实际生成的SHA。

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_moving_grasp_baseline_cuda_20261027_v1.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_moving_grasp_baseline_cuda_20261027_v1 prepare --task-exports results/p1_constructive_moving_history_encode_cuda_20261027_v1_cuda_main01/task_00.json.gz results/p1_constructive_moving_history_encode_cuda_20261027_v1_cuda_main01/task_01.json.gz --task-shas a42bfbb1ea692782355e2572a5df2cd048b72cf3e6c41fee40eb12503674959b 101add21a7a40591167a6ab8e8b22c3c62cb1fd700c741fd36cd2073e2d5df74 --loader-sha 246b9194fd31c874d7f0da67c4dec17365f93294e6ed460fe46e2509673e1b63 --encoder-report results/p1_constructive_moving_history_encode_cuda_20261027_v1_cuda_main01/report.json --encoder-sha e7fb554327b3bb5a96190f6291f8450b2b0723cd981817715cfb7e50aa266bf6 --sourcepack results/p1_constructive_moving_history_source_20261027_v1_main01/sourcepack.json --source-sha c2d8d573f94884ffc081163f25f2bfa269f98ffe8155887704109219910ffec6 --output results/p1_moving_grasp_baseline_cuda_20261027_v1_prepare01.json
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_moving_grasp_baseline_cuda_20261027_v1 verify --prepared results/p1_moving_grasp_baseline_cuda_20261027_v1_prepare01.json --prepare-sha PREPARE_SHA
timeout 650s env PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_moving_grasp_baseline_cuda_20261027_v1 run --prepared results/p1_moving_grasp_baseline_cuda_20261027_v1_prepare01.json --prepare-sha PREPARE_SHA --output results/p1_moving_grasp_baseline_cuda_20261027_v1_cuda_main01
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_moving_grasp_baseline_cuda_20261027_v1 consume --output results/p1_moving_grasp_baseline_cuda_20261027_v1_cuda_main01 --report-sha ACTUAL_REPORT_SHA
```

新plan须MAIN未来另行明确接线物理consumer，不因旧arm同名就自动admit旧runtime。实现测试不证明实际CUDA/物理效果、机制收益或P1通过。experiment-code技能仅用于固定协议/durable日志/失败测试，不执行通用训练/调参/图表模板。

## 实现方验证（2026-10-27）

上述禁pyc/cache命令：26 passed in149.40s。完整默认750 joint、15×10 sync、两个moving非零history root的producer→own consumer均在RAM走通，3604次显式非Module算术transition、1800次scalar backward；不是CPU模型forward。全产物RAM writer按64MiB总量及final reserve逐写检查，没有tmp/result文件。覆盖原参数/预算/公式alias不变、lifted state stopgrad与live动作history、原非零context、sync/reset/终点验证、actual输入与API计量、篡改/partial/首错保留、资源cap、CPU模型入口拒绝和旧zero loader隔离。

实际saved-only `build(...)` RAM smoke：2slots、139项源/模型/官方/optimizer/出口绑定、canonical prepare预计30,280B；该独立进程未导入torch、未写prepare/result。三文件冻结后由MAIN审读与实际prepare/CUDA验证；没有实际运行结果或科学准入结论。本次goal完成仅指实现交付，P1仍false。
