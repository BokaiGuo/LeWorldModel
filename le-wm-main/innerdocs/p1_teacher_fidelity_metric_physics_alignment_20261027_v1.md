# Teacher latent-Q 与 saved physics 指标的描述性符号对齐

仅新增本 stem 的 source/tests/innerdoc；旧分解、live analysis v2、实验执行器及共享文档保持不动。
采用 experiment-code 的自包含验证与结果记录要求，但按用户范围不训练、不生成图、不新增执行器。
只 import 两个纯 saved-array 分析模块；无模型/神经/physics 执行、adapter、Q拟合或CPU神经实验。

## 固定 authority 与严格闭合

使用原分解 `results/p1_teacher_fidelity_error_decomposition_20261027_v1_saved_main01.json`：
SHA256 `483c40c33eb856c45b380e7d0892e555784236d8e7584c198b3881098c16a0e7`。
使用原 live analysis v2 `results/p1_moving_live_feedback_analysis_20261027_v2_main01.json`：
SHA256 `5df7dd90b3da010d1bafbb66198334272ec6470b7b0f7bc62991e4b0dae0b5d8`。

入口只接受新输出路径，不能在CLI临时换root/metric/cohort。
先核分解全部原输入文件SHA、实现源码SHA，并用纯saved分解器完整重建，要求报告所有原字段精确相等。
再核原analysis的全部input/source/tests/doc SHA，用其纯saved `build` 完整重建并要求整报告相等。
两侧共享数据绑定必须完全相同；原driver源码是原analysis单独冻结的authority，核固定DRIVER_SHA与原prepare绑定；
原analysis tests/doc以原analysis自己的封存SHA独立验证，不假定属于teacher runtime闭包。
若任一原analysis绑定、重算或配对不能闭合，立即失败，不写成功报告。

teacher prepare的原analysis/live report/live prepare路径与SHA必须与实际消费对象全部相等。
12个episode按 `(source_slot,prefix_arm,mode,episode_id)` 显式join，不依赖数组顺序；
逐个核task_id、selection_sha、encode5/10自身latent receipt、goal SHA、label/prefix数组SHA及actual_Q精确相等。
对四层全部I−O、R−I、R−O，必须原same-prefix配对成立，逐项delta与原analysis已有contrast精确相等。
任何缺失行、重复身份、错receipt、缺关键checkpoint/Q、错SHA或部分运行均fail-closed。
输出前再次重核所有输入SHA。结果保留原分解全部132个latent数组receipt，并列全7state/goal及其float64数组SHA。

## 指标与符号：不混合评分

固定12 episode＝2个source task×2个prefix arm×O/I/R；独立任务数始终2，不以12行或4层代替n。
每行列predicted endpoint Q、actual endpoint Q，以及step5/step10真实保存观测的latent Q。
Q是原生192维latent mean squared distance，仅描述latent cost，非物理效用。
每个物理endpoint保留全部7维state、goal、signed residual（agent XY、object XY、angle、agent velocity XY）。

各指标独立列出，不合成跨指标加权分数，不按有利指标挑root：

- `native_distance7`：完整7维norm，包含velocity；不能用XY或前5维冒充；
- `allxy_position_error4`：agent+object的前4维norm；`agent_position_error2`单独列agent XY；
- `object_position_error`：object XY norm；
- `wrapped_object_angle_error`：`abs(atan2(sin(dtheta),cos(dtheta)))`；
- `native_angle_term`：原legacy `min(abs(dtheta),2*pi-abs(dtheta))`，不暗修缺陷；若为负，保原值但方向比较为unknown；
- `native_success_recomputed`与`wrapped_allxy_success`：由保存state复算，不伪称新native receipt；
- `native_success_receipt`：原driver终点实际保存flag；step5没有该receipt，必须null；
- `object_success`：原协议无object-only success定义，始终保留null，不臆造阈值。

每层三种contrast、两个endpoint、每个指标都列lhs/rhs和`delta=lhs-rhs`，不round、不epsilon化。
距离类改善符号为`sign(-delta)`，success改善符号为`sign(delta)`；latent Q改善符号为`sign(-Qdelta)`。
`same_sign`直接比较改善符号；若任何一侧恰零，status=tie，strict_reversal=false，nonzero_agreement=null。
两个非零符号异号才strict_reversal=true；任一未知值或无有效方向则status=unknown，所有一致性判断为null。
predicted Q仅与step10比较；step5对应比较显式null，不能把suffix终点预测误当prefix预测。
同时保predicted-vs-actual latent-Q反转，和actual/predicted Q各自相对物理指标的方向；两种错配不混称。

## 命令与本次实际结果

cwd `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_teacher_fidelity_metric_physics_alignment_20261027_v1.py
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_teacher_fidelity_metric_physics_alignment_20261027_v1 --output results/p1_teacher_fidelity_metric_physics_alignment_20261027_v1_saved_main01.json
```

已实际生成上述完整描述性报告，SHA256
`fcf6d1f7f272cfcd3972886cc4b51214d0b2d72ab369318b8d0627cc3d97ad63`。
独占测试文件本次运行结果：`37 passed`，仅RAM mock与NumPy算术。
包含803个消费文件SHA、12个episode、12个配对contrast，各自两endpoint及全10指标；n=2。
旧报告存在时拒绝覆盖；复跑须提供另一个新路径。

以下仅展示全部四层R−O终点raw delta，完整O/I/R数值及I−O/R−I见JSON，不以此表代替全量报告：

| 层 | actual latent-Q | native7 | allXY4 | objectXY2 | wrapped angle | native success |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.08337655337607058 | 96.31166766772441 | 9.183377373553853 | -3.3641781683254877 | -0.19596919578135003 | 0 |
| 1 | -0.14321710547049737 | -30.378485182209715 | -36.60139035663909 | -7.711351655885283 | 0.059316304238295814 | 0 |
| 2 | -0.04219944671806286 | 13.929888510360968 | -11.434463466724623 | 0.9885288608083158 | 0.007916616648651065 | 0 |
| 3 | -0.06992661996426389 | -27.1466702572377 | -10.104048418945851 | -5.609283277765009 | 0.21775761189746223 | 0 |

例如层2的actual latent-Q改善，但完整native7/objectXY/wrapped-angle变差，不能只呈现allXY改善。
四层R−O native-success均tie而非反转；未知object-only success不算失败或成功。
这些均为已知两root的post-hoc描述，不能声称新控制收益、普遍对齐或独立确认；无显著性推断、P1/P4准入。
测试仅RAM mock/NumPy合同与算术，非科学证据；完整原报告重建与实际saved分析另行验证。
未来真实实验仍须MAIN审读后HOST CUDA，本交付未启动任何真实实验。
