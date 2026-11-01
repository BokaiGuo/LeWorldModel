# Extended moving baseline RGB：固定双臂实现交付

独占新source/tests/innerdoc；旧RGB与overlay全部冻结不动。只运行RAM mock，不生成
prepare/result，不运行神经、物理或actual saved build。MAIN负责审读、真实运行和科学结论。
按experiment-code固定合同、日志、故障测试实现，不训练/Qfit/按结果调整。

## 确定接线与封闭状态

已只读读取Meitner当前实际源码 `p1_moving_extended_baseline_physical_20261027_v1.py`，
不是猜想已完成的physical报告。provider报告schema为该stem的`.report`，成功记录status
`moving_extended_baseline_physical_recorded`；prepare为`.prepare`，manifest为`.selection`。
新增plan.root_sha256、row.branch_index/selected_plan，actualledger另列environment_construct/
environment_close；全部显式核验，不套用旧baseline成功标签。

MAIN确认且本实现只读hash验证过Meitner最终source SHA：
`a0f7348ebe8c6e061b5d63a1ba4a8ecb064a26e7f7a1ef4be57fe8b07270d799`。
`PHYSICAL_SOURCE_SHA`已固化，任意其它source SHA、未知authority、缺失authority均封闭。
RAM测试只patch模拟provider输入，不产生假的prepare或实际report。
physical tests/doc实存SHA也已只读核对MAIN给定的91587a9a... / 73ae46ab...；
actual prepare依赖闭包由physical封存bindings提供并逐项重hash，不以mock代替实际报告。
当前provider自身对白名单authority在import前核源码SHA，再调用其真正ownconsume；
GRASP/no-composition分别为 `GRASP_H3_bounded` / `joint_H2_spectral_packet_mapped`。
未知scheduler/authority不通过；未来只能在已发布、明确冻结的provider/ownconsume合同接线后支持，
不能把任意module路径、普通GD或自报commands当成授权扩展。

原sourcepack仍是c2d8d573...，只hash来源/evaluator证书依赖，不从中选择动作或读取witness候选。
原moving task00/01出口及真实SHA、load_task、native192、原goal/rootlatent/Q完全保留。
prepare显式physical report/prepare/source SHA，逐authority ownconsume，完整重建每条实际plan
（含root_sha256、own_plan_sha256、selection_digest），再要求完整等于physical prepare。
RGB arms必须等于事前physical manifest全量同序；不可运行后选子集。

## 预算

2固定known moving roots；外层预算最多4 arms，每臂每根prefix5/terminal10，共4A张。
Meitner当前收紧为固定双臂有序GRASP/no-composition，因此本轮实际仅接受完整两臂，
4分支、8图、8encoder/8projector；单臂、其它旧baseline、未知scheduler都拒绝。
未来provider若另行批准新合同，仍须显式新的源码认证，不能绕过当前双臂白名单。
最多16encoder+16projector，predict/actionencoder/AD/physics全部0，不重编码root/goal。
16MiB完整输出cap，200MiB floor，512KiB最终报告预留，启动需216MiB；编码wall300秒。
原planner/physical/源编码成本仍分别计账，不声称全算力等价。四臂以上manifest整个拒绝，
不能静默截断；扩展预算必须新合同。

每张保存原PNG逐字副本 `image_XX.rgb.png`（file SHA、pixel SHA、uint8[224,224,3]），
以及原native float32[192] NPY。最大原始像素量16*224*224*3=2,408,448 bytes，
PNG与NPY/日志/报告均纳入16MiB硬cap，不是仅估算cap。原physical完整PNG、journal/hashchain、
guard、actual operations、构造/close计数、raw10commands、完整moving history、root+goal和
native terminal来源检查失败均封闭，缺5/10不以early-stop替代、不补图、不筛根。

编码CUDA-only、MATH SDPA、TF32off、eval/inference/frozen模型hash前后检查。
实际encoder/projector模块meter与禁止predict/AD guard沿用冻结RGB算术基础；无CPU神经fallback。
回调只收到真实RGB ndarray；state/证书不进入编码选择或控制器。

## metrics / 保存合同

actual_Q=mean((actual192-goal192)^2)，root_Q、actual_Q_change、actual_vs_root_mse；
rawpixel_mean、rawpixel_vs_prefix5_mse（float64逐像素均方差），terminal相对prefix的实际Q变化。
prefix对自身差为0，terminal只和同root/arm/selection的真实prefix配对。
未知模型预测不补算，predicted_Q/MSE保持null，不把未知当0。
PNG先保存与raw_image_saved日志后才做神经dispatch；模块错误也保留真实输入PNG。
全4A图行预置，部分失败保留attempted/未开始行、真实module attempted/completed与首错；不重试。
run完成status `moving_extended_baseline_rgb_encoded_not_admitted`，不代表P1/P2或新独立样本。

## CLI（MAIN未来执行）

cwd `/home/gbk/code/LeWorldModel/le-wm-main`。以下只列MAIN真实运行所需接口，
尚未提供实际physical报告；不是已执行命令或伪造SHA。source参数使用上述固定完整SHA：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_moving_extended_baseline_rgb_cuda_20261027_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.p1_moving_extended_baseline_rgb_cuda_20261027_v1 prepare --physical-report /ABS/ACTUAL_EXTENDED_PHYSICAL/report.json --physical-sha ACTUAL_REPORT_SHA --physical-prepare-sha ACTUAL_PREPARE_SHA --physical-source-sha MAIN_CONFIRMED_SOURCE_SHA --arms GRASP_H3_bounded joint_H2_spectral_packet_mapped --output results/p1_moving_extended_baseline_rgb_cuda_20261027_v1_prepare01.json
.venv/bin/python -B -m experiments.spectral_reachability.p1_moving_extended_baseline_rgb_cuda_20261027_v1 verify --prepared /ABS/NEW_RGB_PREPARE --prepare-sha ACTUAL_RGB_PREPARE_SHA
.venv/bin/python -B -m experiments.spectral_reachability.p1_moving_extended_baseline_rgb_cuda_20261027_v1 run --prepared /ABS/NEW_RGB_PREPARE --prepare-sha ACTUAL_RGB_PREPARE_SHA --output /ABS/NEW_RGB_OUTPUT
```

纯saved入口build/verify，数组入口assemble/compare；没有旧producer execute或隐式全局猴补。
当前无send_input工具，未假称与Meitner直接发送消息；MAIN转达schema，最后对freeze接口核验。

## 本次验证边界

23项RAM测试通过：完整mock build/provider ownconsume接线、外部prepare绑定拒绝、
完整双臂/生命周期/源prepare身份、commands/goal/history/guard/缺5或10的封闭，
lossless PNG与NPY、独立mean192Q、模块失败前保留原图、硬cap/floor与未知authority拒绝。
没有创建prepare/results，没有执行actual saved build、神经或物理。
最多16图仅为外层存储上界；当前真实授权严格8图，不是可任意扩臂的接口。
MAIN须在真实physical terminal后提供report/prepare外部SHA，方能prepare/run。
knownroot校准评价不是P1准入；不根据native outcome挑臂，不预设理论结论。
