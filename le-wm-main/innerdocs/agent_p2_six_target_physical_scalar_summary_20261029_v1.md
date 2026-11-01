# 六槽物理独立标量比较器 v1

仅saved数值consumer，不调用native runner/fullstrict、不导入producer数学、不运行NN/physics。原six physical三件套与Pauli RGB均不改。本文件定义固定分析，真实report/receipt由MAIN实际传入；RAM fixture不是实验结果。

## 协议与ABI

固定slot0..5：root0/off的prior_FIRST、program_first_postcondition、arbitrary，随后root1同序。读取physical own prepare/report/consume三份外部SHA绑定文件，严格核schema、完整六rows/plan/task/root/goal/raw bits、receipt bindings、source/protocol pins、完整metric_breakdown、资源向量。不是只核status，也不重新消费journal或重验PNG，完整native真实性依赖MAIN传入的原fullstrict receipt。无自签科学门，known2/fresh0。

原report execution_binding.prepared/prepare_sha256指向显式prepare；strict.physical_report_sha256/bindings核原report。所有CLI路径必须canonical绝对路径，尤其output/summary，不因cwd隐式接受relative。真实输入尚未提供不生成ready/未来hash；不把fake fixture当actual authority。

固定6×cp0/5/10/15共24位置；每根三对(prior,program)/(prior,arbitrary)/(program,arbitrary)×四cp共24pair。原native早停保null，缺cp15不填terminal；单列六native_terminal及其真实步号，不作异时terminal排名。每根cp0核全snapshot/state/file+pixelSHA控制，plan的task/blind/history/原goal逐项同源，PNG文件名可以不同。

独立float64按原native7顺序agentXY/blockXY/raw-angle/agentvelocity计算：native7_squared=joint_xy_squared+raw_angle_squared+agent_velocity_squared，jointXY另拆agent/block。native距离为七维欧氏范数，reward为负距离；native success仍jointXY<20且min(abs(delta_angle),2pi-abs(delta_angle))<pi/9，保留其多圈可能负值行为；wrapped abs(atan2(sin,cos))与wrapped success仅诊断不替换。所有实际返回native scalar、final、receipt metric_breakdown逐项核验。平方分解两种float64归约的残差/rounding_bound单列，非科学门放宽。

pair数值差统一right−left，原始正/负/tie分栏；误差负值较小、reward/success正值较大，不把native min-angle当通用好坏指标。每cp相对cp0的error reduction与reward gain另列，全部指标不选取。全资源counts原样输出并校验六env/12reset及实际public/physics/render等；planner_authority完整保留（unknown不补0），不宣称equalFLOPs/timefair/P1/P2 admission。

## MAIN CLI

在`/home/gbk/code/LeWorldModel/le-wm-main`运行，P/R/C/O必须实际绝对路径，PH/RH/CH为对应真实SHA，不使用未来占位证书。已有prepare不要重建。

```bash
OPENBLAS_NUM_THREADS=1 /home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python -B -m pytest -q -p no:cacheprovider /home/gbk/code/LeWorldModel/le-wm-main/tests/test_agent_p2_six_target_physical_scalar_summary_20261029_v1.py
MOD=experiments.spectral_reachability.agent_p2_six_target_physical_scalar_summary_20261029_v1
P=/home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_six_target_physical_20261029_v1_prepare_MAIN01.json
PH=e695bdd3f7f134801414b34d63f5afee0615528423f0360eed7b92b8eb760675
R=/home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_six_target_physical_20261029_v1_MAIN01/report.json
RH=ddbb339f87e0f49b1d99d32ae4f41340d72e6c19f8199fe69e6d0261f78bca2b
C=/home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_six_target_physical_20261029_v1_consume_MAIN01.json
CH=7579693032046b2427a7d5b492a90131029fe87d39147d06fe49123a3cfead13
O=/home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_six_target_physical_scalar_summary_20261029_v1_MAIN01.json
/home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python -B -m "$MOD" analyze --prepared "$P" --prepare-sha "$PH" --report "$R" --report-sha "$RH" --strict "$C" --strict-sha "$CH" --output "$O"
# OH为上步真实summary SHA，O文件名须以本stem加下划线开头。
/home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python -B -m "$MOD" verify --summary "$O" --summary-sha "$OH"
```

真实接线（CLI传参而非代码硬填结果）：P为`/home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_six_target_physical_20261029_v1_prepare_MAIN01.json`，PH=`e695bdd3f7f134801414b34d63f5afee0615528423f0360eed7b92b8eb760675`；R为`/home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_six_target_physical_20261029_v1_MAIN01/report.json`，RH=`ddbb339f87e0f49b1d99d32ae4f41340d72e6c19f8199fe69e6d0261f78bca2b`；C为`/home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_six_target_physical_20261029_v1_consume_MAIN01.json`，CH=`7579693032046b2427a7d5b492a90131029fe87d39147d06fe49123a3cfead13`。worker只核文件SHA及真实schema字段形状，没有调用真实analyze/verify/fullstrict。实际snapshot含state/state_float64_bits/bodies；response含native_metrics，receipt另有六项metric_breakdown；它们不是伪造旧四schema。

写新summary限16MiB、floor200MiB、排他写入/fsync，不覆盖原产物；verify整份重算比较所有输出与来源SHA。首次RAM回归27 passed /0.41s；加入真实保存ABI纯字段形状回归后，最终28 passed /0.46s，exit0。三件套至此冻结；未运行真实analyze/fullstrict/NN/physics。待MAIN独立审读、回归与实际saved analyze/verify；没有剩余代码占位或未来authority假设。
