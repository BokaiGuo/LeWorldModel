# Common-resource 六分支同时间 native 标量汇总

仅新source/tests/本说明。旧runtime、physical、model summary与结果均不改。固定两known off roots×三方法，共6分支；每分支cp5/10/15共18记录，每根每cp完整joint/composition、joint/GRASP、composition/GRASP三pair，共18pair。无NN/physics/网络/拟合，无winner，不读物理outcome选择配置。P1/P2/mechanism/fresh/等FLOPs均false。

## 认证与独立算术边界

`build_summary(output, report_sha, receipt, receipt_sha)`要求实际physical目录、实际report SHA与完整persisted `physical.consume` return的路径/SHA；没有未来默认SHA。先检查冻结physical三SHA，再执行其**完整saved consume**，全JSON精确比较外部receipt；status-only、wrapper envelope、指标或资源变化均拒绝。导入strict是原执行/源闭包/数组/动作/PNG认证，**不冒称独立动力学验证**。该consume也会重建原planner saved strict，不调用模型。

summary的数值仅从认证`responses[step].state`和原source native goal独立float64重算，绝不调用producer.metrics或读取producer的final_metrics/reward/metric_breakdown来当计算结果。逐根三方法绑定原task路径/SHA、source slot/root、native goal/pose、model hash、causal context、latent goal、warm raw10；不同方法的实际动作可不同，不为凑公平而改动作。每条保存原plan digest、请求动作前step段字节SHA、实际返回数、原state JSON pointer及同时间PNG身份。

## 指标与缺失

安装原生 `.venv/lib/python3.12/site-packages/stable_worldmodel/envs/pusht/env.py:347`的eval_state：joint XY前4维L2<20，原角度项`min(abs(dangle),2*pi-abs(dangle))<pi/9`；保留其可能负值，不“修复”为wrapped成功。`:360`的_get_obs顺序为agent_x/y、block_x/y、block_angle_mod2pi、**agent_vx/vy**。父physical strict认证其源闭包。本模块不import环境。

原native7距离/reward（含未wrap角度及agent velocity的混单位原公式）与agent XY、block XY、joint XY、raw angle、velocity、wrapped-angle诊断各自分栏；平方分解只解释原native7，不把pixel/radian/velocity相加发明新评分。不把agent惯性变化当block机制。wrapped_success仅诊断，绝不代换原native_success。

收益=同一原root误差减**同一请求时间**误差；pair delta=`b-a`、error_reduction delta也明确方向。native_angle_term不是全局单调误差，保留负角度标志，不将其负值解读成改进。只对相同cp、相同方法pair给两根known sum/mean和known/missing计数，不混cp、不宣称18个独立样本、不做显著性推断。

native earlystop、missing export、engineering failure保全六分母：未达到请求步为null，无last-endpoint回填/补步。若实际labelled return存在而该步render失败，scalar仍可用，RGB单独missing；若无该步labelled return不从其它state推测。baseline_reference来自原认证source，另列baseline_physically_verified，不把未实际重建的root参考冒充新执行。实际终止/truncate及error、所有资源attempt/return、planner资源与physical资源分开保留。没有单一“总成本”。

## MAIN命令

cwd `/home/gbk/code/LeWorldModel/le-wm-main`。所有输入变量均取本次MAIN真实输出，不能换绑旧物理。worker只运行fixture测试，不生产consume。

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p2_common_resource_physical_summary_20261029_v1.py

.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_common_resource_physical_summary_20261029_v1 build \
  --physical-output "$P2_COMMON_PHYSICAL_OUTPUT" --report-sha "$P2_COMMON_PHYSICAL_REPORT_SHA" \
  --strict-receipt "$P2_COMMON_PHYSICAL_RECEIPT" --strict-sha "$P2_COMMON_PHYSICAL_RECEIPT_SHA" \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_common_resource_physical_summary_20261029_v1_MAIN01.json

.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_common_resource_physical_summary_20261029_v1 verify \
  --summary /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_common_resource_physical_summary_20261029_v1_MAIN01.json \
  --summary-sha "$P2_COMMON_PHYSICAL_SUMMARY_SHA"
```

只exclusive写本stem≤16MiB JSON、floor200MiB；原physical完整128MiB输入/日志/PNG上限沿用其strict，不放宽reader/门槛。verify重新执行完整strict和算术重建，不只自证checksum。无真实report/完整receipt或来源SHA不匹配即拒绝。RAM fixture不是物理/模型科学证据；experiment-code技能仅用于结构化保存和回归，不执行训练/调参/绘图建议。
