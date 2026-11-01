# Commoncontroller native saved 数值比较 v1

仅独占新增同stem source/tests/doc；不改冻结runner、gate、threshold或产物。使用experiment-code的独立算术、日志和RAM回归部分；不训练/筛选/生产。source不import任何producer/driver，不调用physical.consume/fullstrict/NN/physics。MAIN已有完整own consumer仍是原生证据认证者，本工具**导入其外部SHA封存的完整receipt**并做数值join，不替代其journal/PNG/native全部验证。

## 实际来源与身份闭合

报告 `/home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_commoncontroller_physical_20261029_v1_MAIN01/report.json`，实际SHA `3164f9fb3808e6d579ae9aada8306427b0acbf9a657eda2f2e7e46f68f4f5076`。

完成receipt `/home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_commoncontroller_physical_20261029_v1_consume_MAIN01.json`，实际SHA `bf976c4f28f3b4634d2d78594779cc74933bf8870d78aef2db8cb182a49342b3`。CLI显式传参，worker不运行实际analyze/verify或producer consume。

source锁原runner SHA `7e715a9005a4408fbc7899184f338aa4ae6ee6b20dd414356744aa830ceff5f0`，原批准protocol SHA `42e082498484f8bb399f819d31531c2a77f73b161e0bd2e15bb8ec900f5e8de6`。读取report→prepare真实路径/SHA，核完整prepare六plans/authority/source/root/goal，完整receipt六rows与report逐JSON字节等同、全部counts、planner_authority、physical_report_sha256、bindings中的report/prepare/source/protocol均闭合，所有真实science flags严格false。

不只核status：固定六槽/2known/0fresh/0inherit、ACTIVE0/1/3/4、missing2/5、errors[]、每槽task/actionbit/终止/步数/checkpoint、所有native returned标量与六项metric_breakdown均独立核验。两missing无原生操作/PNG/终态，metric=null。metricbreakdown任意标量、身份、缺项或顺序变化拒绝。完整资源向量按四env原生命周期复核，不只比predict，unknown上游FLOPs保留null。

## 固定分析表与方向

六槽 × cp0/5/10/15/native_terminal，共30行；两个root × 相同五栏目，共10 prior/program配对，全部固定不挑根/指标/时刻。每行完整state7/residual7、原native metric、分轴距离及平方、原始和wrapped角、velocity、native_success与diagnostic wrapped_success。cp0只评价真实state，无public terminated返回时保null。

独立NumPy float64：full7原角差和velocity都进距离；XY分agent/object/joint，raw角平方、velocity平方另列。原native_angle_term照安装公式min(abs(angle_delta),2pi-abs(angle_delta))，可能负；不静默替换为wrapped角。native success仍原jointXY<20且nativeangle<pi/9，velocity不进入成功门。wrapped诊断单列，object_success原未知null不伪造。

**逐root逐cp及native终态完整平方分解**：每个非missing行的 `metrics.native7_squared_decomposition` 固定列 `native7_squared = joint_xy_squared + raw_angle_squared + velocity_squared`，三项、其和、恒等式残差与gamma16舍入界全部保存。每个prior/program配对的 `squared_margin_decomposition` 固定列Δnative7平方、ΔjointXY平方、Δraw角平方、Δvelocity平方及三差值和、残差/gamma32舍入界。这里只约束同一组float64数的加法重排误差，不是数据拟合epsilon或科学gate放宽。两个root×五栏目全表，不根据终态某项正负挑指标。

解释必须分开：native7距离是含velocity和原始角差的mixed-unit欧氏距离；位置项下降但velocity项上升时，整体native7仍可变坏。此为该root/时点上可直接量化的指标断点，不能据此称所有分支统一原因、一般encoder失配或所有物理任务都失败。原success不看velocity；wrapped角、原native min-angle、raw角平方三种不同量不能互换。各root完整表与跨时变化保留，不能只摘位置改善或只摘native受损。

所有差值定义 **program − prior**：distance/error负表示program较小，native7_reward/success正表示program较高；sign严格positive/negative/tie，无新epsilon/阈值。另列error_reduction=prior_error−program_error，以及各分支相对真实cp0的误差下降/reward gain。

native_terminal只用真实own最后返回，列各自实际step；提前终止cp15保持null，不把终点复制进cp15。终态配对即使步数不同也保留，但same_control_time=false且明确非同时间对照。没有own终态时保持null，不拿warmup末态或imagined latent代填。原rootwise cp0完整snapshot/state/PNG与task/history/goal须匹配，target不同是该对照的处理变量。

不做CI、选择器、聚合挑选、timefair/fresh/P2pass结论。prior FIRST来源物理时间未知、program target来自原cp5，不由共同controller的native full15变成同目标时间/同生成预算实验。两known roots仍是两个known roots。

## MAIN CLI与边界

所有输出排他，≤16MiB，floor200MiB；输入JSON上限128MiB，拒绝重复键/非有限数/软链；源/输入在结果生成后再次核SHA。只保存新summary，不写旧receipt。verify按原参数整份重建saved summary（不重建原物理fullstrict）。

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q tests/test_agent_p2_commoncontroller_physical_comparison_20261029_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_commoncontroller_physical_comparison_20261029_v1 analyze \
  --report /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_commoncontroller_physical_20261029_v1_MAIN01/report.json \
  --report-sha 3164f9fb3808e6d579ae9aada8306427b0acbf9a657eda2f2e7e46f68f4f5076 \
  --receipt /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_commoncontroller_physical_20261029_v1_consume_MAIN01.json \
  --receipt-sha bf976c4f28f3b4634d2d78594779cc74933bf8870d78aef2db8cb182a49342b3 \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_commoncontroller_physical_comparison_20261029_v1_MAIN01.json
# SUMMARY_SHA填实际analyze返回值。
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_commoncontroller_physical_comparison_20261029_v1 verify \
  --summary /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_commoncontroller_physical_comparison_20261029_v1_MAIN01.json \
  --summary-sha "$SUMMARY_SHA"
```

worker仅RAM/fixture测试和真实report/receipt字段布局只读核对，未生产分析。测试覆盖全30/10分母、earlystop不同terminal步数/nullcp15、receipt/breakdown/绑定/资源/标量篡改、signedzero动作、完整snapshot非state-only、fixture CLI analyze→verify与排他写；无producer math import、无NN/环境/原生fullstrict。最终回归：上述pytest命令30 passed / 0.66s，exit0。三件套交付后冻结；实际analyze与整份verify由MAIN执行。
