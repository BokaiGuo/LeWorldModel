# T3 固定80对结构诊断：source-first

新增独占三件套：`experiments/spectral_reachability/agent_p1_t3_structure_diagnostic_20261028_v1.py`、同名 `tests/test_*.py` 和本说明。原 margin source/tests/doc 及所有 producer、结果、gate、阈值保持不变。本 worker 不执行生产 analyze，不启动 CUDA/neural/physics/网络，不拟合参数、不挑根。

## 输入与验证边界

唯一输入为 `results/agent_p1_pair_margin_decomposition_20261028_v1_MAIN01.json`，固定 SHA `f723c5eb73ecebd0615a2ded0aa4b741c6f132ca5f6063c673f12d5e98539143`。其完整上游闭包由 MAIN 原 analyze 和独立 full verify 验证；本 derivative **不重复重建该闭包**，仅核固定报告 bytes、其中三件冻结文件 manifest 与磁盘 SHA、原 roster、76/4、坐标/时间、精确恒等式、原 float64 分类。输出明确区分 `input_full_saved_native_rebuilt` 和 `upstream_closure_rebuilt_by_this_consumer=false`，不能把此工具独自当作新的原生闭包验收器。

保持固定8 anchor × 每5词全部10对，顺序 i<j，不跨根/角色/context；40槽、39编码、76可比和4未知。未知 `(10,12),(11,12),(12,13),(12,14)` 留在全80分母，不补图、不零填数学项。来自两已见根，不是80个独立样本。物理标签完全不参与计算或选择。

## 预先固定的分解与反事实

原 margin 为 `m=predicted_cost_i-predicted_cost_j`，原真实编码 margin `o=m+B+R+Q`：

- `B=common_bias_projection=2mean((rhat_i-rhat_j)*e_j)`；
- `R=relative_response_projection=2mean((rhat_i+e_j)*(e_i-e_j))`；
- `Q=relative_response_quadratic=mean((e_i-e_j)^2)>=0`。

此处 Q 是二次误差项简称，不是重拟合或替换原 metric Q。原 raw MSE 已除192一次，本工具不再归一化。完全使用原 binary32 精确有理数分解的 numerator/denominator，核 `o-m=B+R+Q` 和两个原 exact residual=0，无近零容差或 sign snapping。

固定列出全部8个 retained mask，bit0/1/2 对应 B/R/Q，1为保留、0为移除：0、B、R、BR、Q、BQ、RQ、BRQ。输出 mask 数字顺序0..7，每项有保留/移除名称、精确 correction、`m+sum(retained)`、符号、分类、严格跨越。没有最佳mask、按结果筛组合、最大改善或新 selector。

当 `m!=0`，每项朝跨越方向的贡献为 `progress_k=-sign(m)*term_k`，threshold=`abs(m)`。严格跨越当且仅当 retained progress 之和 **大于** threshold；等于为 observed tie，不是跨越。逐项另列 progress/threshold 的精确比值，它仅是尺度描述，不是拟合权重，也不是收益比例。`m=0` 无初始严格方向，threshold/progress/比例均null，分类保留 initial tie。

每项报告：

- 单独加入是否足以严格跨越；
- 完整三项已跨越时，移除此项是否不再严格跨越（相对于本固定分解的必要性）；
- 项朝跨越、反向或为零。

因此可区分一项足够、多项都必要的联合跨越、另一项抵消跨越等情况；必要性与单项充分性均不互斥，不输出唯一归因标签。三项相消另以精确 `L1=sum(abs(term))`、净 correction、正/负总量、`opposing=min(positive,negative)`、`cancelled_L1=L1-abs(net)=2*opposing` 报告；全零时 cancellation fraction=null，不除零。该量只针对总误差校正项之间的相消，不包含初始 m。

## 不可作的解释

共同参考严格沿原 i<j，`c=e_j`。B 并非从全部词估计出的共享偏置；改参考端会改变 B/R/Q 归因，而总 margin 恒等式不变。本工具不寻找更有利的参考。Q 对 i-j 恒为非负不意味着它总有害或总有利：要结合 m 的方向。R 含参考误差和相对误差的交互，移除 B/R/Q 不等同于物理上能独立干预该项。

尤其8组 pairwise 数学反事实不一定能由一组全局一致的40个 endpoint 修正共同实现；没有真实 encoder/model 可部署校正，没有新控制动作。观测编码用于回顾归因，不是部署时可得输入；单独项移除后的跨越变化不证明因果修复或真实任务提升，更不据物理标签选根。原 float64 margin/分类单独保留并报告与 exact 符号是否一致；反事实 exact 分类不取代原评分合同。

MAIN 已核原生 distance 的速度项可以改变 latent→native 排序；本工具仅解释预测→观测编码 T3，不碰 T4 的物理轴，也不据此宣布普遍 encoder-物理不对齐、解决原生成功或通过 P1。独立 freshness、非贪心能力机制和强基线统计门仍独立存在。数学完整性检查不是新 science gate。

## MAIN 审读后运行

工作目录 `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p1_t3_structure_diagnostic_20261028_v1.py

.venv/bin/python -B -m experiments.spectral_reachability.agent_p1_t3_structure_diagnostic_20261028_v1 analyze --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p1_t3_structure_diagnostic_20261028_v1_MAIN01.json

.venv/bin/python -B -m experiments.spectral_reachability.agent_p1_t3_structure_diagnostic_20261028_v1 verify --analysis /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p1_t3_structure_diagnostic_20261028_v1_MAIN01.json --analysis-sha <analyze返回的实际SHA>

.venv/bin/python -B -m experiments.spectral_reachability.agent_p1_t3_structure_diagnostic_20261028_v1 consume --analysis /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p1_t3_structure_diagnostic_20261028_v1_MAIN01.json --analysis-sha <analyze返回的实际SHA>
```

只依赖 Python 标准库。analyze 仅向自有stem新文件 exclusive `xb` 写入，cap4MiB，不覆盖；verify/consume从固定快照重新计算整个诊断并比较完整canonical bytes，不是只验自己报告hash。stdout仅简短回执，不复制旧报告和49k闭包条目。报告绑定本三件套SHA，MAIN生产运行后不再编辑。

## 测试与待办

遵循TDD：先测试缺模块失败，再实现。合成测试覆盖共偏置/相对响应/联合必要性、相消但不反转、正负初始方向、exact tie/初始tie、全部mask、全80和4unknown、物理标签不依赖、float64/exact分离、fraction规范/零分母、原SHA/冻结source变化拒绝、完整重建比较及独占写入。没有读取生产数组或执行原closure测试。

本轮新33项与冻结原分解47项联合运行：**80 passed in 4.96s**。原三件套没有修改。

待MAIN源码审读、正式saved analyze/verify/consume；本说明不预填本批项移除统计，不宣称新实验收益。
