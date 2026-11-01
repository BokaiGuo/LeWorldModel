# fixed5：saved预算与任务结果比较

独占源码/测试/doc均为本stem。只读分析已封存两根×fixed/adaptive四臂，不重新规划、调配额/阈值或挑结果。不提供模型/物理/网络入口。真实CLI由MAIN审读运行；本地仅RAM及已保存元数据算术测试，不写MAIN结果。

## 身份与消费者

固定CUDA report SHA `8067a743e3893b120ac8d57582c8552cb5b1dec182f7ca4bd0c6f70ffacaeeb5`，physical v2 report SHA `5fdfc8cfea7f6b4fe78d3e19b59fe946aad73a537850c52a86f2d395055b5e45`。先核外部SHA及冻结`agent_p3_fixed_execution_saved_20261028_v1`三SHA，再完整调用它的consume；该入口经v2.verify/material完整消费CUDA、原source/tasks/inputs/PNG/journal。利用其返回的嵌套producer_consume，不再重复调用同一CUDA consumer。CUDA原报告per_branch/prefixes/winners须与严格consumer相同；CUDA prepare、两份明确signal report必须在完整bindings及prepare sources中SHA一致。最终再验全闭包漂移。

严格保留固定顺序(slot0 fixed/adaptive、slot1 fixed/adaptive)、source/context/binding/goal/request digest/raw5。未知/失败行仍在四分母；无可信endpoint则指标None，full-denominator mean不以0补齐。已知根仅2，不是4独立根。

## 资源口径

从原resource_receipt.events按明确slot/mode及operation分组，调用原core.audit_ledger重新核shape→delta、attempt/return、API/sample/J/VJP与inclusive wall。signal原账、search、selected_prefix_verification、new_model、total分别保存；只有同一axis可跨阶段相加。正常每臂实际forward API/sample为signal8+search24+prefix1=33，独立Jacobian API2/VJP384。fixed12个H1候选、adaptive4个H3候选都是原配额；不能改成最小H1或把完整H3规划当执行15步。物理最多请求5步，按实际return与原生早停保留，不padding。

replay/metric调用数减少只属于这些计数轴，不折算FLOPs或每次调用相同work。API、module hook、samples、context tokens、J、VJP、raw backward callback各保留独立单位，nested axes不合计成“总调用数”。输出每轴adaptive−fixed及主forward/AD轴是否有计数减少；compute_advantage仍False。backward-engine work、FLOPs、CUDA time未知不能填0。

CUDA report wall（含run内加载/验证等）、new ledger wall、inclusive operation ns、两份signal panel whole wall、physical execute-only wall分别记录。不把signal panel wall按两根重复计入、不将nested时间加成总wall；无exclusive per-branch/per-method whole-job wall，不能声称端到端加速。保留原saved SVD验证成本及历史acquisition原schema账，单列不混入本轮controlled预算、不假装历史免费。

## 任务口径及C3边界

native7 distance/reward包含agent速度和原始角差；jointXY是前4维联合欧氏误差，objectXY只取block位置2维，agentXY、native angle、wrapped angle及velocity分开。native_success严格继承jointXY<20且原native angle<pi/9，object_success保持None（无另立阈值）。模型winner full-H Q按其H1/H3不同终点保存，不当共同物理成功；prefix Q只是模型诊断。每根及每mode输出全分母误差/成功和配对差，负error delta仅表示这两根某项误差较小，不等于native成功或泛化。

无新child observation/context、同child付费信号、child replanning与完整闭环树；固定配额不是cost-minimal H1；FLOPs和独占whole-job wall缺失；P1/适用P2/P3、fresh independent cohort未准入。所有科学/compute/walltime/freshness门False；代码明确给缺口，不由测试、prepare或consumer成功升级C3。

## MAIN绝对路径CLI

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
/home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python -B -m pytest -q -p no:cacheprovider /home/gbk/code/LeWorldModel/le-wm-main/tests/agent_p3_cost_comparison_20261028_v1_test.py

/home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python -B -m experiments.spectral_reachability.agent_p3_cost_comparison_20261028_v1 analyze \
  --cuda-output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_fixed_execution_cuda_20261028_v1_cuda_MAIN01 \
  --cuda-report-sha 8067a743e3893b120ac8d57582c8552cb5b1dec182f7ca4bd0c6f70ffacaeeb5 \
  --physical-output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_fixed_execution_physical_20261028_v2_MAIN01 \
  --physical-report-sha 5fdfc8cfea7f6b4fe78d3e19b59fe946aad73a537850c52a86f2d395055b5e45 \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_cost_comparison_20261028_v1_MAIN01.json
```

output可省略则仅stdout；指定时仅允许results内本stem全新文件（排他xb），存在即在消费前拒绝，输出≤8MiB。不覆盖原产物。tests以已SHA绑定的MAIN严格receipt模拟消费者返回边界；不冒充本实现者已运行真实完整CLI。MAIN仍负责全闭包实际运行和独立审计。
