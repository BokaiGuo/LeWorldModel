# 固定40槽：latent排序与原生任务指标分栏

本stem只新增source/tests/innerdoc和独占saved输出。没有等价的完整40槽分析：现有rank-error只有native_success，`moving_overlay_task_metric_bridge_20261027_v1`是旧8臂协议，不能移植其分母；MAIN physical audit已验证40个实际末返回，复用其真实authority但补齐当前缺少的latent→物理目标成对分解。无新神经/物理/网络/拟合/选根，不修改旧封存或共享文档。父goal不覆盖，工程交付不是P1/C2科学通过。

## 输入authority与末返回

固定rank SHA `e23e60588fa9766b690e605022e089656c4c80ad185853773e23a942b5e0228f`，重新完整rank.build/native arrays strict；physical report `464656755bb83cec14b7acdd457641592b11be6d335f285166687df15368bc01`与partial receipt `44acd71c60c8edc7fbfa196c2f46f7d8692ef487ec0c3a01b6b8f7e5071e9212`重新调用原physical.consume并要求完整相等。所有原JSON、实际artifact census、prepare/source/model闭包绑定，不仅信任success字符串。40槽按原4task×2role×5word固定顺序，不择winner或新阈值。

每槽要求20个真实raw_return，使用suffix5（绝对20）末返回，不拿prefix/最后可用PNG冒terminal。读取末返回info NPZ中完整block_pose、goal_state7、goal_pose3、agent position/velocity；逐项核snapshot/state/last_return、task/job/goal合同。index12仍保留partial/RGB失败，但其真实suffix5 state/info已返回，所以物体pose/原native任务指标可报告；actual latent/Q仍null。缺图不能抹掉真实末返回，也不能补图。

## 不同成功合同不能混淆

当前绑定的`stable_worldmodel/envs/pusht/env.py`以原`eval_state`判定成功：agentXY+objectXY联合L2<20，且原`min(abs(angle delta),2pi-abs(angle delta))<pi/9`，不检查velocity；reward=-完整state7距离。保留原角项可能为负的事实，不偷偷修公式。本报告独立重算并与真实返回success/distance/reward严格一致。

原任务goal为返回goal_state；默认outline `goal_pose`只是renderer外观合同，**不等于构造的taskgoal**。报告实际完整unwrapped block pose、task goal物体XY/angle及XY差/误差、wrapped angle差，另列物体对outline的诊断，不混成加权指标。agent、object、allXY、velocity分别报告，不将object-only投影非正当完整任务失败。

**coverage缺项：**本次原native step/info及全部journal没有返回coverage/final_coverage；绑定wrapper源码虽有旧success_threshold/coverage_arr属性，但step实际按上述whole-body eval_state计算，不能据此伪造coverage。`native_coverage`与`object_only_coverage`均null，来源状态明确unknown。没有另造shapely/新物理评分替代原native authority，不拿新几何定义或阈值冒原成功。coverage相关80组内对、20ZS对全部unknown而非0/失败。这是精确未决依赖：如需coverage，须有该次sealed轨迹可合法匹配的原coverage authority；当前不存在，不能通过新生成伪补。

## 两条成对误差轴

严格复用原matching：同root/condition/task/goal/role的80组内对，另列20个同word Z/S对（prefix/context不同，不宣称隔离因果）。每对分别列：

- predicted latent Q→actual latent Q：精确delta/sign与反转；涉及index12的4组内对及1ZS对unknown。
- actual latent Q→各原生物理指标：各指标独立delta/sign分类。距离/误差lower-better，success和coverage higher-better仅符号取向，不构造跨指标分数。tie不算反转；coverage未知不参与已知分母。完整分母、available、unknown、discordant、ties均显式。

不由latent负结果推断C2失败；latent→physical discordance仅是已知两根上的描述性目标失配，不识别动力学/接触等因果。P1=false、C2_failure_inferred=false，known roots仍2不是40，无physical benefit结论。

## saved-only CLI

从`/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p1_teacher_library40_task_metric_20261028_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.agent_p1_teacher_library40_task_metric_20261028_v1 analyze --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p1_teacher_library40_task_metric_20261028_v1_analysis_MAIN01.json
.venv/bin/python -B -m experiments.spectral_reachability.agent_p1_teacher_library40_task_metric_20261028_v1 verify --analysis /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p1_teacher_library40_task_metric_20261028_v1_analysis_MAIN01.json --analysis-sha <实际输出SHA>
```

输出严格新建不覆盖，verify重建完整报告；RAM测试不是科学证据。最终冻结后仅MAIN审读，不原地改闭包。

本次实际工程验证：17个新RAM用例通过，与原rank-error/端点语义consumer联合64项通过。真实saved analyze已完整消费40槽末返回，预测→真实latent的80组内对为76可用/4未知/12反转；真实latent→allXY有4失配、→objectXY有22失配（均76可用/4未知；objectXY另13个tie）。task_success在76个可用对中全tie；两类coverage均80unknown，不是0coverage。这些是已知两根上的描述性分栏，不是C2失败或P1准入。独占持久化报告使用`analysis_NASH01.json`，MAIN可另建`analysis_MAIN01.json`独立重算。
