# P3 joint H2/H3：NEW sixteen-branch physical consumer

仅消费 `p3_joint_horizon_goal_metric_scheduler_20261027_v1` 的**真实完整**四根×四臂CUDA report，不能拿旧endpoint、旧八臂consumer输出或conditional P1结果代替。planner prepare SHA固定为 `7886406284604cf17ba97671030ef32fde9d32f6f4c18b8293db0c2129654a47`，source/test/doc保持原封存，不修改。

本次仅NEW module/tests/innerdoc、NumPy/小PNG/接口fake tests。不执行环境构造、真实神经、AD、CUDA、physics或训练。实际planner report已只读校验，consumer prepare及执行由main统一完成；不重跑规划。所有P admission false。

## 身份、动作与上限

四臂为 `old_Q__uniform / old_Q__dynamic / goal_patch_Q__uniform / goal_patch_Q__dynamic`：**两种Q×两种schedule**，不是四个不同Q。共16独立fresh physical branches，各1次；每臂完整H2/H3 winner对应10/15 raw controls，selected-path最多240 steps，实际budget为封存16个H的 `sum(5H)`（160..240），不padding、不裁剪、不补H、不重新选winner。root-search/reset/warmup为另行真实工作，不包含在240 selected-step cap中，不称等全预算。

只读重建实际 `results/p3_joint_horizon_goal_metric_scheduler_20261027_v1_cuda_main01/report.json`（SHA `23eae72d18fa02e9997859311bef282fc4f5e3478790d846524a3fa6f6b70950`）得到16 plans：15个H2、1个H3，故本轮预期 **165 selected +240 warmup =405次公开env.step返回**，不是240次全部物理。collector按现有源码只reset、不step；实际collector和逐臂warmup reset均独立记录attempts/returns，不用理论数量替代运行计数。

prepare先调用sealed planner完整model-free exporter，重建own-Q响应/bank/argmin及H1/H2/H3 history算术，再按稳定 `(root_sha256,arm)` join，拒绝缺失/重复/错source/动作/目标/Q/metric id/上下文/horizon/report/export hash。明确消费planner `physical_ready=False` 的模型见证，由本NEW consumer建立物理来源绑定，不把planner状态翻成P1成功。

原四根reference只提供初始root/goal/RGB/history/evaluator合同，不提供terminal。严格重新生成同seed/oversample roots，四根state/goal fingerprint upfront匹配；每臂fresh reset/warmup，原始三帧RGB/goal、完整physical state和incoming normalized action history一致。四臂顺序沿planner seal的平衡顺序，不能按结果重排。

## 执行安全与保存

复用durable的 `guarded_execute`（非旧八臂执行器）：RoyalBlue upfront及每个raw command的RGB guard；每次env.step前写attempt，返回后先记return，再render/label。保存全initial/observation PNG、journal文件SHA、raw命令与step顺序。失败attempt不退费，terminated/truncated或guard失效立即停止该路径；执行错误阻断后续arms并补齐blocked census。

新consumer自身branch保存按阶段捕获reset/step/evaluator、journal hash和root JSON错误，不以后者覆盖已有failure。full-H必须所有5H命令返回且fresh endpoint；partial endpoint须最后一张durable RGB的primitive_step等于实际returns、attempts==returns且状态可观测。step或render失败的旧图不编码、不充当新endpoint。0-step upfront guard失败保留initial-only partial scope，不叫full-H成功。

`PhysicalMeter`在真实公开env.step/reset调用外逐次写attempt/return，分root_collection、warmup、selected；每臂warmup严格15，selected严格5H，collector step cap为0。通用总step上限480=240warmup+240selected，不是本轮405的预测替身，也不计私有模拟器substeps。失败attempt不退费，预算journal写失败另存错误并阻断，不把不完整receipt称为完整；最终selected计数还与durable branch独立对账。

全体physical rows先写 `physical_report.json`，随后main可显式 `--encode-terminals`。同一frozen weights、native encoder+existing projector，单图CUDA：各最多16次，predict禁止，不跑AD。每个fresh endpoint只编码一次，保存完整192原数组，再算old_Q/new_Q两种cost，own-Q由arm label标识；partial保留partial标签。编码/模型hash/source检查/hook恢复失败分别留存，不隐去physics receipts，不自动重试。

metadata/saved-array导出与源码重建的SVD属于继承验证成本（每次planner exporter为16 source-validation+32 response SVD），不是新神经或新的规划。实际encoder/projector attempts/completed单列。

`counted_encoding`在encoder/projector调用前保留attempt，成功才加completed；异常写含error的失败event。失败event本身写失败也保留首个编码异常及secondary journal_errors。完整编码门要求两者各16次attempt==completed且无journal_errors；部分执行仅报告已观察下界，不冒称完整。

## 分层与磁盘

每根两个Q-effect对比（分别uniform、dynamic）及两个schedule-effect（分别old_Q、goal_patch_Q），共16条paired描述。same-H2/same-H3、mixed-H2/H3、partial-same-steps、unmatched分开；只有相同reset、full且same-H时给pose差，mixed/partial不混作完整same-H效果。保存所有official/pose原指标，终点Q评分不反馈动作。没有跨根挑有利子集或任何P阶段准入。

prepare要求200MiB最终余量+32MiB协议预算，main execution起始要求200MiB+100MiB输出预留；每次guard render检查200MiB+16MiB暂存余量，拒绝时不发下一步。disk检查不放在proxy.step内部，避免未调真实env.step却记作物理attempt。并发外部写入不能由本线程保证，因此main仍需统一磁盘调度；绝不自动删除任何asset/seal。

建议consumer prepare额外预留约15–25MB、physics/endpoint输出预留100MiB（不是实测成功证据）。复用root_metrics表避免每臂重复保存192×192 Q；实际source closure/协议体积以prepare后du为准。没有planner实际report时，不创建真实prepare目录。

## main命令（真实report出现、review后）

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_p3_joint_horizon_goal_metric_physical_20261027_v1.py
.venv/bin/python -m experiments.spectral_reachability.p3_joint_horizon_goal_metric_physical_20261027_v1 --prepare --planner-output results/p3_joint_horizon_goal_metric_scheduler_20261027_v1_cuda_main01 --planner-report-sha256 23eae72d18fa02e9997859311bef282fc4f5e3478790d846524a3fa6f6b70950 --output results/p3_joint_horizon_goal_metric_physical_20261027_v1_prepared01
sha256sum results/p3_joint_horizon_goal_metric_physical_20261027_v1_prepared01/prepare.json
```

main独立verify receipt后自行执行；agent不运行：

```bash
.venv/bin/python -m experiments.spectral_reachability.p3_joint_horizon_goal_metric_physical_20261027_v1 --execute-physics --prepared results/p3_joint_horizon_goal_metric_physical_20261027_v1_prepared01 --prepare-sha256 ACTUAL_CONSUMER_PREPARE_SHA --output results/p3_joint_horizon_goal_metric_physical_20261027_v1_main01 --encode-terminals
```

全部测试为逻辑/文件/constant fake环境接口验证，不认证真实接触、CUDA数值或模型控制效果。

测试期间13项失败源于fake RGB为32×32而生产guard要求official uint8 224×224；修正fake输入尺寸，未弱化生产guard或断言。这是测试接口缺陷，不是物理实验或理论负结果。测试覆盖collector/warmup/selected实际调用分账、失败attempt保留、编码失败journal与partial endpoint，而非仅验证selected数量。
