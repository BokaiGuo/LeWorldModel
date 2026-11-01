# P2 resolved controls comparison v1

纯 saved consumer，不启动神经、物理、网络或训练。不修改 frozen resolver、v3 或任何旧产物。使用 experiment-code 技能组织真实保存证据、RAM mutation tests 和排他回执；不采用其通用图表/必须非零结果建议。

## 固定输入与执行边界

输入为 resolver saved02，SHA `baa16c6bb9661be94c897077a48824278148c66efc01e5981e69af824b198685`；v3 inventory_MAIN01，SHA `0683c5d90afb00680f2bcfe45d0fc6bf8309d8c82a4c9448b26960069ebe289f`。源码常量给出绝对路径。consumer 先检查 frozen resolver/v3 源码 SHA，再重建两份输入逐值相等；按 v3.ASSETS 逐一核完整固定2/2/4行、顺序、root/arm、原report绑定。通过 v3.evidence_value 逐项重读 saved02 catalog 的 pointer/SHA；不伪造完整 v3 prepare 请求。

输出三个资产对、每对全部十项控制的 match/mismatch/unknown、逐task差异位置、全部原root/arm集合、v3 evidence catalog、原missing清单与最小补证请求。算法名、family、arm名称不是控制字段；arm仍完整保留在分母。未接收或比较任何作者性能报告值，单列 author_reported_values=[]；不输出本地性能排名或效果delta。

## 比较语义

按完整 task_identity 的 task_id 集合建立跨来源根对齐，不将本地 slot0/1 当全局根。只有同task全集闭合才比较各组件；不同全集的 task/root 为 mismatch，其余字段因无共同闭包而 unknown，不悄悄取交集。task顺序可变但逐task值必须一致；重复task的不同arm保留，若同task控制互相冲突则拒绝。unknown不填0、不当mismatch或match。

目前 beam vs no-composition 恢复九项 metadata match，resource_axes unknown。两个 sampling 跨资产对十项均 unknown；sampling四行完整保留。此结果不是公平实验认证：

- checkpoint match仅为权重/配置身份声明一致，原 loader loaded bytes 尚未认证。
- action_support match仅为域/normalizer合同一致，不等于候选覆盖、搜索分布或AD/FLOPs一致。
- resource_axes 未闭合；actual counts 不变成 scoped limit/inherited budget。
- source/task身份不构成独立根或freshness证明。
- eval_metric match来自task的投影/metric字段，不外推到物理native成功等价或校准有效性。

scientific_admission、P1/P2、run_authorized、equal_budget、equal_FLOPs始终false；composition_benefit=null。即便将来十项 metadata 全match，也不会自动开放执行或科学门。

## 最小后续可执行缺口

1. sampling：提供原report.prepare_sha256对应prepare的明确原路径/引用凭据，随后解析原task/sourcepack SHA及全部四行join。不可按文件名猜，也不可将旧causal padding任务替成moving任务。若原任务确实不同，应新预声明同moving任务的全臂执行，而非修饰旧数据。
2. beam/no-composition：无需再采集task/goal/normalizer身份；仍需全部v3 RESOURCE_AXES的原每root/arm limit与inherited预算。旧记录没有的预算不能事后伪装；可在新执行前冻结共同任务和明确资源轴合同。AD与API/sample成本分栏，不能称等FLOPs。
3. 原加载字节：需原loader权重/配置字节绑定receipt；若不可得，只保留declared-only。现在读取文件或新运行receipt不能回证旧loaded bytes。
4. 新实测控制应冻结同root、goal、实际非零history、normalizer、动作域、全部臂与每模块计账；不根据旧优劣挑根/对手，不追加root或新神经调用到本consumer。

## 命令（cwd=/home/gbk/code/LeWorldModel/le-wm-main）

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p2_resolved_controls_comparison_20261028_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_resolved_controls_comparison_20261028_v1 consume --output results/agent_p2_resolved_controls_comparison_20261028_v1_MAIN01.json
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_resolved_controls_comparison_20261028_v1 verify --output results/agent_p2_resolved_controls_comparison_20261028_v1_MAIN01.json --sha ACTUAL_RECEIPT_SHA
```

输出xb排他创建，8MiB上限；verify要求外部SHA并全量重建。无模型加载或模拟依赖；单测包含RAM正负mutation及实际saved资产闭包，不构成科学通过。
