# P2 composition increment saved-data audit v1

本文件是独占同stem新文档，不修改共同研究计划或任何旧封存。core仅stdlib/NumPy；
live saved-data loader复用旧模块（可能有可视化依赖导入），仅算术与文件读取，
不调用神经/物理API、训练、下载、结果调参或筛根。退出0表示生成审计，
不表示P2通过。本版本没有科学准入路径，即使mock P1自称pass也不能开启P2。

## 现有真实资产与审计路径

paired planner：`results/p2_composition_paired_20261027_v2_prepared01/report.json`；
physical：`results/p2_composition_paired_physical_20261027_v2_main01/report.json`；
已有独立outcomes与code audit分别位于同stem `*_report01` / `*_audit01`。
新审计不是另一个实验：复用封存outcomes的只读loader，重新调用原exporter的纯数组
全候选重建，验证source/input/archive、40slots/臂、winner、动作与双重replay收据、
物理journals/RGB、原Q/official/pose。随后独立检查预算算术和准入缺口。
保存导数/重放收据一致性不等于独立导数真实性或物理反事实验证。

稳定root SHA+arm配对，root_index只校验不作为join；输入乱序无影响。所有八臂保留。
预测Q、真实终态encoder Q、object pose和official距离分开，不跨单位相减。
只有完整同H2双臂产生固定H2差值；partials保留原观测，不补动作、不缩短预测horizon、
不把guard终态假作完整H2。缺实际编码显示null/unavailable，不假填0。

## 已知预算，不是假matched

每根共享4predict，joint exclusive84，composition exclusive164；standalone88对168，
paired unique252而非88+168。四根1008predict及1008action encoder，各3024context tokens；
164Jacobian、31488数学标量VJP方向、168SVD、164bounded solve、320选择评估。
两臂同H2/10commands、40slots及相同逐slot prefix不意味着同suffix分布或总计算公平。
nominal/ADforward、候选与winner双replay均包含在上游实际账本。

新core逐项检查shared/exclusive/standalone/unique算术，不能重复计shared。预算缺字段
输出机器可读missing/unknown及null总量，绝不当0；已提供预算自相矛盾则完整性拒绝。
上游全候选数组重建额外168SVD/164solve另列（无新神经）；Q/pose重算与文件检查也有
CPU开销。FLOPs、backward engine API/kernel次数、上游RGB/context/Q构造、统一强制
walltime上限、严格优化/展开与全候选物理成本未齐，不能称全资源matched。

## 可执行P2前置缺口

1. 原P1 independent mechanism consumer与原zero-prefix、greedy、CEM controls及严格
   paired intervals；历史physical outcome为fail（非贪心条件false），不是没有代码。
   model calibration/new paired execution不能替换原P1。此审计不追认新P1证据。
2. 新封存的总资源公平配对runner与强controls，显式AD forward/共享/失败预算，不能
   删除composition额外80predict/root，或只把两臂candidate80相同称matched。
3. 原严格P2全候选物理/编码覆盖与真实每arm来源adapter；当前40×2×4候选仅选中8臂
   执行物理，不能把winner覆盖当320slots。已有最小节点suffix compile资产不再列为缺失。
4. 原机制下独立增益/非贪心证据、共同动作支持/目标消融，以及新独立cohort、停止与失败
   规则、root配对不确定性及多重比较预声明。当前四根已使用，不能作为新独立确认。

## 命令

cwd `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_p2_composition_increment_audit_20261027_v1.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m experiments.spectral_reachability.p2_composition_increment_audit_20261027_v1
```

默认stdout机器JSON。可用`--output NEW_FILE.json`独占写一个小JSON，父目录必须已存在，
不递归准备、不复制大protocol/source archives。输入缺失、source漂移或合同不一致输出
`refused_missing_or_inconsistent_evidence`、integrity=false、P2=false；通过保存数据验证
则`saved_data_audited_P2_blocked`。纯NumPy/mock tests不代表实验成功。
