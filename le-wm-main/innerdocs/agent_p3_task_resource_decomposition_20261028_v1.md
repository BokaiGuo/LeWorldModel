# P3 原生评价轴与资源分解（saved only）

只新增本stem三文件与独占saved产物，不修改旧paired、物理报告、Q、阈值、chooser或MAIN文档。不执行神经/physics/renderer/网络，不拟合或构造native Q decoder。使用experiment-code技能的saved算术/失败回归部分，用户禁止实验优先；不画科学结果图或宣称测试为科学证据。

## 权威与分母

输入固定为已封存完整paired saved01，外部SHA `55d5ab8c49d31b7d0af3ddcb803f114d272c220c6ab9524076e32df035223288`。先pin其source/tests/doc再调用原paired.verify，全量重建原strict数组/物理native闭包；不重复host.consume或任何实验。逐槽使用handoff对应full_old_row的boundary和last_observation，不另找邻近状态、不取有利结果。核同child/root/context/source/goal/raw5/selection及完整旧物理row SHA。

保留24child/48slot、24pair、2 known-root clusters。全部18双侧已知pair分解；5付费拒绝/2未到达仍missing、不填0。均值同时列分母/known/missing/known-only和完整分母均值（缺一即null）。按2 known roots聚类；root等权字段只是各根已知配对均值的描述，不是假装root0缺失配对已完成，更不是24独立根。保留4 native_stopped，其中2在第5步前、2恰在第5步终止，不padding。

## 评价轴

对保存的state7和同一goal7作float64直接相减，逐项平方：

`native7² = jointXY² + raw_angle² + velocity²`，其中jointXY为state[0:4]（agentXY²+objectXY²），angle为原始第5维差，velocity为第6/7维差。不能用wrapped angle或native success的min-angle项替换native7原始角度平方。每行/每pair/聚合核恒等式；固定16eps float64 roundoff guard只用于算术一致性，不是新科学阈值。

所有差均减同一保存goal7（包括其非零velocity），不把目标速度重置为0。沿用原native混合单位，没有无量纲化/新权重；分量大小不能作为物理重要性或评价轴适当性的认证。

差值始终adaptive−fixed；initial−final另列同槽误差减少。native7平方下降而jointXY平方上升时，由angle/velocity项的负差抵消解释，这只是代数记账，不是因果效果或任务提升。还单列距离均值，不把“均值距离的平方”当“均值平方”。原生jointXY+angle成功仍取完整严格报告，不由native7或objectXY小推断。

## 原Q预测边界

冻结runtime task严格ABI只有binding、root_context、latent goal192、projection192×192及action normalizer。原模型目标为`mean((projection @ (goal - endpoint.latent_history[-1]))**2)`。复算parent、selected五步prefix、selected H1/H3 terminal的latent Q及gain；核原selected terminal cost、候选index、实际raw5和prefix端点。不改原selection，H3 terminal不能冒充执行5步物理端点；native提前停止也不能padding到模型prefix5。

当前source没有绑定的latent→native7/component decoder或物理velocity预测合同。因此“Q是否预言native goal/jointXY/velocity变化”分别为unknown/null，报告原因与任务/projection/goal来源SHA；latent Q下降不是native7、jointXY或velocity预测。共享任务/目标图来源不等于有数值native映射，不做相关拟合或新Q。

保留原实际signal+compile资源轴（含付费拒绝），fixed391/adaptive634predict并非等actual compute；不同API/sample/J/VJP/SVD不折合FLOPs，不加nested资源，不把历史physics记为新支付。P1/P3/C3、eqcompute、freshness等门均false；C3评价轴分解不构成C3通过。

## 可执行命令

工作目录 `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
OPENBLAS_NUM_THREADS=1 /home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python -m pytest -q /home/gbk/code/LeWorldModel/le-wm-main/tests/agent_p3_task_resource_decomposition_20261028_v1_test.py

OPENBLAS_NUM_THREADS=1 /home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python -m experiments.spectral_reachability.agent_p3_task_resource_decomposition_20261028_v1 analyze --paired-analysis /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_budget_paired_analysis_20261028_v1_saved01.json --paired-sha256 55d5ab8c49d31b7d0af3ddcb803f114d272c220c6ab9524076e32df035223288 --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_task_resource_decomposition_20261028_v1_saved01.json

OPENBLAS_NUM_THREADS=1 /home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python -m experiments.spectral_reachability.agent_p3_task_resource_decomposition_20261028_v1 verify --analysis /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_task_resource_decomposition_20261028_v1_saved01.json --analysis-sha256 ACTUAL_DECOMPOSITION_SHA
```

输出NEW独占、16MiB cap、200MiB disk floor，重入拒绝。实际analyze/verify终态与freeze SHA由交付消息记录，冻结后不回写本文档。MAIN正在运行的其他verify不重启、不干预。
