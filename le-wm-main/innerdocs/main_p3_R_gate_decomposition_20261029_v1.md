# MAIN 原生R资格断点的独立标量分解

源码 `main_p3_R_gate_decomposition_20261029_v1.py` 仅用Python标准库，不导入producer、NumPy或神经框架；完整固定两根×H1/H3、36个zero/候选的192维immediate/nominal以及10维bounded-ridge残差/罚项，以math.fsum重算。继承显式固定的完整saved receipt，不在此重建上游fullstrict、不新调用模型或physics。

14项回归通过（0.29秒）；实际analyze及整份verify成功：`results/main_p3_R_gate_decomposition_20261029_v1_MAIN01.json`，SHA `e1fa1a527e0d5a025cfb3ba5f6fac35900d1d68467e6127a039e9ffe908a113c`。算术误差界只验证复算一致，原始资格仍使用封存值严格大于/小于，不加容差改变选择。

恒等分解：reach = immediate + nominal_drift − bounded_linear_gain，其中nominal是postcontext的H1零动作预测，gain是该同状态局部有界线性控制相对零动作的ridge代价降低。不能把reach的变化全称为“能力变化”，它也包含任务残差和自主漂移。此分解不声称gain是物理可控性的完全度量。

|原control0组|暂时退步候选|reach改善候选|同时符合资格|
|---|---:|---:|---:|
|root0 H1|4|4|0|
|root0 H3|4|3|0|
|root1 H1|6|2|0|
|root1 H3|6|2|0|

root0的全部8个暂时退步候选，其bounded_linear_gain相对zero也下降；不能说这几项已经具备模型内的后续能力提升。root1的12个暂时退步候选中8个gain提升，但幅度不足以抵偿nominal任务残差增长，因此reach仍更差。例如root1H1/index2的immediate增量0.08291796、nominal drift增量0.01721052、gain增量0.03562189，合成reach增量0.06450660。这里不是一个统一offset，也不是gate实现把已有合格候选漏掉。

资格0/32是当前两根、原control0、固定spectral8变体的完整模型内有限证据，不是实际物理R效果、fresh确认、所有候选空间无解或一般理论反例；仍需检查候选生成目标和真实反馈时刻的因果接线。下一actual-cp5合同独立事前固定，不因本轮结果扩大bank、调阈值或删除根。

另MAIN直接读取P2 history3 tracker两根完整保存数组，以math.fsum复算returned mean终态192维成本，与封存float64代价在2.3e-16以内一致。两根mean cost为0.8376906938950434/1.1247193481483777，末轮elite平均cost为0.8354180353587246/1.1174687220066852；差为0.00227265853631875/0.007250626141692473。该差异证明不能把原CEM返回的elite平均值当实际输出mean的成本；共同tracker已付费重放并区别二者，不将此数值差直接归因为历史P1根因或控制受益。

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q tests/test_main_p3_R_gate_decomposition_20261029_v1.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.main_p3_R_gate_decomposition_20261029_v1 verify --output results/main_p3_R_gate_decomposition_20261029_v1_MAIN01.json
```
