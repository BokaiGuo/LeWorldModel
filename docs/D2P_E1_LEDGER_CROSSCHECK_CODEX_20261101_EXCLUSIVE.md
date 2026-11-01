# E1 保存账本交叉检查（CPU，只读）

本次核查以现有产物为范围，不修改共享账本或运行模型、GPU、物理。
独立测试文件为 `le-wm-main/tests/test_codex_e1_ledger_crosscheck_20261101_exclusive.py`。

输入是 ancestor integrated ledger（SHA256
`0bd6601848d43ed8c5e0763f98e7858a7d6909347046b4a277497eadc0a5988b`）
及较新的 Hi unallocated bridge（SHA256
`7673ac35fcf7103e8231b5b6e0ea1578da05528ebb18fce587119a5a8879261c`）。
本次检验两个文件实际字节、共同来源摘要、六槽身份、原账本输入和预检结果逐槽一致，
以及新增行级事件未被转为已分配 proposal 成本。测试也检查对身份、账本、分配、
门槛和目标字段的篡改被拒绝。

结果：两根各有 arbitrary、learned、compiled 三槽；同一根三槽共同 task 摘要一致，
但 target 摘要各不相同。因此各臂自己的 tracking loss 不能视作同一目标上的性能排名；
摘要不同本身也不是对目标语义可比性的完整审查。六槽预检均 incomplete，proposal
及 inherited 均缺 receipt。两条 learned 行中，各自保存了 Hi high predictor 返回
14 次、98 input rows；这些只说明记录的行级执行，不是该目标的完整因果费用。
两条 compiled 行保留部分祖先事件，但 inherited scope 仍 unknown、receipt 为空。
事件可能嵌套或跨阶段重叠，不相加为 FLOPs，也不把未知当零。

这项检查是两个下游保存投影之间的结构交叉核对，继承其上游证据边界，
不独立证明源事件无漏记、端到端预算公平、任务成功或独立泛化；无显著性分析，
两根也不能当作独立充分样本。下一步需要分别补齐 proposal/祖先事件的因果归属、
失败尝试与共享成本、跨阶段去重和可认证 receipt，再结合独立根与真实闭环结果。

验证命令：

```sh
cd /home/gbk/code/LeWorldModel/le-wm-main
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_codex_e1_ledger_crosscheck_20261101_exclusive.py tests/test_codex_e1_hi_unallocated_event_bridge_20261030_v1.py tests/test_codex_e1_ancestor_integrated_ledger_20261030_v1.py
```

实际结果：31 passed in 5.63s。测试只读已有结果，未重新生成或改写旧产物。
