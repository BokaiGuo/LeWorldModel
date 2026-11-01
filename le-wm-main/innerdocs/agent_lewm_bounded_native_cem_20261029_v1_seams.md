# bounded CEM：冻结后三项接缝补充RAM回归

仅新增本说明和 `tests/test_agent_lewm_bounded_native_cem_20261029_v1_seams.py`，
不修改原source/tests/doc/protocol，不新增生产runner或更改科学协议。
原source SHA：`d0f249632793fb1ceca98849e7e716b7cd4d382d092a0adc1ba7c8fa99bab47f`。

复用原真实临时Store fixture（新模块N4/I2/K2，两根完整planning/delivery/nativeIO），
调用完整 `consume`，不替代NPY/JSONL reader。identity入口仅在fixture中替换为合成输入，
所以不是生产source closure、fullstrict或真实CUDA数学认证。

新增覆盖：

- per-root/per-iter seal缺失、criterion提前、update提前；篡改后重建合法canonical hashchain及外部SHA，
  确认失败原因是顺序门本身，而不是hash不匹配。
- 两根×whole/segmented全四路criterion提前、缺layout return、缺global seal，同样合法重新封存后拒绝。
- final raw重载后再tanh，连同新normalizer输入输出及delivery tokens全部写为合法新NPY，
  reader必须在实际sealed raw与normalizer input不一致时拒绝；另核新增mapping事件拒绝。
- 原真实normalizer函数用NumPy-backed fixture tensor逐索引核验
  `(1,N,2,5,2) -> (1,N,2,10)`（N4及300）和 `(1,2,5,2) -> (1,2,10)`。
  不调用torch，不模拟或声称CUDA数值相等；AST同时核raw reload循环没有再次mapping。

MAIN回归命令（工作目录 `/home/gbk/code/LeWorldModel/le-wm-main`）：

```bash
.venv/bin/python -m pytest -q tests/test_agent_lewm_bounded_native_cem_20261029_v1.py tests/test_agent_lewm_bounded_native_cem_20261029_v1_seams.py tests/test_agent_lewm_native_cem_slice_20261029_v1.py tests/test_agent_lewm_finalmean_paid_evaluator_20261029_v1.py
```

无生产prepare、模型/NN/CUDA/physics、生产consume/fullstrict或后代agent。
生产CLI仍是原冻结实现的prepare/verify/run/consume，见原同stem文档；此补充不改变prepare源码闭包。

实际上述联合回归：135 passed，16.27s，exit0；新增接缝测试33项。
已复核原source/tests/doc/protocol SHA均与前次冻结一致。
