# P1 已知根变长 CEM 与 structured 配对账面复核

2026-11-01，只读复核已有真实闭环收据，无新模型推理或物理执行。
按明确 root 索引分别校验双方目标／根绑定、模型前后哈希；structured
报告固定 SHA、节点顺序、账本上界和前向 hook 一致性；变长 CEM 复用
已有保存事件审计，检查候选、节点账本、物理事件和首次成功。

| root | 方法 | 原生成功步 | 已封存节点 predict 上界（0/5/10/15） | RGB 编码 |
|---|---|---:|---|---:|
| 0 | structured | 15 | 174 / 192 / 183 / 未启动 | 3 |
| 0 | variable CEM | 17 | 195 / 194 / 193 / 193 | 4 |
| 1 | structured | 14 | 174 / 192 / 183 / 未启动 | 3 |
| 1 | variable CEM | 6 | 195 / 194 / 未启动 / 未启动 | 2 |

两臂已观察节点的 action encoder / predict / predict samples 账面
上界均不超过 195；这是**事后检查**，不证明 structured 的旧运行
事前受 195 硬门控，更不证明总算力相等。structured 每个已运行节点
另外计 backward 240、VJP 240、Jacobian/SVD 各 1；CEM 账本未计
selection/replay，输出将其设为 null，不把账本的零误作未发生。
未启动节点同样为 null。不同提前终止、未认证实际 GPU 成本或覆盖、
共享工作和模型加载缺项均阻止等成本结论。

CEM 历史报告未直接锚定其 CEM 事件文件：事件内部链与语义核验不等于
由报告外部认证。structured 的本次复核未重新执行或完整重放物理。
两根已见结果，不能作独立泛化或 P1 准入证据，也未更改任何 gate。
未来的公平确认需在未见根／目标上事前冻结双方控制器、统一适用的
节点 cap、失败与早停规则，并分别审计完整资源向量和真实闭环回执。

复核与合约测试：

```sh
cd /home/gbk/code/LeWorldModel/le-wm-main
env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_variable_cem_paired_cost_audit_20261101_leibniz
env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_variable_cem_paired_cost_audit_20261101_leibniz.py
```

结果：双根保存收据复核通过；6 项 CPU 合约测试通过，覆盖根／目标、
模型、节点名单、上限、hook 的错误拒绝与缺失值语义。没有运行 CUDA。
