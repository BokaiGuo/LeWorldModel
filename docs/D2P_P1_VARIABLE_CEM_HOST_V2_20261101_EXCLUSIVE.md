# Variable CEM host v2：事件文件报告锚定

新入口保留既有 variable CEM 执行路径与安全锁，使用独立名称、授权和 prepare；不修改旧版本及封存结果。报告的 `cem_journal` 在控制器 journal 关闭后记录完整文件 SHA-256、事件数、链尾与 `closed` 状态；缺失、仍打开、写入中毒或链不一致时报告降为 `partial_or_failed`。消费者须取得可信的外部 report 摘要，核验 report 后调用 `verify_anchor`。摘要不是签名，攻击者若能同时替换 report 和 journal 且外部 report 摘要未固定，则不能提供真实性保证。

原有物理 journal 文件 SHA 与链尾、节点 ledger SHA 和合同、prepare SHA 继续由原执行路径记录；此变更未证明实际 GPU 成本、hook 覆盖、模型推理正确性或物理轨迹真实性。节点账本仍为 `closed_unverified`。既有根和预算仍属探索性，不主张独立确认或等成本。PNG 文件的字节摘要与像素语义、环境状态的真实性、prepare 全部来源的独立认证仍须另行审查。

报告生成前若进程异常终止，可能没有最终报告；若在执行器返回之前（包括关闭操作）抛出异常，底层外层 launcher 的失败报告可能没有 `cem_journal` 字段。这些情况均不能视为成功，消费者必须要求该字段及其 `closed` 状态。若事件锚定失败，物理 journal 的早先 `run_finished` 可能仍写有原执行状态，以最终报告的失败状态为准。旧报告没有本合同，不得事后补造锚。

CPU 检查（在 `le-wm-main` 目录）：

```bash
env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_codex_p1_variable_cem_host_v2_20261101_exclusive.py
env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.codex_p1_variable_cem_host_v2_20261101_exclusive --root 0 dry-prepare
```

prepare 与真实运行需要单独授权；本次不运行 GPU 或物理环境。
