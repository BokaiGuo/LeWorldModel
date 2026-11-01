# P5 固定164保存接线：持久化与独立入口验证

仅复用已封存 fixed164 join 与其四个固定输入 SHA，不改旧件；本入口单独 write/verify，verify重新从原保存材料执行整个已封存 join 并逐字节对比，不是第二套独立数学实现，也不重跑上游 full strict。结果绑定旧 join source/tests/doc SHA 与本三件套实际 SHA；任何源漂移或输入 SHA 不符均拒绝。

保留164枝、80对、4 baseline，known roots=2、认证 fresh=0，P1/P5=false；继承原 join 缺终态/早停时整批拒绝，不填0、不复制上一个值，也不减分母。JEPA patch MSE与LeWM goal Q属于不同特征空间，输入权限也不同；并栏不是可比较分数、同输入排名、因果或独立确认。0新神经、0新物理。

write仅允许 `results/` 内本stem新文件，排他创建不覆盖，输出≤8MiB、保留200MiB空闲，写完fsync文件及目录；失败可能留下partial，绝不自动删除或覆盖。verify需实际文件SHA并重建全部内容与源码闭包；若源改动，不能以旧hash绕过。SHA不构成签名。

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p5_fixed164_physical_model_persist_20261029_v1.py tests/test_agent_p5_fixed164_physical_model_join_20261029_v1.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p5_fixed164_physical_model_persist_20261029_v1 write --output results/agent_p5_fixed164_physical_model_persist_20261029_v1_MAIN01.json
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p5_fixed164_physical_model_persist_20261029_v1 verify --result results/agent_p5_fixed164_physical_model_persist_20261029_v1_MAIN01.json --sha256 ACTUAL_SHA_FROM_WRITE
```

本次仅运行fixture回归，未生成实际结果；MAIN审读后可自行运行上述saved-only命令。实际上游source与strict有效性继承既有固定receipt，不冒称本wrapper再次认证每个原生事件。
