# Stage-C saved receipt writer

此独占 stem 只把已冻结的 Stage-C saved consumer 结果持久化为新 receipt；不修改 consumer、physical 目录或既有报告，不运行神经/物理，不创建 v2 manifest。

writer 严格绑定：

- `report.json` SHA `683dc110081c9e86f384370f46df31f7d1135bb94d42329d97abb79bcf3dfc7c`
- `protocol.json` SHA `b8600844d51f29782779cda19eafedc6fba6e07e1d9c22a5e8bf4278d4b220a6`
- `physical_report.json` SHA `e696af5758c88172dafccb30cf62901fe91f9640ebf4df7ad57d4f2f19da8a4d`
- frozen consumer SHA `9255f3a34149776305b8503b3088251c6461095337eda3957555688f0c469fa2`

调用既有 consumer 后，以 `xb` 排他创建 receipt、写入并 fsync 文件与父目录。receipt 保留完整 `consumer_result`，并硬性写入 `composition_benefit: null`、`scientific_claims: []`、`P1_admission: false`、`P2_admission: false`。已存在目标、consumer 非 verified 状态、收益非 null 或 admission 改变时拒绝写入。

## 命令

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p2_stagec_composition_saved_receipt_20261028_v1.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p2_stagec_composition_saved_receipt_20261028_v1 \
  --directory results/p2_composition_paired_physical_20261027_v2_main01 \
  --report-sha 683dc110081c9e86f384370f46df31f7d1135bb94d42329d97abb79bcf3dfc7c \
  --protocol-sha b8600844d51f29782779cda19eafedc6fba6e07e1d9c22a5e8bf4278d4b220a6 \
  --physical-sha e696af5758c88172dafccb30cf62901fe91f9640ebf4df7ad57d4f2f19da8a4d \
  --output results/agent_p2_stagec_composition_saved_receipt_20261028_v1_MAIN01.json
```

receipt 是 saved integrity 回执，不是组合收益、P2 admission 或科学结论。
