# P2 Stage C：multi-program composition saved contract

本 stem 是独占的 saved-only contract/consume bridge。它不修改既有 composition producer、physical runner、paired-outcomes exporter、P1 gate 或共享表；不启动神经、CUDA、physics、网络、训练或拟合。

## 已有证据接线

固定输入目录：

`results/p2_composition_paired_physical_20261027_v2_main01`

实际输入 SHA：

- `report.json`: `683dc110081c9e86f384370f46df31f7d1135bb94d42329d97abb79bcf3dfc7c`
- `protocol.json`: `b8600844d51f29782779cda19eafedc6fba6e07e1d9c22a5e8bf4278d4b220a6`
- `physical_report.json`: `e696af5758c88172dafccb30cf62901fe91f9640ebf4df7ad57d4f2f19da8a4d`

实际 schema 有意分别是 `.report`、`.execution`、`.physical_before_encoding`；consumer 按保存文件的真实 schema 校验，不按文件名猜测。

## 独立检查范围

- 固定 4 roots × 2 methods：`joint_h2_no_composition` 与 `conditional_suffix_composition`，8/8 行不可删。
- 每行固定 H2、10 planned/attempted/returned/completed primitive steps、完整 journal、10 observations 和 terminal encoding；缺项 fail closed，不补帧、不替换 root、不按终态重选。
- protocol plan、report row、physical-before-encoding row 的 root/method/selection SHA 精确 join。
- journal 中每个 RGB path 的 PNG bytes SHA 与 224×224 RGB 格式；目录中 report/protocol/physical/notes、8 physical JSON、8 encoding JSON、8 journals 和全部 PNG 做 exact artifact census。
- 固定 report flags：`new_predict_calls=0`、`calibration_only=true`、`P1_admission=false`、`P2_admission=false`、`physical_efficacy_verified=false`、`fair_total_budget_comparison=false`。

代码不导入旧 composition consumer 或 producer。它不重新算已有 Q，也不输出 contrast、均值差、success gain 或因果结论；`composition_benefit=null`、`scientific_claims=[]` 是硬编码合同边界。native/PNG/journal 的存在与完整性不是 P2 admission。

## 命令

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p2_stagec_composition_saved_consume_20261028_v1.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p2_stagec_composition_saved_consume_20261028_v1 \
  --directory results/p2_composition_paired_physical_20261027_v2_main01 \
  --report-sha 683dc110081c9e86f384370f46df31f7d1135bb94d42329d97abb79bcf3dfc7c \
  --protocol-sha b8600844d51f29782779cda19eafedc6fba6e07e1d9c22a5e8bf4278d4b220a6 \
  --physical-sha e696af5758c88172dafccb30cf62901fe91f9640ebf4df7ad57d4f2f19da8a4d
```

成功状态为 `saved_contract_verified_not_admitted`，不是效果验证。现有 saved run 实际通过：8 branches、80 selected public steps、8 terminal encodings、116 bound artifacts；neural/physics calls by this consumer are zero。失败时不写结果、不修改输入资产。

本实现不替代既有 `p2_composition_paired_outcomes_20261027_v2` 的 full arithmetic exporter；它提供独立的 Stage-C contract/census 接口。若将来要声明组合收益，还必须另有预注册比较、完整资源公平性和科学准入证据；本交付不会开启这些门。
