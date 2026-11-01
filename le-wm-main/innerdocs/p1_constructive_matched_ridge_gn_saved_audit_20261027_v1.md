# Matched-ridge GN 独立 saved 算术审计

仅新增本stem源码/tests/本文及未来小报告。不导入variant/producer优化器、consume、selected或AD/model实现。仅复用独立旧saved审计器的JSON/hash/有界gzip、数组/history/prediction、投影、lane账本等纯工具；不调用其旧λ的`gn_audit`或总`audit`。

## 明确范围

外部参数必须提供完整reportSHA、prepareSHA；禁止从report自己取expectedSHA。绑定冻结variant源SHA `d72f6986070f2d75db4f0df711bda97a0ef636fabe566a271ff4b2ec337f0d26`，两批准exports及源/模型hash、完整两根result/plan/journal。模型文件只hash，不加载。不读physical、blind NPZ、sourcepack、certificate或旧outcomes，不把物理未完成当数据补齐理由。

独立复算8个J方向：逐轮192行raw-control VJP×唯一tanh链→192×20 logitJ，绑定fresh incumbent的ADforward两packet/terminal/history；gradient为原unpenalized MSE。主复算使用独立未归一化normal equation：

```text
(A.T A + .002 I) delta = -A.T r
```

再用归一化SVD与 `lambda_mean=.002/192`交叉检验。不是调用producer.algebra/optimize，也不修改sealed globals。报告每轮最大direction差及normal-equation残差范数。容差沿用旧独立算术工具 `rtol=1e-8,atol=1e-10`，不是收益/科学准入阈值。

重建endpoint投影和五固定scale float32线段、NumPyfloat32 tanh10×2动作；独立重算所有35次H2评价cost和70个path；核request/nativepacket/complete因果顺序、每轮先freeze后试五候选再稳定选优、全新三重replay后更新，零初始化、4更新完整。所有旧incumbent/tie/rejected trial均保留，不因实际cost更差调整任何参数。

lane逐attempt/return及索引/嵌套核算；每根78predict+78actionencoder，含8ADforward，不再另加；4J/768数学VJP、768scalar grad API、4SVD/solve/proposal/selection。两根156/156、8J/1536VJP。ADforward与fresh试验scope分别检查；API不与数学方向混称。plan selection seal、report commands/source_slot/arm/blindSHA/taskID逐字段绑定；拒绝partial/错误/未知或漏return，不释放或修补旧计划。

## 本地snapshot与正式外部绑定

实际位置：`results/p1_constructive_matched_ridge_gn_cuda_20261027_v1_cuda_main01/report.json`。
MAIN目前给出前缀 `7eed1f6...`；本地只读实算完整值为：

```text
report  7eed1f6ab7543e6dc4be367c5141ab23cb44dc014fae770fc58fe813422137d5
prepare cec47cb477b06cbbecc9a402f6f17415d3a67cec576679aab906dc68a733bf05
```

测试固定此本地snapshot。正式小报告由MAIN确认完整SHA后显式执行，作者不把自行实算SHA冒充另一次外部确认。代码与tests就绪不等于neural/physical/科学通过。

只读RAM全审计已验证8734events、两根modelcost `1.0146352777046366/1.4551953375149658`、全部8J/1536行、156predict156actionencoder；actual输出1,420,122B。direction与独立normal equation最大差约`7.94e-10`，保存方向normal-equation残差范数约`1.32e-12..8.05e-12`。这支持“实际执行了匹配系数”的saved一致性，不支持“修复有效”。本工具不读取旧GN结果做收益排名。

## 执行

项目根 `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_constructive_matched_ridge_gn_saved_audit_20261027_v1.py
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_matched_ridge_gn_saved_audit_20261027_v1 --report results/p1_constructive_matched_ridge_gn_cuda_20261027_v1_cuda_main01/report.json --report-sha 7eed1f6ab7543e6dc4be367c5141ab23cb44dc014fae770fc58fe813422137d5 --prepare-sha cec47cb477b06cbbecc9a402f6f17415d3a67cec576679aab906dc68a733bf05 --output results/p1_constructive_matched_ridge_gn_saved_audit_20261027_v1_saved01.json
```

exclusive输出<128KiB、写前保留200MiB；失败生成小型rejection报告，源资产不变。没有大fixture/下载/neural/physics。gzip只接受完整单stream+换行，hash/源/任务/模型与输出census均重查，结束TOCTOU。报告绑定自身与复用独立工具源SHA。没有重执行神经/导数真值证明、无P1准入、不等同D2P优化问题或预算。
