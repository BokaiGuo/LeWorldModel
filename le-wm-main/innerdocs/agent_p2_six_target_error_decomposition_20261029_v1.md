# 六槽 saved 三层误差分解 v1

仅新增本stem源/测/文。固定known2 × 三目标 × cp0/5/10/15共24位置；每根三种无序配对 × 四cp共24pairs，方向right-minus-left。每槽保留四个cp0差分（含自身）及三个相邻差分，共24/18。此工具不导入任何上游consumer，不运行fullstrict、模型、编码、AD或physics；MAIN严格认证为导入证据，本地小数组算术为独立重算。

## 权威与输入

四文件路径及SHA在源码ACTUAL中固定：RGB prepare `14aace0fad5c941fce475a41d45bb1dfa1bad9388cba9f767bdd4eb42aa4f393`、report `26e00d955ee2c106e4452b97ddca26a7731d721f7e00605f5750e3a38981460f`、completion `f54b1ba7fc7657db1b738f1fe8be5ca280d790a14d5fae0a1a5326d0ac5bcb23`、receipt `c16d99ba82c8d3adeb68515e2035cc191bd060ab6c273e07ff21cef9a23d870b`。另外核report绑定的native seal及24个gzip NPY latent。只读这些小数组/元数据和源闭包；不读取PNG、权重或祖先大型数组。三个上位协议及冻结RGB源码SHA亦固定。

四文件互指、completion、完整分母、model before/after、整份嵌套receipt、资源元数据、native seal逐项核对。读取原始f32预测/target/goal，逐slot核完整raw15字节/prefix、task、causal-history SHA、target、原goal source、实际cp PNG来源/状态/response。cp0 latent必须与observed history逐字节相等。预测仅用paid actual deployment tokens，不用旧normalized mean分数。独立重建原receipt全部24scalar rows/24pairs并精确比对；失败不出认证结果。没有新认证祖先完整执行的主张。

## 算术与三个断点

所有latent输入保留f32；差/乘转f64，用math.fsum/192。对p（模型）、z（真实编码）、t（本槽目标）、g（原任务目标）分别保存：

`MSE(p,r) = MSE(z,r) + mean((p-z)^2) + 2mean((p-z)(z-r))`，r=g或t。

`MSE(x,g) = MSE(x,t) + MSE(t,g) + 2mean((x-t)(t-g))`，x=p或z。

保存带符号交叉项、重建值及浮点roundoff residual；不引入容差、clamp、identity通过阈值，也不把target距离单项解释成因果质量。

逐pair独立分类A：Δmodel task Q→Δactual task Q；B：Δactual task Q→Δjoint XY error；C：Δactual task Q→Δnative7 distance（不是负reward）。反转仅用`a<0 and b>0`或反向比较，禁止乘积判号以免下溢。各断点分别统计same nonzero sign/strict reversal/tie/null，tie再分both/first/second zero。±0均tie；null保留、不填分数；非有限输入拒绝。所有24分母不变。

MAIN已报告描述性A反转3、B反转1、C反转7及ties6/null0；这些数字**不是本工具验收常数**，工具分别重算各断点及其tie/null。Repeated pairs不独立，不做显著性或跨pair独立性推断。

native7独立按原状态/goal重算，拆agent XY、block XY、raw angle、agent velocity四组平方；wrapped angle仅并列诊断，不替换原native min-angle或success公式。latent、XY、angle、velocity不混单位；尤其C中velocity贡献不叫模型预测误差。无winner输出。

## MAIN执行

在`/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p2_six_target_error_decomposition_20261029_v1.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B experiments/spectral_reachability/agent_p2_six_target_error_decomposition_20261029_v1.py build --output results/agent_p2_six_target_error_decomposition_20261029_v1_MAIN01.json
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B experiments/spectral_reachability/agent_p2_six_target_error_decomposition_20261029_v1.py verify --summary results/agent_p2_six_target_error_decomposition_20261029_v1_MAIN01.json --summary-sha256 <build实际输出SHA>
```

路径及四SHA也可显式经`--prepare/--prepare-sha256`等四对参数传入，但必须等于固定实际authority，不能换一批结果。输出排他创建、fsync文件和目录；已存在则拒绝，不覆盖失败。单JSON/source128MiB、单数组压缩/展开2MiB、输出16MiB、磁盘floor200MiB、整次本地读/算/写120秒；超时保留已写输出但命令失败。verify仅重建本地分析及输入哈希，不fullstrict。新encoder/predict/AD/physics均0，FLOPs未知。

RAM测试覆盖四文件join、临时gzip NPY、build→exclusive write→verify、原receipt scalar/pair篡改、source身份/raw前缀/response、cp0、null传播、三断点ties/下溢反号、分解正负cross terms及native单位。测试全为fixture，不计算实际24pair结果。

边界：known2非fresh；prior物理时距unknown；目标生成budget不同；结果均同批描述性诊断，不宣称公平baseline、物理winner、P1/P2机制或科学准入。生产saved build/verify由MAIN执行，本worker未执行。
