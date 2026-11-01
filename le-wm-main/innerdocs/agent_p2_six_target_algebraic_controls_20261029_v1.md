# 六槽纯saved代数负对照：只实现已批准第2节

对应冻结local_intervention协议SHA `c9c03fe6df2188be7865ad55d8aae9fd6b2665c56e7bb57329639a20329a712c`，不扩展动作实验或协议链。仅新增本stem source/tests/doc；不改旧文件。工具无NN/physics/fullstrict，实际build/verify由MAIN执行。

## 固定输入及认证继承

直接使用冻结decomposition模块的`actual_refs/documents/source_join/load_latent/original_rows`局部函数，不调用其build/verify/analyze或任何consumer。该模块仅stdlib+NumPy，无上游导入。固定其源码SHA `6aacaf5fb77546c6987f8d3f3de9318716c66953742ab944d4cdba8f50cf7b9f`，连同其源测文、原协议闭包及本次三件套全部核SHA。

RGB原四文件SHA固定为：prepare `14aace0fad5c941fce475a41d45bb1dfa1bad9388cba9f767bdd4eb42aa4f393`、report `26e00d955ee2c106e4452b97ddca26a7731d721f7e00605f5750e3a38981460f`、completion `f54b1ba7fc7657db1b738f1fe8be5ca280d790a14d5fae0a1a5326d0ac5bcb23`、receipt `c16d99ba82c8d3adeb68515e2035cc191bd060ab6c273e07ff21cef9a23d870b`。CLI不提供换批次入口。额外只读取report绑定native seal和24个小latent NPY；不读新decomposition分析、不读PNG/weights，不扫描目录。

复核四文件互指/完成状态/整份导入receipt/模型前后、原native seal、同root/task/history/raw前缀/target/goal/cp源。原f32输入独立重算24行原scalar/identity/gain并与MAIN RGB receipt精确比较；cp0逐字节核。此为导入MAIN认证条件下的局部独立算术，不冒独立fullstrict或模型验证。报告保留source_time/target_origin，known2、prior时距unknown、费用不齐不变。

## 固定操作与输出

全24位置不删。p/z/t/g先核原f32，再转f64；所有派生向量持续f64，不投controller、不重编码、不改原数组/原评分。

- 几何对照：s=−1/0/+1，辅助`t_s=t+s·2^-8·u0`。保存72个辅助target完整192值/f64字节SHA及p、z分别对它的tracking、几何、signed cross、原task MSE、重构和roundoff。p/z/g不变，原task residual E/C_g原样并列。不因G变化宣称实际target收益。
- 残差对照：a=−1/2、0、+1/2、1，`p_a=z+a*(p-z)`。保存96个派生预测完整f64向量及分别对g/t的直接MSE、E/C、重构、roundoff，以及a²E/aC期望值和缩放差。浮点派生p_a可能产生舍入，不强制a=1向量等于原p；额外原p距离如实保存。±1/2相同能量/反向cross是实数代数期望，不引入容差验收或通过阈值。
- z缺失：模型几何对照可保留，真实几何及残差派生为null；仍保留72/96设计分母与missing reason。源错/hash错不当missing。无winner、threshold或新科学gate。

原MSE、cross及舍入均用f64乘减、math.fsum/192；signed cross不截零。派生向量只是数学负对照，不是真实预测改善、oracle控制器或物理证据。输出不做pair排名/新统计分析。12分支真实动作实验未实现、未授权。

## MAIN命令

工作目录 `/home/gbk/code/LeWorldModel/le-wm-main`，用模块入口保证本项目包可导入：

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p2_six_target_algebraic_controls_20261029_v1.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p2_six_target_algebraic_controls_20261029_v1 build --output results/agent_p2_six_target_algebraic_controls_20261029_v1_MAIN01.json
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p2_six_target_algebraic_controls_20261029_v1 verify --summary results/agent_p2_six_target_algebraic_controls_20261029_v1_MAIN01.json --summary-sha256 <build实际SHA>
```

原读取器上限单metadata/source128MiB、NPY压缩/展开2MiB且分配前核header；输出16MiB、free floor200MiB、整次读/算/fsync120秒。排他创建，源/输入读后全部复核SHA，超时保留已写失败文件，不覆盖重试。verify仅重新读这批小资产并全量重建本输出。新encoder/predict/AD/physics均0，FLOPs=null。

RAM覆盖精确二进制分解、正负cross、f64派生不能偷转f32、roundoff保留、null、不改输入/负零、24分母、四文件fixture build→write→verify、篡改/partial拒绝及无consumer调用。worker不做实际saved build/verify。MAIN验收之后才可执行；不是物理部署或科学机制准入。
