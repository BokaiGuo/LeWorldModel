# blind4 task bridge v2 — original recipe envelope correction

本独占版本修复v1准备层的真实schema错误；旧v1源码/测试/doc及结果全部保留不改。experiment-code仅用于定向修复与受限RAM/metadata回归，不做科学实验。v1 RAM fixture把原recipe伪装成平铺dict，未覆盖封存文件仅有schema/gzip_base64/plan_sha256的真实envelope，造成original[k]会KeyError；旧通过记录不能证明真实prepare可用。

## 实装修复

最新producer仍为 `agent_p4_fresh_source_20261028_v2.py`，SHA `5d2103a883b58db85e9877fb25697ef835e317c529771691a3bdec701a582d3a`；consumer仍为 `agent_p4_fresh_source_saved_consume_20261028_v2.py`，SHA `dcdd2e76559f084a87435e9aaf551eac1ff1d9affdba7eac09ed888a322d29d5`。未修改它们。

producer.build现有路径为 `moving.old.read_prepared()`，old即 `p1_constructive_reachable_source_20261027_v1.py`（SHA `c32b9f169adaed31d07ecef5c507ce409c2ecec5743f8c6fde4fab928b92a878`）。本v2直接调用同一个 `recipe_codec.read_prepared(payload)`，不猴补globals，不重写解包器或假schema。

`read_recipe`先核原recipe明确path/字节SHA `98ffb806018a270f5ef27ffa76be929393a74355c213bc274bf158c72d80b086`，以及本次producer prepare中继承的recipe和decoder绑定。Reader限定envelope至199999字节；现有helper还强制小于200000字节、精确三字段schema、validate=True base64、gzip最多读取2MiB+1并拒超限、解码对象semantic SHA及原verify重建/source/exclusion SHA。保留原helper的完整验证，不降成仅JSON解析。解码后原bindings/exclusions文件也计入本桥Reader闭包；semantic metadata_authority_content_id按原helper规则只验证hash格式，不当路径。admin新增original_recipe_plan_sha256，使用原source.digest。

随后逐字段验证本次prepare的recipe/witness/packet_logit_cap/environment严格等于originaldecoded对应字段，而非等于envelope、默认值或臆造字段。原历史cert/外部approval预检仍在任何recipe/sourcepack/blind读取之前；calibration_only始终拒绝，missing cert failclosed。此功能不新签authority/freshness，不打开保留根。

## 保持的接口与停点

维持全四槽/12 source phases及失败分母，task_export完整3latent/2tokens、H2x5、identity Q与原normalizer，planner_inputs完整四任务xGN/D2P。仅schema/output stem升级v2。仍无run/采集/神经入口；encoding_provenance_verified与execution/P1/P4/freshness释放均false。未来16encoder+16projector仅编码阶段计划cap，实际费用null，不能抹掉source已付物理成本。

原helper full verify会读取其封存bindings与exclusion输入（只沿原明确引用，不刷新历史）；真正未来prepare必须先有合法完整历史证书，MAIN决定是否执行。worker未对真实封存recipe运行该full verify，也未运行真实prepare/生产consume或打开reserved数据。

## 回归和MAIN CLI

RAM fixture通过原codec.build的显式source-only历史生成完整原生plan，再经原prepared_bytes生成真实三字段gzip envelope。没有collect_current_exclusions、环境实例化或模型调用。正常路径执行未经替换的原read_prepared/verify；额外只读回归仅校验实际71KiB封存recipe的字节SHA和顶层schema，不解码遍历它的真实历史引用。

覆盖四继承字段独立变异、平铺JSON、错误schema/base64/semantic SHA、2MiB解压cap、外层字节漂移、decoder/原source继承绑定与原recipe重建错误；保留v1的冷导入、calibration/cert前拒、泄漏拒绝、全分母和verify漂移回归。测试不是科学通过，且mock source consumer的RAM接点不代替真实consumer运行。

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p4_blind4_task_bridge_20261028_v2.py
# 仅未来合法v2 source/cert/外部MAIN approval齐备且MAIN明确授权读取后：
.venv/bin/python -B -m experiments.spectral_reachability.agent_p4_blind4_task_bridge_20261028_v2 prepare --source-dir /ABS/LEGAL_V2_SOURCE --source-sha ACTUAL_SOURCEPACK_SHA --prepared /ABS/LEGAL_V2_PREPARE.json --prepare-sha ACTUAL_PREPARE_SHA --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p4_blind4_task_bridge_20261028_v2_prepare_MAIN01.json
.venv/bin/python -B -m experiments.spectral_reachability.agent_p4_blind4_task_bridge_20261028_v2 verify --input /ABS/BRIDGE_V2_PREPARE.json --sha256 ACTUAL_BRIDGE_V2_PREPARE_SHA
```

当前仍缺合法历史cert/approval和相应合法四槽source闭包，以及后续count4真实HOSTCUDA编码/planner runner。没有可填写的未来SHA，不以新seed视fresh。只写新本stem三文件，输出仍排他4MiB/200MiB余量，Reader128MiB总读取cap不放宽；失败保留，不跑physics/GPU/CPU神经/网络、不创建后代。

实际回归：新v2共56项；联合冻结v1、moving encoder和authority接口共192 passed in 2.56s，另CLI --help成功。联合命令如下。v2冻结后不再原地追加；MAIN审查后才决定真实prepare/verify，worker没有运行该生产路径。

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p4_blind4_task_bridge_20261028_v2.py tests/test_agent_p4_blind4_task_bridge_20261028_v1.py tests/test_p1_constructive_moving_history_encode_cuda_20261027_v1.py tests/test_agent_freshroot_authority_20261028_v1.py
```
