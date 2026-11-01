# Hi/P2 五组目录：MAIN data-only 导出与 saved join

仅新增同stem source/tests/本文，不改任何旧文件。worker仅RAM/saved metadata测试，没有读取weights/NPY、执行NN/physics/fullstrict、训练或refit。使用experiment-code的source-first/日志/失败保留边界，不做其训练/结果改进流程。

## 已检查的复用入口

定向检查既有 `agent_p2_three_program_20261028_v1_cuda_MAIN01/loader.json`：只有aggregate tensor_hash及原文件同buffer收据，没有逐key目录。新stem不存在同名实现；对明确相关源码/文档/tests文件名作定向检查，没有扫资产大目录。不声称全库都无其他目录。

A直接调用已固定SHA的 `agent_p2_equal144_baseline_cuda_20261028_v1.loaded_model` 和 `validate_loader`，不重写其权重加载/重映射：原文件读为immutable bytes→核SHA→同buffer deserialize→remap→missing/unexpected严格空→CUDA eval/frozen→原aggregate hash。原函数内部CPU checkpoint反序列化和构造初始化仍是原边界；**不是CPU神经forward或CPU fallback**。本模块先要求MAIN授权/原venv torch/真实cuda:0 allocation，拒绝无CUDA环境。原P2 weights是本地已信任且按SHA绑定的weights_only加载，不执行Hi foreign pickle或导入其原类。

精确资产：

- P2 prepare `results/agent_p2_three_program_20261028_v1_prepare_MAIN01.json`，SHA `86af2d24f73e769fb3b91fa8a571f8ca265df95c93af780d3924ea644585d461`。
- 原loader `results/agent_p2_three_program_20261028_v1_cuda_MAIN01/loader.json`，SHA `2401302225f042755bd9a8e8a06874ebafd367dee5937434b442c739b45b4660`。
- `/home/gbk/.stable_worldmodel/checkpoints/pusht/lewm/config.json` SHA `2564086e961e7b5c7c04dffc451091115b389a590645ff19653c64fd0bc16e09`；weights.pt SHA `48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607`。worker未读weights。
- 原CUDA aggregate model hash `4b12c01f5859f4b46bea478f3a8d5e2d3f516df456a622a3feec31bde7e33546`。

prepare仅核小metadata/config、明确原Python import依赖闭包及本三件套SHA，不调用原material/build/consume。原prepare绑定的weightSHA原样留给loaded_model实际读取时检查；不会假装prepare已重新验weights。所有原source pins在源码明确列出；闭包中已纳入原prepare bindings的文件也须逐项一致，漂移拒绝。Python闭包以源码显式import引用递归，最多512文件，不扫描目录。实际构造后另记录已用module类文件、torch functional/hydra/omegaconf等runtime文件SHA并前后比对；这是记录执行环境的来源边界，不是完整二进制kernel/安装包供应链认证。

## A：导出究竟保存什么

每module `_parameters/_buffers` 原位置枚举，包含显式None、persistent与nonpersistent buffers，`named_modules(remove_duplicate=False)`保留alias路径。每非None项保存role/dtype/shape/stride/offset/C-index原值bytesSHA及byte数；分别按tensor对象identity与CUDA storage pointer/size建立**key集合**等价类，输出中不保存不稳定指针值。无别名时包含self，None是空alias且没有值/SHA，不填零。严格只接受原float32/int64小端。持久非None key集合必须完整等于runtime state_dict。

两遍逐key目录完全相等；原aggregate hash也必须前后相同并等于旧实际值。原参数/全部buffer CUDA frozen/eval，复制时version未变。只用detach/CPU contiguous取证，不调用模型forward/AD；加载返回后对所有module装拒绝forward的prehook，finally清理，不改原方法/globals。构造期间无forward由原loaded_model固定源码承担，不伪称hook覆盖构造前代码。

目录使用**P2 runtime remapped key**；不再读第二遍checkpoint去恢复序列化前key，故不虚构original_key。remap的七条names-only规则由纯RAM AST回归与原source精确核对，无transpose/QKV拆合/值转换。

run排他新目录保存report（成功或失败），600s cooperative总wall、16MiB metadata输出、200MiB disk floor；不保证抢占原loader/kernel，磁盘/强杀不能保证最终文件。保留load intent/return收据、完成目录、失败列表、未完成null，不重试不退款。predict/encoder/backward/physics均0；两遍CPU取证和load/wall独立说明，不把它称免费或equal-FLOPs。

## B：saved消费与join的证据边界

`consume`重建本prepare源/小资产，完整复核loader同buffer收据/事件及原model hash、目录前后、runtime源和资源记录；**不加载weights，不独立重算其tensor值**。目录原值SHA来自MAIN真实导出，不能从单项hash反向复算旧aggregate hash；receipt显式 `independent_tensor_rehash=false`。不把该消费叫独立全权重验证。

`join`对Hi既有export完整byteSHA和真实exporter verify receipt精确相等检查；使用原 `state_dict_paths/rows.source_entry/array`，不读任何NPY。当前Hi导出明确module/node/storage alias均空，本版本严格只支持此已知nonalias ABI；若出现非空就拒绝，不能假装已认证未知alias。保留491总条目中的None与425非None；五组非None分母198/9/6/81/9。

固定五前缀：encoder→encoder、projector→projector、action_encoder→action_encoder、low_predictor→predictor、low_pred_proj→pred_proj。前缀变换后仅原七条ViT key remap，碰撞拒绝。每组全missing/unexpected/mismatch都输出；role、dtype、shape、persistent、None、value bytes/SHA、tensor/storage alias集合逐项严格一致才可标 `tensor_catalog_equal`。stride/offset差异另表完整保留，它们不改变C-index数值身份，但阻止从目录等值推断计算等价。空组即使两侧都空也为unestablished。

结果永远保留config/computation/normalizer/original-class/scientific配对false。目录相等不是同192坐标、同pixel/history3函数、原scaler或三臂baseline可运行证书。原config逐字段与不同实现计算、原normalizer资产继续是独立前置；不借用P2常数，不refit、不fit mapper、不挑root/outcome。

## MAIN准确CLI（worker未执行生产）

cwd `/home/gbk/code/LeWorldModel/le-wm-main`；所有输出必须不存在，SHA变量取前一命令实际输出，不填写未来SHA。

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_agent_p2_hilewm_pairing_catalog_20261029_v1.py
.venv/bin/python -m experiments.spectral_reachability.agent_p2_hilewm_pairing_catalog_20261029_v1 prepare --output results/agent_p2_hilewm_pairing_catalog_20261029_v1_prepare_MAIN01.json
# 将实际prepare SHA赋给 HILEWM_PAIRING_MH
.venv/bin/python -m experiments.spectral_reachability.agent_p2_hilewm_pairing_catalog_20261029_v1 verify --manifest results/agent_p2_hilewm_pairing_catalog_20261029_v1_prepare_MAIN01.json --manifest-sha256 "$HILEWM_PAIRING_MH"
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.spectral_reachability.agent_p2_hilewm_pairing_catalog_20261029_v1 export --manifest results/agent_p2_hilewm_pairing_catalog_20261029_v1_prepare_MAIN01.json --manifest-sha256 "$HILEWM_PAIRING_MH" --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_hilewm_pairing_catalog_20261029_v1_cuda_MAIN01 --main-authorized
# 将实际report SHA赋给 HILEWM_PAIRING_RH
.venv/bin/python -m experiments.spectral_reachability.agent_p2_hilewm_pairing_catalog_20261029_v1 consume --report results/agent_p2_hilewm_pairing_catalog_20261029_v1_cuda_MAIN01/report.json --report-sha256 "$HILEWM_PAIRING_RH" --output results/agent_p2_hilewm_pairing_catalog_20261029_v1_consume_MAIN01.json
.venv/bin/python -m experiments.spectral_reachability.agent_p2_hilewm_pairing_catalog_20261029_v1 join --report results/agent_p2_hilewm_pairing_catalog_20261029_v1_cuda_MAIN01/report.json --report-sha256 "$HILEWM_PAIRING_RH" --hi-report results/agent_hilewm_storage_npy_export_20261028_v1_MAIN01/report.json --hi-report-sha256 3cd7dd6783129893f397ddef63bd253e447fbef93582dd5f4dc8d7144310efd3 --hi-receipt results/agent_hilewm_trusted_cuda_validation_20261028_v1_prepare_MAIN01/export_verify.json --hi-receipt-sha256 afec1d335e5380d740930a5e8ea9f9af4bfb445847d3821ad97c54dbd6c28069 --output results/agent_p2_hilewm_pairing_catalog_20261029_v1_join_MAIN01.json
```

没有任何worker生产prepare/export/consume/join结果。RAM测试只用fixture或已点名小metadata；真实加载/导出和配对结论待MAIN执行。原frozen facade/probe/所有结果保持原样。
