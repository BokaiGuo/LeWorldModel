# Generic selected physical RGB encoder v2

独立新文件，不改 v1。仅两份批准原 task × 事前固定 1–4 个安全 arm 名，完整 root-major × arms 分母；每 branch 编码 prefix5、terminal10。适用 feedback4branches（8 images）与 matched_ridge_GN2branches（4 images）。不从成功率或 Q 过滤行、不去重、不替换早停 endpoint。MAIN 独占 HOST CUDA。

## 合同与信任边界

显式 physical report 外部 SHA→execution_binding→physical prepare SHA→selection manifest SHA→各 compiler report 外部 SHA/显式 JSON pointers。调用冻结 physical.selection()（纯 saved pointers），绝不调用其 material/verify/load_source。selection_digest 沿用默认 spaces 的 physical.source.digest，非 compact。完整 compiler artifact map 逐文件 SHA、physical journal hash chain/operation census、branch started/closed、10条 literal commands、checkpoint state/observation/snapshot/guard、原 task goal RGB 持续绑定，均核验。物理 sourcepack/source prepared/certificate/witness 仅保留 manifest 路径声明，**不打开或重算其内容**；本工具不认证其科学有效性。仅读 physical prepared bindings 中 .py/.md 源码及文档，不重放特权数据 loader。

两次 --task-export/--task-sha 为原批准 task，按 source_slot 0/1，不按 arm 猜路径。load_export/task_interfaces 验 envelope、模型/normalizer/目标与 native192 identity Q；blind 元数据与 compiler pointer 相符。native root hash 只核 physical report 中 root+goal 和其 source_slots 一致，不将 native state 送入 encoder。原 root/goal latent 不重新编码，不训练、不新 Q。

PNG 与 v1 相同：原 RGB224 PNG、文件/pixel SHA、无自动颜色转换；同 p0.image_transform/_image_tensor，CUDA batch1，MATH SDP、TF32 false。继承冻结 enc.guard/Ledger/Store，encoder→projector 每图实际 attempt/completion；4×arms 次 encoder 与 projector（2 roots×arms×2图），0 predict/action_encoder/AD/physics。无 dedup、无重试；partial 保留全部分母、首错/后错、已写 NPY/journal，失败不宣布 complete。完整 source/model SHA 运行前后重验与 tensor before/after；输出 cap2MiB，reserve200MiB。

diagnostic=actual_Q、root_Q、actual_Q_change、actual_vs_root_mse，都是原 native192 identity mean-square。predicted_Q/predicted_vs_real/sign_reversal=null、prediction_status=unknown_not_recomputed；这不是零误差。不同 compiler 的 imagined prefix/observed suffix 语义不同，本版不猜字段或拼接预测，更不新跑模型预测。不选优，不冒充公平预算或本地强基线，不开放 P1/P2。原模型/物理/历史 encoding 成本不退款。feedback 仍是已知任务 saved-prefix feedback 反事实重放 pilot，不是在线试验。

## MAIN 完整命令模板

cwd `/home/gbk/code/LeWorldModel/le-wm-main`。`PHYSICAL_REPORT`/`PHYSICAL_SHA` 替换为已完成实际4branch或2branch结果及外部批准SHA；`ARM` 可重复1–4次且须等于原 manifest arms 顺序。`NEW_PREPARE`/`NEW_OUTPUT` 必须新路径。下面不自动准备或启动任何实验：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_constructive_selected_rgb_cuda_20261027_v2.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_selected_rgb_cuda_20261027_v2 prepare --physical-report PHYSICAL_REPORT --physical-sha PHYSICAL_SHA --arm ARM --task-export results/p1_constructive_blind_encode_cuda_20261027_v1_cuda_main01/task_00.json.gz --task-sha 714f471761278fa654542d4bb0878ce18e3c304df6158005af46d2129533265a --task-export results/p1_constructive_blind_encode_cuda_20261027_v1_cuda_main01/task_01.json.gz --task-sha 70865d56f9b66f0072b2ad7de01cc09d9a9275047925e188104a13d5182ce1f2 --output NEW_PREPARE
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_selected_rgb_cuda_20261027_v2 verify --prepared NEW_PREPARE --prepare-sha PREPARE_SHA
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_selected_rgb_cuda_20261027_v2 run --prepared NEW_PREPARE --prepare-sha PREPARE_SHA --output NEW_OUTPUT
```

feedback 用 `--arm line_search_GN --arm full10_conditional_D2P`，matched 用 `--arm matched_ridge_GN`。PREPARE_SHA 取 prepare stdout，不猜测。

## 下游输出

report.schema=`p1_constructive_selected_rgb_cuda_20261027_v2.report`，完整 status=`selected_rgb_encoded_not_admitted`，errors=[]、source_verified/model_unchanged=true。rows 固定 root-major×arm×[prefix5,terminal10]，包含 index/branch/root_slot/source_slot/arm/role/task_id/task_export/task_sha256/blind_sha256/provenance、latent={file,sha256,dtype:float32,shape:[192]}、metrics。artifact_sha256 绑定 NPY、events、binding；外部调用者仍须绑定整份 report SHA。旧 v1 consumers 不应把 v2 当相同 schema。

RAM mock 测试只证明接线/分母/记账/故障合同，不是模型或物理实测。任何缺帧、partial、错误 SHA、任务/arm/order/goal 不符在 CUDA 前拒绝；新 actual feedback/matched physical 完成后 MAIN 可 prepare。没有自动消耗当前运行的 partial 目录。

交付验证：31 RAM tests 通过，含1–4 arms 全分母、native lossless NPY、partial/no retry、未记账 callback 拒绝、完整 build 的 matched2branches/feedback4branches 通用 compiler JSON pointers 及禁止打开 sourcepack/source prepared。批准旧 actual physical report SHA `d205850e95f639b649fba9b5c6ee8059f2507431753a538b1ce30eccbd234300` 的只读 RAM build 通过：8 images、prepare 序列化97,334 bytes；未写 prepare、未运行模型或物理。该 saved-data 接线实核不代表新 feedback/matched 已完成编码。
