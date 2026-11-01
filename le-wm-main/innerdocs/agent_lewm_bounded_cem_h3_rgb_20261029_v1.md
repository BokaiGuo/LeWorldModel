# Bounded CEM H3 actual RGB bridge v1

固定known2 × cp0/5/10/15共8槽，绝对native步10/15/20/25。实际root0在own10原生成功早停，cp15缺失；root1 full15。只编码7张实际RGB，不补图、不去重、不筛根。原147PNG/45public/458physics是继承成本，不重新执行。native_success直接保留：XY+angle成功不等于含velocity的native7距离为零，不用reward或Q替代success。

## Authority / trust boundary

以下路径均在 `/home/gbk/code/LeWorldModel/le-wm-main/results/`。

- physical prepare `agent_lewm_bounded_cem_h3_physical_20261029_v1_prepare_MAIN01.json` SHA d17ffc349dedd7d310764f8cb6d7c4c615c9d70bc2e85ee85cd3a5991db431a8
- physical output `agent_lewm_bounded_cem_h3_physical_20261029_v1_physics_MAIN01/report.json` SHA 8705d1877a37d81b835b04935aadaf42af4a78a4e1fb4b9524fadb2f516f59d1
- 同目录completion.json SHA c9310caa342d89637e7f49cd58c3258e2279bb75c027af0ad345183dfe81a63a
- MAIN完整receipt `agent_lewm_bounded_cem_h3_physical_20261029_v1_consume_MAIN01.json` SHA 1e69a2fe082af3c5da32ec28ac8ba10b2375b5a593fa0bf69a60367d7cbc9b86

信任MAIN已持久化完整physical receipt；核四权威、完整rows/plans/counts、inventory、source/model/PNG绑定。不调用旧producer.consume/verify，不冒充再次独立核完物理。自己的consume完整重建新编码所有arrays/events/guard、seal/scoring和completion。原CEM numerical fullstrict同样不重复；读取其已付whole预测和原goal192，不再预测。

## ABI / integer bound

每图无损保存7数组：uint8[224,224,3]、f32[1,1,3,224,224]、f32[1,3,224,224]、f32[1,257,192]、f32[1,192]两份、f32[192]。逐图encoder1+projector1；实际7+7，最大8+8。0predict/action_encoder/AD/physics/训练/goal重编码。所有可用图编码和native IO先seal，再做identity192 Q、预测误差及cp0原latent逐位/maxabs/ULP诊断；cp0相等不是门槛。

预算由source.shape_census整数式核：NPY每数组128字节头，gzip上界 n+(n>>12)+(n>>14)+(n>>25)+31；所有重复JSON共16MiB硬上限，另8MiB最终reserve，64MiB总cap，200MiB磁盘floor，600s inclusive wall，CUDA allocated3GiB/reserved4GiB。实际49数组/71events/125文件；最坏56数组/81events/142文件。Store和consumer同时核全文件及metadata预算，不能缩精度/分母。MATH仅requested+observed flags guard，不是kernel证书。

每图原始数组字节依次为150528/602112/602112/197376/768/768/768。实际7图gzip上界10892112B，含全部metadata硬上限和reserve为36057936B；最大8图为12448128+16777216+8388608=37613952B，小于67108864B。实际7图native模块IO为5607168B。JSON是总硬限而非每文件可重复使用的额度；超过即保partial并拒绝，不推断可删字段。

继承CEM H3的192predict/54012samples/68AE/90028AE tokens另列，非本桥新费。与1902/common-resource预算、方法不同，不作公平排名/官方复现/科学放行。缺失cp15实际score/native_success保持null；原已付模型forecast可存在。

## MAIN commands

在项目根目录执行；只有MAIN可run。以下shell变量仅缩写绝对路径，prepare参数固定实际authority。

```bash
RGB_PY=/home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python
RGB_MOD=experiments.spectral_reachability.agent_lewm_bounded_cem_h3_rgb_20261029_v1
RGB_RES=/home/gbk/code/LeWorldModel/le-wm-main/results
cd /home/gbk/code/LeWorldModel/le-wm-main
"$RGB_PY" -m "$RGB_MOD" prepare --output "$RGB_RES/agent_lewm_bounded_cem_h3_rgb_20261029_v1_prepare_MAIN01.json" \
 --physical-prepared "$RGB_RES/agent_lewm_bounded_cem_h3_physical_20261029_v1_prepare_MAIN01.json" \
 --physical-prepare-sha d17ffc349dedd7d310764f8cb6d7c4c615c9d70bc2e85ee85cd3a5991db431a8 \
 --physical-output "$RGB_RES/agent_lewm_bounded_cem_h3_physical_20261029_v1_physics_MAIN01" \
 --physical-report-sha 8705d1877a37d81b835b04935aadaf42af4a78a4e1fb4b9524fadb2f516f59d1 \
 --physical-completion-sha c9310caa342d89637e7f49cd58c3258e2279bb75c027af0ad345183dfe81a63a \
 --physical-receipt "$RGB_RES/agent_lewm_bounded_cem_h3_physical_20261029_v1_consume_MAIN01.json" \
 --physical-receipt-sha 1e69a2fe082af3c5da32ec28ac8ba10b2375b5a593fa0bf69a60367d7cbc9b86
"$RGB_PY" -m "$RGB_MOD" verify --prepared "$RGB_RES/agent_lewm_bounded_cem_h3_rgb_20261029_v1_prepare_MAIN01.json" --prepare-sha ACTUAL_NEW_PREPARE_SHA
"$RGB_PY" -m "$RGB_MOD" run --prepared "$RGB_RES/agent_lewm_bounded_cem_h3_rgb_20261029_v1_prepare_MAIN01.json" --prepare-sha ACTUAL_NEW_PREPARE_SHA --output "$RGB_RES/agent_lewm_bounded_cem_h3_rgb_20261029_v1_cuda_MAIN01" --authorize-host-cuda
"$RGB_PY" -m "$RGB_MOD" consume --output "$RGB_RES/agent_lewm_bounded_cem_h3_rgb_20261029_v1_cuda_MAIN01" --report-sha ACTUAL_NEW_REPORT_SHA --completion-sha ACTUAL_NEW_COMPLETION_SHA --receipt "$RGB_RES/agent_lewm_bounded_cem_h3_rgb_20261029_v1_consume_MAIN01.json"
"$RGB_PY" -m pytest -q /home/gbk/code/LeWorldModel/le-wm-main/tests/test_agent_lewm_bounded_cem_h3_rgb_20261029_v1.py
```

占位SHA必须由MAIN实际产物替换，不预填未来成功。worker仅NumPy/fake-hooks和真实临时磁盘，不运行上述生产命令；fixtures不构成模型或物理证据。

## RAM validation

独占tests首轮33passed/51.25s：真实临时盘authority→build/write/verify、Store→49数组/71事件→completion→全saved consumer；另含最大8图、不压缩随机数组运输界、全16MiB metadata quota、错误time/root/goal/normalizer/forecast身份、hidden197/256拒绝、禁止predict/AE/AD、native IO/CLS/event篡改、来源/receipt/NPY/orphan篡改、velocity256仍native_success=true。测试中的物理与神经对象均为显式fake，不加载torch/环境。未执行任何实际prepare/encoding/consume生产。
