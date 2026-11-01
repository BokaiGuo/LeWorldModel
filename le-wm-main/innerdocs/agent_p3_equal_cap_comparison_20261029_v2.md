# P3 saved比较v2：真实MAIN reader authority适配

只新增本v2 source/tests/doc。v1比较器和equal-cap producer三件套保持冻结，不改source manifest/结果、不重跑CUDA/NN/physics/fullstrict。v1因要求原producer.consume schema、slot_statuses和arms中model_plan_available/depth/width，不兼容MAIN真实新receipt。v2不重标receipt、不补造字段、不patch globals；直接校验`main_p3_equal_cap_saved_consumer_20261029_v1.consume`真实ABI。

## Authority及最小delta

完整strict已由MAIN执行。receipt路径`results/main_p3_equal_cap_saved_consumer_20261029_v1_MAIN01.json`，实际SHA `2ca340e108cd0c82fea6f9378a95b47f51aad6c49599f9f6cf113d8b32b3cb82`，status `saved_equal_cap_verified_not_admitted`。本worker仅核对保存ABI/字节，不生产分析，不把RAM测试当实际结论。

新receipt明确原reader64MiB单行限制→原producer已固定256MiBreserve，总cap1GiB；完整原数学/资源/计时验证、原producer未改、NN/physics新调用0。v2逐字段严格核对compatibility；验证inherited_producer_sha256和consumer_sources实际source/tests/doc SHA，前后重验，不能把兼容层冒称原预运行manifest一部分。不import MAIN reader，不读journal，不调用consume/verify/build。

读取两套prepare/report/receipt六个外部实际SHA。旧侧继续用冻结v1.validate_bundle；新侧自有validate_new。四槽身份/576分母、prepare/report路径与SHA、common_budget、完整27轴资源、plan_available/timing/backend/unknown同真实report逐项join；新receipt不含depth/width/core_model_plan，直接从已认证report提取并如实说明，绝不伪造receipt字段。缺失arm继续144 null、不填零。

模型选择/H/depth/width/动作bits、全部ordinal576行和27轴差值，完全复用冻结v1的独立描述函数；未使用producer算术。保留core plan/实际release区分、unknown、partial、全部R拒绝。ordinal不是同动作见证。common cap≠actual matching≠equalFLOPs，无预算效益推断，P1/P3/C3/freshness均不签。旧scope/新同步timer不可混同。v1 helper SHA固定 `47a10c014a2bcd04d954b77d434789ee4c3daacf457c547adcb13a8fd6d4d3d4`。

## MAIN实际CLI

cwd `/home/gbk/code/LeWorldModel/le-wm-main`。下面SHA已只读核对；由MAIN决定运行，worker未执行。

```bash
.venv/bin/python -m pytest -q tests/test_agent_p3_equal_cap_comparison_20261029_v2.py tests/test_agent_p3_equal_cap_comparison_20261029_v1.py
.venv/bin/python -m experiments.spectral_reachability.agent_p3_equal_cap_comparison_20261029_v2 analyze \
 --old-prepare results/agent_p3_parent_native_cuda_20261029_v1_prepare_MAIN01.json \
 --old-prepare-sha deb87295f3984929f4734ece0b73770220ec59ac3b621a97e85e2bade60e3599 \
 --old-report results/agent_p3_parent_native_cuda_20261029_v1_cuda_MAIN01/report.json \
 --old-report-sha 75f4c162aaa75223dfd8523381e7614c8a35637afd52278df8a80d6affa34e40 \
 --old-receipt results/agent_p3_parent_native_cuda_20261029_v1_consume_MAIN01.json \
 --old-receipt-sha 622d1cdaf8d1ed69c4e78f8d1f5c7c6114285db6121f0021fbbaf6573a8a76ad \
 --new-prepare results/agent_p3_equal_cap_parent_20261029_v1_prepare_MAIN01.json \
 --new-prepare-sha f12e900b243d0ae775cccd37b06a2228576ef1b22003aebf3cf001f755004ca9 \
 --new-report results/agent_p3_equal_cap_parent_20261029_v1_cuda_MAIN01/report.json \
 --new-report-sha 8c65d3c5c54ba447b279fd0ad938edfab6eb42f7e1b3dbb4a1624987153cc58d \
 --new-receipt results/main_p3_equal_cap_saved_consumer_20261029_v1_MAIN01.json \
 --new-receipt-sha 2ca340e108cd0c82fea6f9378a95b47f51aad6c49599f9f6cf113d8b32b3cb82 \
 --output results/agent_p3_equal_cap_comparison_20261029_v2_MAIN01.json
.venv/bin/python -m experiments.spectral_reachability.agent_p3_equal_cap_comparison_20261029_v2 verify \
 --summary results/agent_p3_equal_cap_comparison_20261029_v2_MAIN01.json --summary-sha ACTUAL_SUMMARY_SHA
```

输出仅新v2前缀路径/32MiB上限/xb+fsync。verify完整重建summary，不重跑fullstrict。

## RAM验收与剩余

联合42 passed/0.71s（v2 19+冻结v1 23）。真实新receipt形状fixture不含旧字段，验证旧v1拒绝/新v2接受；64→256compatibility、source身份、原producer身份、预算/release/timing/backend/分母/unknown/资源/flags篡改均拒绝；partial保留144null；实际writer→CLI verify→全量篡改拒绝/拒绝覆盖；import不加载torch/producer/MAIN reader。fixture只为测试，不代表真实实验值。剩余MAIN审读后执行analyze/verify；不需要新CUDA或物理。无科学gate变化。
