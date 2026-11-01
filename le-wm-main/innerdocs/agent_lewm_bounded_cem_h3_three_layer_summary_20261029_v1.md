# CEM own8三层独立saved分析

仅known2固定cp0/5/10/15共8槽（绝对步10/15/20/25），实际7编码，root0cp15永久null。独立复算model192→actualencoded192→native7，不作与1902/GRASP/Hi等异预算方法的胜负比较。源/测/文为全新独占文件，不修改RGB或物理源。不运行任何编码、predict、physics、继承fullstrict。

## 接口与算术

复用已冻结三层summary的`summary.rows/pairs/denominator/available/comparison_census`形式，而非调用其producer数学。以MAIN完整SHA receipt为信任起点，输入四份显式JSON：RGB完整receipt及prepare、原native完整receipt及prepare。核完整8行、task/selection/goal/原动作history位串、root/cp/绝对时刻/早停、原PNG checkpoint和snapshot state连接；重新读7份小型gzip NPY末192向量，逐位核receipt实际向量。其余原生事件、PNG、模型加载、全IO、seal的验证继承MAIN receipt，不自称重复认证全链。只有4份JSON+7数组+manifest是实际读取输入，另绑定自身三文件。

每槽由literal f32向量经float64逐轴差平方、stdlib `math.fsum /192`重算model_Q/actual_Q、预测误差、Q差。原NumPy reduction独立本地重算并严格核源字段；fsum与NumPy差显式报告，不强制bitexact、不加容差。anchor model_Q是null，不拿零当预测；缺失cp15仍有原付forecast，但无actual/native值。

native由实际state7/goal7逐轴重算：agentXY/objectXY/jointXY、raw angle、velocity平方及native7距离。原生success严格是jointXY norm<20与原始`min(abs(angle),2pi-abs(angle))<pi/9`；保留该原生角项可能为负的语义，wrapped角只诊断。源NumPy norm成功/reward与fsum距离分列；速度256可以native_success=true，不以reward/Q代success。全部7轴贡献及原native指标均保留。

固定每根全部6个无向时间组合（方向later−earlier），共12对，不挑终点或响应。model↔actual及二者对各native距离/误差轴均报告strict sign/tie/null/reversal，绝不用乘积判断符号或设置epsilon。cp0 model项和root0cp15 actual/native配对按null计数。时间配对是轨迹内描述，不是同时间方法对照、不认证因果。无root间排序，不拟合gate，不放宽科学门。

资源分轴复制原CEM、RGB、physical完整向量，未知FLOPs=null。分析自身0NN/0physics/0fullstrict。输入/输出JSON每份16MiB上限，每份小数组4096B上限，解压严格896B，完整输出独占/fsync。verify整份确定性重建本分析，不执行上游consumer。

## MAIN inputs / CLI

RGB实际receipt现由MAIN提供，四权威已固定；不执行worker生产分析。MAIN用apply_patch建立新manifest，四个path均canonical绝对路径：

```json
{
  "schema": "agent_lewm_bounded_cem_h3_three_layer_summary_20261029_v1.inputs",
  "trust": "MAIN_complete_external_receipt_SHAs_not_rebuilt",
  "inputs": {
    "rgb_receipt": {"path": "/home/gbk/code/LeWorldModel/le-wm-main/results/agent_lewm_bounded_cem_h3_rgb_20261029_v1_consume_MAIN01.json", "sha256": "0e5d8736e07ffd07417b39c84471ebd68682e4bfb448e4459517d9a5fa3234fa"},
    "rgb_prepare": {"path": "/home/gbk/code/LeWorldModel/le-wm-main/results/agent_lewm_bounded_cem_h3_rgb_20261029_v1_prepare_MAIN01.json", "sha256": "c40be7df2f08872bd2d698f578862d671fa9ed9fa8a550f23d32471d711dd145"},
    "physical_receipt": {"path": "/home/gbk/code/LeWorldModel/le-wm-main/results/agent_lewm_bounded_cem_h3_physical_20261029_v1_consume_MAIN01.json", "sha256": "1e69a2fe082af3c5da32ec28ac8ba10b2375b5a593fa0bf69a60367d7cbc9b86"},
    "physical_prepare": {"path": "/home/gbk/code/LeWorldModel/le-wm-main/results/agent_lewm_bounded_cem_h3_physical_20261029_v1_prepare_MAIN01.json", "sha256": "d17ffc349dedd7d310764f8cb6d7c4c615c9d70bc2e85ee85cd3a5991db431a8"}
  }
}
```

RGB owner source固定SHA e47c739def9cf170513494eb4bb6244c187f9e2fe4b5ab6be1983a6c3bee9b57，不接其它schema。

RGB report SHA5695f9375b7e0a10028d7149da8bf406e9702570e8fe7f8680b80647ab29e8a2、completion SHA9d9a1604deff303a8d54f53256aca7de4fc587444bf8907d87103afb243f4be4由receipt整份绑定核对，不另读report或重跑消费。

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
/home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python -m experiments.spectral_reachability.agent_lewm_bounded_cem_h3_three_layer_summary_20261029_v1 analyze --inputs /home/gbk/code/LeWorldModel/le-wm-main/results/agent_lewm_bounded_cem_h3_three_layer_summary_20261029_v1_inputs_MAIN01.json --inputs-sha ACTUAL_INPUT_MANIFEST_SHA --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_lewm_bounded_cem_h3_three_layer_summary_20261029_v1_MAIN01.json
/home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python -m experiments.spectral_reachability.agent_lewm_bounded_cem_h3_three_layer_summary_20261029_v1 verify --summary /home/gbk/code/LeWorldModel/le-wm-main/results/agent_lewm_bounded_cem_h3_three_layer_summary_20261029_v1_MAIN01.json --summary-sha ACTUAL_OUTPUT_SHA
/home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python -m pytest -q tests/test_agent_lewm_bounded_cem_h3_three_layer_summary_20261029_v1.py
```

RAM fixtures为独立合成全8槽及7份真实临时盘NPY/完整JSON文件，不加载任何producer/NN/physics模块。测试不构成实际模型/任务结果；MAIN可用已交实际receipt生产分析。该描述性分析已知早停事实，不声称事前独立确认或预注册科学实验。

最终独占28tests与冻结full30三层summary的38tests联合66passed/0.73s。包括临时盘全CLI写出与wholeverify、源/recipe/数组/sha错接拒绝、重封错report/completion/prepare拒绝、全7轴已知数值校核、缺槽summary篡改拒绝。此实现按experiment-code的来源/计量/日志要求组织；用户限定saved-only，未生成训练、选择器或实验图。
