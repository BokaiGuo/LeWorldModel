# Full30反馈三层保存结果汇总

独占saved-only分析，固定known2、4base×O/I/F×7cp=84位置；固定84同base同时间逐对。不是新模型评估、物理执行或上游fullstrict。所有科学准入、P1/P2、C3、等FLOPs声明均false。

## 权威与信任界

CLI输入一个显式path/SHA manifest，六类权威为新IF24完整RGB receipt、原O28完整RGB receipt、feedback prepare/完整receipt、独立model summary、独立native-v2 summary。MAIN已经完整消费的receipt为外部接受证据，不伪称本工具再次重建其全闭包。工具重新hash所有实际读取的JSON与52份小型latent NPY/gzip；重新连接prepare/report/receipt、原native及4job native源、原始任务、raw30位串、cp时间、goal、模型witness、执行digest。独立stdlib fsum计算向量Q、预测误差和native轴，与独立summary核对。不会导入producer算术，也不会读模型/PNG进行新推理。

输入SHA不是未来默认值；未齐、错源、篡改均拒绝。`verify`只完整重建本工具的保存结果，不调用任何上游consumer。外部完整receipt的真实性由MAIN实际SHA接受，不把仅指针当成本地全artifact审计。

## 数学与缺项

Q为192维原identity任务的`fsum((z-g)^2)/192`，严格float32 literal输入、float64差/平方/求和。native7、jointXY、objectXY、velocity平方误差及wrapped angle独立重算。native原始成功角项保留原`min(abs(delta),2*pi-abs(delta))`，不悄然更换为wrapped角；二者分别报告。

O cp0无预测；O其余六cp使用原模型context。I/F cp≤15只有继承的actual编码及另列原O前缀模型Q，不冒充新I/F预测；cp20/25/30必须用各自新24图编码，不因同动作去重。无checkpoint的latent/native项为null，绝不padding。每个跨policy modelQ差仅在模型输入context精确相同时定义，否则null；actualQ/native的同时间差独立保留，不能把不同I/F输入直接作模型Q因果增益。

资源各轴分列：新feedback216predict/8J/1536VJP/24AD-primal，IF24编码与O28继承编码、native继承计数；未知FLOPs为null。分析自身0神经/0物理/0fullstrict。所有ties保持精确zero，无epsilon、无按结果筛选。

动作一致性分别提供完整raw30、截至cp的prefix和当前raw5区间位串，不把只同当前区间当作同一整段动作历史。事件终态control与每cp实际到达分列。状态另核snapshot的原生float64位串。保留全部12策略槽，不能把相同动作或Q ties退款/去重。

## MAIN实际输入构造与CLI

所有路径必须canonical绝对路径。MAIN用apply_patch创建
`/home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_feedback_three_layer_summary_20261029_v1_inputs_MAIN01.json`，完整内容如下（worker未创建生产manifest）：

```json
{
  "schema": "agent_p2_feedback_three_layer_summary_20261029_v1.inputs",
  "trust": "MAIN_complete_external_receipt_SHAs_not_rebuilt",
  "inputs": {
    "if_rgb": {
      "path": "/home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_full30_feedback_rgb_20261029_v1_consume_MAIN01.json",
      "sha256": "972688100781340072a62fa5cfae353005a746ec24d713df4af293f0846834ba"
    },
    "O_rgb": {
      "path": "/home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_two_program_full30_rgb_encode_20261029_v1_consume_MAIN01.json",
      "sha256": "42164a225310c39ff83e398f55501e36e8db9876d5b52dbc697d031e31468166"
    },
    "feedback_prepare": {
      "path": "/home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_full30_control15_feedback_20261029_v2_prepare_MAIN01.json",
      "sha256": "498b561855d7859681a18ed47e0157908a146a796145b929b5436a1e73b79630"
    },
    "feedback_receipt": {
      "path": "/home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_full30_control15_feedback_20261029_v2_consume_MAIN01.json",
      "sha256": "4e7d5e1f6b4b03b28b68eb8a31016211f7d1e0ff3115671923b6df2e97f3d893"
    },
    "model_summary": {
      "path": "/home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_full30_control15_feedback_summary_20261029_v1_MAIN01.json",
      "sha256": "f2559fe07093b084ca0529a0cc458048183e5d0be7f2a7053a8ec451359586e6"
    },
    "native_summary": {
      "path": "/home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_full30_feedback_native_summary_20261029_v2_MAIN01.json",
      "sha256": "bcc292165eb1dffc84a7db5b1cb478be0455f64256a4061ed8c270bc2eacc0d6"
    }
  }
}
```

创建后读取实际manifest SHA：

```bash
sha256sum /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_feedback_three_layer_summary_20261029_v1_inputs_MAIN01.json
```

将实际打印SHA代入INPUT_SHA；OUTPUT_SHA为analyze实际返回值，不伪填：

```bash
/home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python /home/gbk/code/LeWorldModel/le-wm-main/experiments/spectral_reachability/agent_p2_feedback_three_layer_summary_20261029_v1.py analyze --inputs /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_feedback_three_layer_summary_20261029_v1_inputs_MAIN01.json --inputs-sha INPUT_SHA --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_feedback_three_layer_summary_20261029_v1_MAIN01.json
/home/gbk/code/LeWorldModel/le-wm-main/.venv/bin/python /home/gbk/code/LeWorldModel/le-wm-main/experiments/spectral_reachability/agent_p2_feedback_three_layer_summary_20261029_v1.py verify --summary /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_feedback_three_layer_summary_20261029_v1_MAIN01.json --summary-sha OUTPUT_SHA
```

## 测试边界

临时磁盘测试从已存在的保存ABI构造隔离fixture，重新绑定临时JSON与52份小数组引用；没有写results、没有调用生产analyze/fullstrict/NN/physics。该测试验证接线和数值重建，不是MAIN实际结果分析产物。最终实际输出仍由MAIN执行。

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
.venv/bin/python -m pytest -q tests/test_agent_p2_feedback_three_layer_summary_20261029_v1.py
```

整链实际读取67份有界文件（15 JSON＋52 gzip NPY），保存所有本地读取SHA，并在算术后再次rehash；原模型权重、PNG、完整native/journal、completion的全量验证继承外部MAIN receipt，不在此重复。JSON每份及输出均16MiB cap；小数组每份4096bytes cap。首轮测试唯一失败为fixture误把文件数写成68，修正为67；未改变消费要求，也不算科学负结果。
