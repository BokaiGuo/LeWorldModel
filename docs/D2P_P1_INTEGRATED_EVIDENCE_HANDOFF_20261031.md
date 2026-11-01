# P1 共同后缀与真实反馈结果的整合

完整 H10 面板消费、生成固定 588 行的配对表之后，
`experiments/spectral_reachability/p1_response_integrated_evidence_20261031.py`
将它与已验收的真实中点反馈结果关联。共同后缀每个 anchor 的
42 行与各自规划反馈的每 anchor 一条记录分开保存，避免把同一
规划结果重复 42 次作为独立样本。两条 cp5 前删失保持原状态；
anchor 3 在中点重规划原物理轨迹不完整，单独标记缺失，不补值。

整合器核对 14 个可用 anchor 和 42 个探针的完整顺序、反馈的
13 条身份、根／条件／角色及来源哈希；错配和缺探针拒绝。
测试用真实固定 anchor roster 构造结构样例，验证删失与拒绝逻辑，
2 项通过；这不是已完成的 H10 实验或科学结果。

H10 全部完成、消费并生成配对表后，按其实际 SHA 运行：

```bash
.venv/bin/python -m experiments.spectral_reachability.p1_response_integrated_evidence_20261031 \
  --panel /home/gbk/code/LeWorldModel/le-wm-main/results/p1_response_h10_paired_table_20261031_RUN01.json \
  --panel-sha ACTUAL_SHA256 \
  --output /home/gbk/code/LeWorldModel/le-wm-main/results/p1_response_integrated_evidence_20261031_RUN01.json
```

命令在 `le-wm-main` 内执行。共同后缀的物理效果不等于同预算
可规划能力；真实反馈前缀和历史控制仍是已知根、总预算未
认证，`model_selected` 也不是认证部署赢家。均不能单独打开 P1。
