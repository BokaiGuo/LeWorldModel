# MAIN：同144预测的真实执行与独立物理指标复算

本分析消费已封存的新四分支执行，不生成动作，不加载模型或环境，不重新执行物理。
调用原 adapter.consume 完整复核 prepare/source/model/action 闭包、所有 PNG、journal hashchain、
操作 intent/return、checkpoint 与真实返回，再把完整原生回执保存到分析文件，绝非只保存成功摘要。

固定输入 report SHA：`50240e6f5e9bca90a5d8734cb5c41a5e06d61337b20dcd3d7fb9693fea45ebcb`。
实际分析：`results/main_p2_equal144_physical_metrics_20261028_v1_MAIN01.json`，
SHA：`6b2e0301614d163cffba3d8eb2f1e522ebf6304c86d8c589eda3fe0469bfd413`。
MAIN analyze 和 verify 均实际 exit0；新增7项 RAM 测试通过。

独立 NumPy 算术直接从返回的 state7/goal7 重算，不调用原 metric：native7 距离与 reward、
jointXY、agentXY、objectXY、velocity、原始角差及 native success。
保留 native min(abs(angle),2pi-abs(angle)) 的行为，包括超过2pi可能为负的已知边界；
不把改良 wrapped-angle 公式偷偷替换进历史原生合同。未据此改成功阈值。
另外保存 native7² = jointXY² + raw-angle² + velocity² 的分项。

40次 plan 返回的原始动作列表、当前 goal、reward、terminated 和 endpoint 精确一致；
40次 warmup 在新目标绑定之前发生，不能把其 reward 拿新 goal 重算，故明确不作该声明。
四个分支全部10步、0成功、0早停；没有删掉失败、换绑历史标签或填补缺失。

两已知根中 spectral−beam 的 jointXY 终误差分别 −18.99630562、−7.79529678；
objectXY 分别 −6.86475687、−6.53937843。
native7 距离分别 −11.00180524、+52.75107445，第二根 velocity 误差增加93.24424638。
这显示混合7维距离不等同于位置目标，也不把 objectXY 改善替代原生成功。

每方法每根144预测调用和144样本是已审 CUDA 账目；谱方法额外付 Jacobian/VJP/SVD/solve，
不是等 FLOPs 或等总资源。仅两个已知根，不是新独立确认、不是 Full D2P 对比，
更不是 P1/C2 或 P2 组合机制成立。所有 admission 保持 false。

CLI（inner cwd，使用项目解释器）：

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.main_p2_equal144_physical_metrics_20261028_v1 analyze --output results/EXCLUSIVE_NEW_ANALYSIS.json
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.main_p2_equal144_physical_metrics_20261028_v1 verify --output results/main_p2_equal144_physical_metrics_20261028_v1_MAIN01.json --sha256 6b2e0301614d163cffba3d8eb2f1e522ebf6304c86d8c589eda3fe0469bfd413
```
