# 六帧 actual RGB / native NPY 独立 saved 审计

NumPy/PIL/stdlib only，不导入producer、producer.analyze、任何模型/物理模块。
不执行encoder/AD/physics。所有旧sealed文件只读，新增独立source/tests/doc及小JSON结果。

## 固定输入

encoder report `results/p1_line_search_gn_rgb_cuda_20261027_v1_cuda_main01/report.json`
SHA `e1ec726ec0d54ef7347030abdaa10a2bdf4f5d4767e327970a2a29c819723dcd`；
encoder prepare SHA `cf384f59734fd7ae247dd3d0bf82414d26b5d000fb5f7851eb36cca07c625e0f`。
physical report SHA `3a41fe83f375e5bb3f390e9f3a0e8e17dd70c2d49a7ebe3a99079f5774de85c1`；
physical prepare02 SHA `f36f915ccfc2687da056810d1241da433190eff430bb54fa7b7b35a6af3818d5`。

验证exact文件census、report/prepare/source/input/model-file/6NPY/6PNG/analysis/journal SHA，
physical journal逐事件index/previous/hashchain、两个branch_closed及6个checkpoint_saved。
读完计算后再次复验全部源SHA，防TOCTOU。JSON拒绝duplicate key；gzip要求完整EOF无尾随垃圾，
压缩/展开各32MiB上限，模型权重仅流式hash，不复制。

NPY逐份以allow_pickle=False从原始字节解析：严格float32 shape192、finite、无尾随bytes，
不偷偷cast后验证。每份文件SHA与本帧report/artifact绑定；6份全部保留，不按相同像素或值去重。
PNG从原始字节独立解码，要求PNG/RGB/224x224/uint8，文件与像素SHA分开算。

从physical prepare02的两条plan直接读取原始task gzip及本次line-search final_prediction gzip，
复核task_id完整原JSON编码、模型/config、normalizer、root原始动作history、fullH2原始动作、
两packet latent/action history内容与layout、whole/segmented replay、selected prediction hash。
physical逐步10命令须与对应selected命令一致，step5/10 checkpoint state来自同branch trace。
用原task Q/goal逐项exact比对prepare中两根root_metrics，不使用别根Q、不拟合新Q。

独立重建帧对照：step0 = root_context末latent，step5/10 = 本次final_prediction两context末latent。
对每帧raw NPY先提升float64做纯算术：
`actual_Q=mean((Q@(z-goal))²)`，`predicted_Q=mean((Q@(pred-goal))²)`，
`native_error=mean((z-pred)²)`。六行与producer保存analysis逐字段exact匹配，未调用其分析函数。

36条image事件按固定六帧顺序完整重建：frame_attempt、encoder attempt/return、projector
attempt/return、frame_saved；唯一连续module index、无额外/缺失/重试/被禁止事件。
原始事件账重算encoder6/6、projector6/6。新predict/actionencoder/AD/physics均0，
旧planner156/156、physical20公共steps及source acquisition是inherited，不能宣称全pipeline等成本。
其中physical原始journal另按reset/render/step的attempted/returned事件逐项计数，独立复核4/22/20；
不将这些旧操作计入本审计或本次image producer的新执行成本。

## 实际数值与界限

| root | actual Q step0 → 5 → 10 | predicted Q step10 | native MSE step5 / 10 |
|---|---|---|---|
| 0 | 1.361121737 → 1.310690230 → 1.268256082 | 1.140511538 | .006957885 / .014401503 |
| 1 | 2.077293442 → 2.040062159 → 2.028360164 | 1.920317098 | .018224319 / .067265590 |

两根step0 native预测误差均0；两根物体xy/angle相对step0均未变，末步recorded official success均false。
原Q真实编码代价下降，不等于物体达到目标；不能回填success，也不能由此断言唯一metric失败根因。
这是两根原calibration descriptive结果，不是独立样本/物理改善/科学准入。
模型before/after tensor hash仅来自真实CUDA receipt；审计重新hash模型文件，不重新load/执行模型。
同样不独立证明encoder数值正确或AD正确，不能把saved consistency当新的neural运行证据。

tests包括actual完整重算、26种NPY/PNG/goal/Q/预测/事件/预算/physical链反例、dtype/shape/CRC边界、
source drift/TOCTOU与独占输出。mock反例只验证拒错能力。输出<256KiB且保留200MiB floor，
只存小摘要与输入SHA，不复制NPY/PNG/模型；原始证据保持原样。

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_line_search_gn_rgb_saved6_audit_20261027_v1.py
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_line_search_gn_rgb_saved6_audit_20261027_v1 --output results/p1_line_search_gn_rgb_saved6_audit_20261027_v1_saved01.json
```
