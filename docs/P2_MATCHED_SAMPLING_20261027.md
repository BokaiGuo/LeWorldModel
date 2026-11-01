# P2 H2 CEM/random：仅预测预算匹配校准

独占新模块 `p2_matched_sampling_20261027_v1.py`、对应测试及本文。
不修改任何旧源码/测试/报告/封存，不写中央文档，不训练/readout、不选根、不调
旧阶段阈值，不删除/commit/push。本任务不启动CUDA。P1未通过，后续仅工程校准。

## 动作空间和比较对象

引用 `results/pusht_p2_matched_composition4_v1_20261027_cuda/p2_matched_report.json`，
核验原始 compilation、输入绑定、源码镜像当前/归档字节和原全部四根动作见证。
每组复用原 root RGB/goal RGB、完整 emb3+between-state actions2、Q、raw action
单位、normalizer SHA、五步 packet。原16命令×16命令是离散256有序双段。

新两组则在 raw live command [-1,1]^4 连续盒采样，float32，每个二维命令重复5步。
不在 latent/normalised-action 空间采样，不用 atanh 参数，不额外tanh改变分布。
因此与旧P2是**同动作坐标/合法盒/长度/预测预算，不同候选空间和proposal prior**；
不冒充离散bank内 CEM，也不据此孤立归因组合结构优势。不声称官方CEM/SRA复现。

## 冻结调度

- 每root每arm恰40个完整双段（80 planning predict calls/samples），串行batch1。
- CEM：4代×10候选，elite2，raw Gaussian初始mean0/std0.5，clip[-1,1]，
  每代首候选为当前mean；population std ddof0，下界0.05，无平滑。
- random：首候选zero，后39个raw盒均匀候选。zero包含在40内，不追加候选。
- CEM首个mean也为zero。每arm固定同一逐root seed（20261027至20261030），
  独立 NumPy generator，不根据outcomes改seed、rank或根。
- 保存全部候选（含重复mean）、实际raw动作、cost和失败；best-ever稳定首次tie。
  CEM更新不是训练/readout，只是推理期proposal search。
- winner独立逐段2次，再从原root完整重放2次，检查search/segmented/full终态精确一致，
  共4 evaluation samples。封存v2 receipt、原始edges、完整上下文及六项provenance。
  kind=progress只是见证类型，不是物理进展。

正常四根两组总672次predict，恰与旧P2每arm80+4相同。实际PredictBudget(84)
计数/限额，失败调用不退款，actual planning/evaluation分别核验；action encoder
pre-hook记录calls/samples及processed context tokens，非新预测步。正常每arm
84 encoder calls、252 context tokens。候选/metric/replay失败全部保留，40候选不
替补，失配禁止物理执行，不padding凑预算。fatal加载/环境错误可能只留started/
journal，不伪造最终报告；所有已写产物保留。全arm都继续处理可恢复候选异常。

只匹配predict calls/samples，非总算力：上游encoder/bank构建、FLOPs、同walltime
未匹配，CEM CPU更新额外成本包含arm实际耗时。Q仍可能奖励物体静止时agent运动，
不是task metric修复，不输出效益准入。

## 命令：先prepare，主线程再CUDA

cwd `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m experiments.spectral_reachability.p2_matched_sampling_20261027_v1 prepare \
  --paired-report results/pusht_p2_matched_composition4_v1_20261027_cuda/p2_matched_report.json \
  --compilation results/pusht_d2p_metric_program_preflight4_v1_20261026_cuda/report.json \
  --policy /home/gbk/.stable_worldmodel/checkpoints/pusht/lewm \
  --output results/p2_matched_sampling_20261027_v1_cuda_v1

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m experiments.spectral_reachability.p2_matched_sampling_20261027_v1 cuda \
  --output results/p2_matched_sampling_20261027_v1_cuda_v1
```

prepare只核验并独占创建输出/源码seal，不分配CUDA、不加载神经模型；cuda只消费
同一prepared目录，先检查全部seal/inputs，再分配主机CUDA，不存在CPU fallback。
重复prepare/cuda被独占文件拒绝，不能覆盖续跑。保持源码稳定后再prepare。
输出protocol/prepared/started、逐候选jsonl、逐root-arm见证及report，预计<20MB
（估计，非实测）；无图片、权重或数据集复制。调度预留单GPU4GiB/10分钟为保守
估计，非实测承诺；实际峰值allocated显存输出报告。

## 物理driver消费

旧 `run_p2_matched_physical_20261027.py` 写死composition/flat_search和bank索引，
不能原样执行新结果。**不改旧driver，不伪造旧arm名或索引。** 主线程新增版本应：

1. 调用本模块 `load_prepared(output)` 校验旧/新输入与所有source seals。
2. 读取新report，调用 `selected_actions(report, protocol, compilation)`，返回
   `{(root_sha256, "cem"/"random"): ndarray[10,2]}`。
3. 外部封存report/protocol/输入hash后，以旧共享RGB guard/实际executor执行全部
   八分支，保留中止、不重选、不补零；完整root/goal图像仍须执行前真实核验。

新consumer重算receipt/edge digest、完整context/domain/provenance、40候选中best、
raw float32动作映射、终态Q评分；它只认证声明/保存内容，不冒充独立模型重放。
真实CUDA独立重放由runner完成。hash绑定不证明真实性，source验证不代替物理结果。

## 测试

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_p2_matched_sampling_20261027_v1.py
```

覆盖固定40/H2/80+4、无额外mean、同seed确定性、失败记账、动作域、重放及全部
八动作consumer/篡改拒绝。真实输入prepare测试仅读封存产物并运行小型NumPy计算，
不加载模型或神经CPU推理。合成测试不是P1/P2或收益证据。
