# Latent State Spectral Transfer：Overleaf 与复现材料包

这是给合作者使用的论文包。它分成两个部分：

- `overleaf/`：可以直接上传到 Overleaf 的 LaTeX 论文源文件和图。
- `reproducibility/`：论文中数字对应的结果 JSON/CSV、实验脚本、数据说明和 provenance。

## 一、如何上传到 Overleaf

1. 打开 Overleaf，新建一个空项目。
2. 将 `overleaf/` 目录里的**所有文件和 `figures/` 文件夹内容**上传到项目根目录；不要把 `overleaf` 文件夹本身再套一层。
3. 将主文件设为 `main.tex`。
4. 编译主文。当前主文 PDF 共 13 页，其中后部为参考文献；正文在 ICLR 的 9 页限制内。
5. 如需单独编译补充材料，将主文件切换为 `supplement.tex`；当前补充材料为 29 页，包含 state-conditioned held-out/permutation-null readout、跨环境 future-only action、Wall split、episode derangement、rank/ridge 稳定性、alignment-intervention boundary 结果、exact action-sequence composition、learned pixel-to-latent 边界实验、跨来源 latent-dynamics transport、action-conditioned operator algebra 与 normalization 对照，以及 leakage-controlled forward-readout 的 multi-horizon profile 与 physical-target negative control。

Overleaf 目录中已经包含 `main.tex`、`supplement.tex`、`references.bib`、ICLR 样式文件，以及主文实际引用的三张 PDF 图和新增的两张 PNG 结果图。`figure_sources/` 中的 SVG/PPTX 是方便合作者修改图的源文件，不参与 LaTeX 编译。

## 二、论文到底在研究什么

论文研究的不是“哪个模型分数更高”，而是一个更基础的识别问题：

> 如果两个预测算子的特征值完全一样，我们能不能仅凭这组特征值判断它们如何传输 latent information？

答案是否定的。传输行为还取决于预测算子的特征子空间，是否和 latent covariance 的高方差方向对齐。论文用一个有限维、可精确推导的 isospectral family 证明这一点，再把同一套测量量应用到 LeWorldModel、I-JEPA、JEPA-WM 和 DINO-WM 的冻结表示或发布 checkpoint 上。

主线可以用四句话复述：

1. **识别失败**：相同 predictor spectrum 不保证相同 transfer。
2. **对齐机制**：covariance--predictor commutator 随 eigenspace rotation 呈精确的 `|sin(2θ)|` 规律。
3. **边界证据**：spectrum-preserving null、各向异性控制，以及不同 learned checkpoint 的 held-out 测量显示该现象可测、但依赖环境和 horizon。
4. **measurement consequence**：任何 spectral-transfer 结论都应额外报告 shared-eigenbasis 假设，或报告 commutator、principal angle、spectral-projector overlap 等 alignment statistic。

## 三、数据包中每类数据的意义

`reproducibility/results/` 里的文件是已经算好的论文结果。它们不是杂乱的日志，而是把每个论文结论对应到可读的数值结果：

| 目录 | 给合作者的直观解释 | 论文作用 |
|---|---|---|
| `theory_operator_alignment_sweep/` | 固定 predictor 的特征值，只旋转特征向量；观察 commutator 和 transfer readout 如何变化 | 主定理的完整角度扫描 |
| `theory_operator_alignment_null_sweep/` | 保持特征值不变，但改用坐标置换或随机基 | 排除“只是重排坐标”或“任意随机基都会这样”的解释 |
| `theory_operator_alignment_decoupling/` | 一次只改变对齐角度，一次只改变谱的各向异性 | 说明 alignment 和 spectral shape 是两个不同因素 |
| `theory_operator_alignment_anisotropy_sweep/` | 连续改变 latent spectrum 的幂律指数 | 说明 eigengap 决定效应大小，但不决定是否对齐 |
| `theory_operator_alignment_metric_robustness/` | 用多个 transfer readout 重复测量 | 说明结果不是某一个特定标量的偶然现象 |
| `theory_dim_sweep/`、`theory_mechanism_sweep/`、`theory_robustness_sweep/` | 改变维度、encoder/predictor 指数、噪声和随机种子 | 补充材料中的稳健性和机制结果 |
| `theory_empirical_bridge/` | 冻结 LeWorldModel PushT latent 的 covariance-relative readout | learned latent 的一阶测量实例 |
| `jepa_feature_measurements/` | I-JEPA 对六个视觉数据集的原图/水平翻转 feature displacement | learned predictive representation 中的 view-transfer 测量 |
| `jepa_operator_rank_sweep/` | 用 held-out view pairs 拟合低秩 operator，并检查 rank 稳定性 | 说明 feature predictability 与 operator identification 是不同问题 |
| `temporal_world_models/` | JEPA-WM/DINO-WM 的 held-out 多步 latent MSE、commutator、spectrum-preserving null、action-order intervention、split robustness 和 estimator ablation | 将同一测量协议带到 action-conditioned world model，并显式报告协议边界 |
| `temporal_world_models/hardened_cross_environment/` | PushT、PointMaze、Wall 的 future-only action 消融、Wall 分割稳健性、episode derangement、rank/ridge stability 及 SHA-256 manifest | 新增附录的完整对比与消融证据 |
| `alignment_interventions_20261003/` | 学习型异质 synthetic predictor 与冻结 PushT latent adapter 的对照干预 | 检验 alignment diagnostic 与预测误差之间是否存在必然关系 |
| `conditioned_controllability_pilot_20261003/` | 三个 seed 的 state-conditioned action-to-dynamics held-out readout 与 action-permutation null | 检验冻结 latent 中 action information 的 offline detectability；不等同于 causal controllability 或 planning |

### 最重要的三个 temporal JSON

- `dino_wm_pointmaze_temporal_metrics_200.json`：DINO-WM PointMaze，200 个 holdout episodes，seed-42 loader split 重建；三个 horizon 的 commutator 都高于 null。
- `dino_wm_pusht_temporal_metrics.json`：第二个 DINO-WM PushT checkpoint，公开的 18,685/21 train/validation split；horizon 1、2 低于 null，horizon 4 高于 null。这是有意义的 horizon-dependent 结果，不应被改写成单调规律。
- `jepa_wm_pointmaze_temporal_metrics_200.json`：匹配环境的 JEPA-WM PointMaze，200 个 holdout episodes；full-rank 和 rank-truncated commutator 都高于 null。

新增的 `*_action_intervention_comparison.md`、`*_within_episode_sensitivity.md`、`pointmaze_split_robustness.md`、`pointmaze_split_operator_controls.md`、`h8_long_horizon_controls.md`、`jepa_context_ablation.md`、`operator_direction_ablation.md` 和 `pointmaze_sample_efficiency.json` 对应补充材料的动作顺序、split、长时域、context、方向和样本量消融。它们均保持固定 checkpoint replay 的证据边界，不应解释为训练曲线、模型排名或下游控制性能。

每个 temporal JSON 都包含 `protocol` 字段，里面写明 context length、action block、horizon、checkpoint epoch、数据 hash、split 方式和 normalization。合作者不需要猜这些数字是怎么来的。

### Alignment intervention 的证据边界

`alignment_interventions_20261003/` 包含两项配对实验。学习型 synthetic benchmark 使用三个初始化种子和相同的数据划分，比较无正则、global alignment 和 action-conditional alignment；冻结 PushT adapter 使用 100,000 条确定性 latent rows、80/10/10 row-level split 和三个种子。前者显示 pooled geometry 不匹配时 global alignment 会损害预测，后者显示 adapter 本身贡献了主要 one-step 改善，而 alignment 在较小权重下影响极小、较大权重下可能反向影响 MSE。两者都不是端到端 LeWM 重训、真实 multi-step rollout、online control 或 planning 证据。

新增的 `action_sequence_composition_20261003/` 补上了 ordered two-action product 与随机序列 rollout 之间的可审计中间层：对 11 个 mode/weight 条件和三个 seed 枚举全部二元 action sequence，在 $h=2,4,8$ 分别检查 4、16、256 个 learned-versus-generating operator products，并与同一 checkpoint protocol 的 ID/OOD rollout 对齐。h=8 的 pooled Pearson 为 ID 0.996、OOD 0.997；这些是受控线性 synthetic diagnostic 的描述性关联，不是因果中介、nonlinear LeWM、PushT、online planning 或 GPU 证据。

## 四、原始大数据在哪里

为了让 Overleaf 上传轻量、避免把受许可约束的模型权重塞进协作项目，本包只包含结果级数据和 provenance，不把 31GB 原始数据、404MB checkpoint 和 2.5GB I-JEPA 权重复制进来。需要重跑实验时，使用原工作区中的以下文件，或按 `reproducibility/provenance/source_manifest_zh.md` 下载：

- DINO-WM PointMaze archive：`data/dino_wm_osf/point_maze.zip`
- DINO-WM PushT archive：`data/jepa_wm_osf/pusht_noise.zip`
- JEPA-WM PointMaze checkpoint：`data/jepa_wm_checkpoints/jepa_wm_pointmaze.pth.tar`
- JEPA-WM PushT checkpoint：`data/jepa_wm_checkpoints/jepa_wm_pusht.pth.tar`
- I-JEPA ViT-H/16 checkpoint：`external/ijepa_vith16_1k_hf/model.safetensors`

这些文件的 SHA-256、大小、来源和当前结果的对应关系都在 provenance 文件中给出。

## 五、如何共同编辑

- 论文正文：只改 `overleaf/main.tex`。
- 补充材料：只改 `overleaf/supplement.tex`。
- 参考文献：改 `overleaf/references.bib`。
- 图：优先修改 `overleaf/figure_sources/` 的 SVG/PPTX，再导出覆盖 `overleaf/figures/*.pdf`。
- 结果数字：不要直接在 LaTeX 中凭记忆改数字；先检查 `reproducibility/results/` 中对应 JSON，再同步正文。
- 内部复现和审计材料放在 `reproducibility/`，不要把它们添加成论文章节或论文表格。

## 六、快速检查

在 `overleaf/` 目录本地编译：

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error supplement.tex
```

当前包已验证：主文和补充材料均可编译；主文 13 页、补充材料 29 页；论文台账化用语检查为 0；reader-facing table 检查为 0；新增 forward-readout 表和既有干预/learned pixel-to-latent 表已完成检查。
