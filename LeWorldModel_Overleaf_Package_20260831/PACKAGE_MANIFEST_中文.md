# 论文包清单

打包日期：2026-10-05

## Overleaf 文件

| 文件 | 作用 |
|---|---|
| `overleaf/main.tex` | 主文入口 |
| `overleaf/supplement.tex` | 独立补充材料入口 |
| `overleaf/references.bib` | 参考文献 |
| `overleaf/iclr2027_conference.sty` / `.bst` | ICLR 2027 官方样式和 bibliography style |
| `overleaf/fancyhdr.sty` | 页眉样式依赖 |
| `overleaf/figures/*` | LaTeX 实际使用的 PDF/PNG 图 |
| `overleaf/figure_sources/` | SVG/PPTX 编辑源 |

## 结果级数据

`reproducibility/results/` 只包含论文使用的 JSON/CSV、split manifest，以及新增跨环境消融、alignment-intervention 和 pixel-to-latent 边界实验的汇总结果。它们可以直接被合作者打开检查，不需要先下载几十 GB 原始数据。

新增证据的显示索引为 `reproducibility/provenance/display_plan.csv`；该 30 行索引已在原工作区根目录下通过缺失路径、重复 ID 和空字段检查。包内路径使用 `reproducibility/` 前缀，工作区审计路径使用 `results/`、`experiments/` 等工作区相对路径。

本次新增 `reproducibility/results/alignment_functional_synthetic_20261003/` 和
`reproducibility/code/alignment_interventions/run_alignment_functional_synthetic.py`，
对应补充材料 Table 34。它记录 33 条逐 seed 结果、11 个聚合条件，并把
composition、pooled/global commutator、conditional commutator、随机序列 rollout
和 candidate-set planning proxy 分开保存。该结果是 CPU-only、synthetic-only；
planning proxy 不是 online task success 或 causal control。当前复现实验环境中的
PyTorch 无法初始化 CUDA/NVML，因此不承诺 CUDA 或 GPU 效率结论。

本次新增 `reproducibility/results/sigreg_alignment_phase_20261003/` 和
`reproducibility/code/alignment_interventions/run_sigreg_alignment_phase_synthetic.py`，
对应补充材料 Table 37 和 phase figure（Figure 3）。该交叉实验包含 63 条逐 seed 记录、21 个
条件组和 3 个 matched seeds，并用 stable rank 检查 SIGReg-induced covariance
flattening 是否足以解释 commutator 变化。结果是 CPU-only synthetic mechanism
evidence；它不支持将较低 commutator 写成较好预测、composition、planning 或 GPU
效率的充分条件。

本次同步 `reproducibility/results/action_sequence_composition_20261003/` 和
`reproducibility/code/alignment_interventions/run_action_sequence_composition_synthetic.py`，
对应补充材料 Table 35、Table 36 和 Supplementary Figure 2。它包含 33 条逐 seed
记录，枚举二元 action sequence 在 $h=2,4,8$ 的 4/16/256 个 exact products，并
与同 protocol 的 ID/OOD random rollout 做逐记录一致性检查。h=8 的 ID/OOD pooled
Pearson 分别为 0.996/0.997；这是 CPU-only、synthetic-only 的描述性结构诊断，
不支持 nonlinear LeWM、PushT、online planning、causal control 或 GPU 效率结论。

本次同步的 conditioned-readout 证据位于
`reproducibility/results/conditioned_controllability_pilot_20261003/`，对应补充材料 Table 19 和页 7 的渲染审计。它只支持冻结 latent 上的 state-conditioned offline detectability，不支持 causal controllability、online planning、goal reachability 或 multi-step rollout。

本次新增 `reproducibility/results/external_rank_target_specificity_20261004/` 和
`reproducibility/code/external_frozen_dynamics_audit/run_external_rank_target_specificity_audit.py`，
对应补充材料的 rank-and-target-specificity 审计。它在 21 个外部 episode、
$k=1,2,4,8,16,32$、$h=1,4,8$ 的留一 episode 外推中，区分 dynamics-band、
residual-band、完整 latent 和 physical displacement targets；结果只支持
target-specific frozen predictive diagnostic，不支持 physical causal effect、
online planning、task success 或 retraining gain。

本次新增 `reproducibility/results/contact_predictor_teacher_forced_20261004/`
和 `reproducibility/code/contact_predictor_repair/`，对应补充材料的
teacher-forced multi-step predictor repair 表。该实验固定视觉 encoder/projector，
只修复 action encoder、predictor 和 prediction projection；两个独立 seed 使用
48 个训练接触案例和 16 个 held-out 案例。h=4 的 paired calibrated gain 在两个
seed 上均有正的 episode-bootstrap 区间，h=2/h=8 的点估计均改善但区间仍跨零；
native readout control 在各 horizon 均恶化。该证据是单 checkpoint 的 targeted
repair diagnostic，不是原始 frozen LeWM 已具备 physical causality、online planning
或 task success 的证据。

本次继续加入 `reproducibility/results/contact_predictor_deep_20261005_*`、
`contact_predictor_deep_summary_20261005/` 和
`contact_predictor_ablation_summary_20261005/`。扩展 protocol 保留 `+a/-a/zero`
三臂、observed-latent upper bound、horizon profile、calibration-size sweep，
以及 epoch、anchor-weight 和训练案例数消融。两个 seed 在 h=4 的三臂 point
gain 均为正；observed-latent upper bound 仍明显优于预测 latent，说明修复只
恢复部分 predictor/readout mismatch，不能被写成原始 frozen checkpoint 的
物理因果或 planning 能力。单 seed 消融明确标注为 sensitivity diagnostic，
没有被提升为新的多 seed 排名结论。

本次再同步 `reproducibility/results/contact_predictor_deep96_20261005_seed8421/`
和 `contact_predictor_deep96_20261005_seed9271/`。这是相同 protocol 的扩大复现，
每个 seed 使用 96 个训练接触案例和 24 个 held-out 案例；h=4 的 $+a$ gain 为
`+37.711 [+19.572,+55.555]` 与 `+31.047 [+18.726,+44.464]`，且 $-a/zero`
三臂 point gain 同样为正。h=8 的点估计仍为正，但一个 seed 的区间跨零；因此
该结果强化短时 targeted repair，而不宣称长时 physical causality、counterfactual、
online planning 或 task success。

## 大文件策略

原始数据和模型权重没有复制到这个协作包中：DINO-WM/JEPA-WM 原始数据约 4.4GB，I-JEPA 权重约 2.5GB，完整工作区数据目录约 31GB。下载来源、大小、哈希和 split 解释在 `reproducibility/provenance/source_manifest_zh.md`。

## 关键文件哈希

| 文件 | SHA-256 |
|---|---|
| `overleaf/main.tex` | `cb749c79a9e03b378f5c3e1e233f55c2d658c24a4b8642c1deb75e5b00d7d61e` |
| `overleaf/supplement.tex` | `d1ca166d3d01ff05c40baf31485e7d5a58c5e03bc5c792fd240a1472fe8d00a4` |
| `overleaf/main.pdf` | `24a4c79868543a4205573488fdf888377402133d50e2fa29d8097f8980dcc62e` |
| `overleaf/supplement.pdf` | `82bde38cb7f83c5119c715778fcee9c24ab05abbb16c3af45301003097f1db4c` |
| `overleaf/iclr2027_conference.sty` | `797deef41724e93761426ac0cbcca46279a91cc650dd1f0ce76a4f08d2098ea6` |
| `overleaf/iclr2027_conference.bst` | `2d67552db7ed38ccfccb5957b52f95656e25c249724761d3cf5f7922ad1844c5` |
| `overleaf/figures/hardened_future_action_ablation.png` | `f0c6f3ccbd67e2261f173929af2470e521cdfd891836423d6dc9a55aff41985e` |
| `overleaf/figures/hardened_split_robustness.png` | `5502dafdb12c035810d539022a6f16b84910cbd1ce68d17eed88469770edde37` |
| `overleaf/references.bib` | `0bb7b2153d6a14fc0c3158d10ff23fd0f130d18b04feebe2a685a2bfdd393de2` |
| `overleaf/figures/operator_pair_intro_figure.pdf` | `201966748c68e28111ce26e93d5d62823322d8c9a8631ffe05466e0943f8b321` |
| `overleaf/figures/operator_pair_main_figure.pdf` | `dfc39fb8eae7474092197a859a98e238114f23436d080cb4316283b808326a6e` |
| `overleaf/figures/ijepa_feature_anchor.pdf` | `6127c5a06189bfc2e9c54fcaf9076f3e3293621ef941de35edb8c68869605cce` |
| `overleaf/figures/sigreg_alignment_phase_diagnostics.png` | `dbab535cd4783d6e7c514ee889734aed049aff9630f1d032623aa3264e5c5710` |
| `overleaf/figures/composition_rollout_scatter.png` | `0d1090fabc391763f5c80d0083bf7f886434f7360f2d9eb6847f092351970893` |
| `reproducibility/code/alignment_interventions/module.py` | `0b258a9e8dc24c29fcb1e8c50a09ec78b8ea85aeb79e21dd8adf712396646620` |
| `reproducibility/code/alignment_interventions/run_pixel_world_model_synthetic.py` | `d51a0cfd9857ff55fccece7339aa033276055e15624d798866eac14cf5173ce3` |
| `reproducibility/results/pixel_world_alignment_20261003/pixel_world_model_synthetic.json` | `be3906da7f021c58ed4071bad0d3d29b946f181c4317d0b457a46cd0a1bf29d3` |
| `reproducibility/results/pixel_world_alignment_20261003/pixel_world_model_synthetic.md` | `e60cb0184d82188d2daab1aa23e550c181045b4cf1cef5bf6bb55520a00b8f24` |
| `reproducibility/results/pixel_world_alignment_20261003/notes.txt` | `0013ab2cabd2c71f58c254503bed640dcc65788f5fc3d823192619eabaa909d3` |
| `reproducibility/results/pixel_world_alignment_20261003/visual_audit.md` | `009fb391b08ff3f34234e06c396997945b6a773fe4c3a9a0940f2a5293ab3167` |
| `reproducibility/code/equivariance_interventions/run_predictor_equivariance_synthetic.py` | `0f43404cc6425e64fcfdd161e3c7c39122feddc9c4fbfa0474165cf5ab7aac4f` |
| `reproducibility/code/equivariance_interventions/regularizers.py` | `bd5c6acb6d826c6ea3f86afc9d7070ad3da01841336802d0d47d6c8cfdca7d28` |
| `reproducibility/results/predictor_equivariance_synthetic_20261003/predictor_equivariance_synthetic.json` | `f7239eca11b6f18c38c576fe0eefbdb8d2b7c32203aacb01d2d85b4e021e5728` |
| `reproducibility/results/predictor_equivariance_synthetic_20261003/predictor_equivariance_synthetic.md` | `4d9670ea08aab0eaf26ab4ed615823be7cb21ee7dfb53d55987f63ba0f8985c3` |
| `reproducibility/results/predictor_equivariance_synthetic_20261003/notes.txt` | `62eb8542c8c03781d8f7290e442566eb9b40642c133d01788e5edca0d63928fa` |
| `reproducibility/results/predictor_equivariance_synthetic_20261003/visual_audit.md` | `dca1fd26370c13b8c87b88f5dd657e0c9850d6a5dd073ec59fd5aa3b346157cc` |
| `reproducibility/results/predictor_equivariance_synthetic_20261003/rendered/supplement_page-10.png` | `46ee9cea3381653004e14d5129fa8cd2f33a08c280b385720c0bd40e4854db25` |
| `reproducibility/code/equivariance_interventions/run_equivariance_functional_synthetic.py` | `2bedc6fd1d784cdb8a341321356f85b13105fa1ed4da03923d5d87eff07e5b85` |
| `reproducibility/results/equivariance_functional_synthetic_20261003/equivariance_functional_synthetic.json` | `b15671f0b022a2d8e58e22d6480d5313d9fa771b906a4e53df0299c242f82b90` |
| `reproducibility/results/equivariance_functional_synthetic_20261003/equivariance_functional_synthetic.md` | `075b5bfcccba61aeb97ce4b286f88d6c3d6cf9c5c78f86cbcf5a14a41dce4857` |
| `reproducibility/results/equivariance_functional_synthetic_20261003/README.md` | `519400bc648ba2dcdc2aab0a5936071071773c1a62711539c0125cdbffd70feb` |
| `reproducibility/results/equivariance_functional_synthetic_20261003/notes.txt` | `8b99e797ce77b92929f7a72e8ae4f3b6f48f203797d6d455d24ef99d382b7b7d` |
| `reproducibility/results/equivariance_functional_synthetic_20261003/visual_audit.md` | `d3ed51212921376779f76015e983adb5d261b572915438bfe712342d4cbacb97` |
| `reproducibility/results/equivariance_functional_synthetic_20261003/rendered/supplement_page-11.png` | `57896cee85fd2e4d9a61f3d60c7aa76b66f86f824edabe4ca1800e139a37e60d` |
| `reproducibility/provenance/claim_ledger.csv` | `be8aaef944f6963912016e68fa65f5ba42f247ac4d5d138f1983a44daf9544b7` |
| `reproducibility/provenance/figure_table_crosswalk_中文.md` | `06231aaebe1f232396e589cda870bd79ffe990b674422ae16412cd7f9e93ec7a` |
| `reproducibility/provenance/revision_log.md` | `b55d5237bbf52c52ac10ee33aa552c96a75efb7af2156f6493da897880a496b2` |
| `reproducibility/README_中文.md` | `6ce23edfe252b004c4b098d3b146513b0ec6155f00f00c75199b30c9e9cfd5f2` |
| `reproducibility/code/README.md` | `ff3c6fcb8364a2f38943296867859ca7ddee95a73eab60d799e830f8cf1dba89` |
| `reproducibility/code/alignment_interventions/run_action_sequence_composition_synthetic.py` | `1e9d9eab901edc026df4c197c3efc1949df2856ae924104eb3d973cd66bc997c` |
| `reproducibility/results/action_sequence_composition_20261003/README.md` | `8363dee0e6eb019a908a5c9c3110660be023e84545b94f5ae25364c0d0a6fab3` |
| `reproducibility/results/action_sequence_composition_20261003/action_sequence_composition_synthetic.json` | `36482ae9bb67591b96c700e8d84ca38c697290aa5bc47b852bde2985f4bc77bd` |
| `reproducibility/results/action_sequence_composition_20261003/action_sequence_composition_synthetic.md` | `b079633cbd18e28f7c02a4f2ce7476d9c7bc5e10f04713ef5c944c196780492d` |
| `reproducibility/results/action_sequence_composition_20261003/composition_rollout_scatter.png` | `0d1090fabc391763f5c80d0083bf7f886434f7360f2d9eb6847f092351970893` |
| `reproducibility/results/action_sequence_composition_20261003/notes.txt` | `525cc211650c453f6f540de5c56c43d3fdef23ce54cd1bac7c8f9964a2978021` |
| `reproducibility/results/action_sequence_composition_20261003/visual_audit.md` | `bdfb58a5f95854c0b6de3b3fcb646c1673f8fe2b4c53e439dc2b2e50a6cba764` |
| `reproducibility/results/action_sequence_composition_20261003/supplement.pdf` | `dd642a13695e6bad77d002fc489af8eccb74b1e1944216768f0d37f63a23c2f7` |
| `reproducibility/results/action_sequence_composition_20261003/rendered/supplement_page-17.png` | `6b633711e8a07e30e299dbcdc900c68278b01ddbdbd5985f78dd3b7228e98ca5` |
| `reproducibility/results/action_sequence_composition_20261003/rendered/supplement_page-18.png` | `5d44f745fca3655a4221fbb63c93eb4ed4a36e960d8741b71f32de309ae68f6a` |
| `reproducibility/code/alignment_interventions/run_alignment_functional_synthetic.py` | `2f90cb3188d4b7b1e81d435fafb9c0ec75a2e4619692579ee1c57a2d4884d7a7` |
| `reproducibility/results/alignment_functional_synthetic_20261003/alignment_functional_synthetic.json` | `187f0345d9d4c13bf67f30a410a63f7692a9912d2c2eab072efd4bbfd9f889d8` |
| `reproducibility/results/alignment_functional_synthetic_20261003/alignment_functional_synthetic.md` | `705200d1495d4aea76b1bd5d2a654170e8c643fb399c60461b309851c8269988` |
| `reproducibility/results/alignment_functional_synthetic_20261003/README.md` | `4596f47c9985361b3d8ecdba5dbca4df698e1c1d15fb912088bc2ac3e643b67c` |
| `reproducibility/results/alignment_functional_synthetic_20261003/notes.txt` | `409e1ccf480eb12720807c8653770b9cdffae0bc6fc57e400ba5cdee4b569400` |
| `reproducibility/results/alignment_functional_synthetic_20261003/visual_audit.md` | `a9649bcabe9364911992f9ab0176b980397efbd6a0cfbd443ec878728f38758c` |
| `reproducibility/results/alignment_functional_synthetic_20261003/rendered/supplement_page-16.png` | `00e967b2592048b45ad7fe32ca8faab56c1589c169b4c5cdedb4b4dca4bdc683` |
| `reproducibility/code/alignment_interventions/run_sigreg_alignment_phase_synthetic.py` | `d53bc5ef645178d3354a11f235d221eef915298c9b2c20070a37ba01194b58c1` |
| `reproducibility/results/sigreg_alignment_phase_20261003/README.md` | `6dee780c8324b542f5112d0f0791c1f4ca253283fde36d9c20fb5f8d3408d4f8` |
| `reproducibility/results/sigreg_alignment_phase_20261003/sigreg_alignment_phase.json` | `dba61d2c3fa4653522e40c25d07fc17a0c65fae0cb8cc7605649bd1466f6d730` |
| `reproducibility/results/sigreg_alignment_phase_20261003/sigreg_alignment_phase.md` | `7fa451a53f73ad4e322fc07aa9a192f4569e02fe29c75013c57f315a972c0b11` |
| `reproducibility/results/sigreg_alignment_phase_20261003/notes.txt` | `bf5a23f3a49598187cfd8e9bee1fa09c0db65f65642928caf58e6eaa1ee51134` |
| `reproducibility/results/sigreg_alignment_phase_20261003/visual_audit.md` | `ad1c25dfaa9cab1b711d08de1f0b5f16544f715a0ef1b7b144cb99be7834f5c9` |
| `reproducibility/results/sigreg_alignment_phase_20261003/phase_diagnostics.png` | `dbab535cd4783d6e7c514ee889734aed049aff9630f1d032623aa3264e5c5710` |
| `reproducibility/results/sigreg_alignment_phase_20261003/rendered/supplement_page-17.png` | `9c00753677d1f3cb21a4cff744f3498430b11fa4d5f91f5c46023ec47786c507` |
| `reproducibility/results/sigreg_alignment_phase_20261003/rendered/supplement_page-18.png` | `4aec7c09ad248aaac10463ae3deafbc312aec251d6a36ba2aa285b92a3c0df89` |
| `reproducibility/results/sigreg_alignment_phase_20261003/supplement.pdf` | `76d48025fa6b8ef966b90811ab75fda2d9d9ea84d76d523fc5338488092fb851` |
| `reproducibility/results/sigreg_alignment_phase_20261003/display_artifact_audit.json` | `c02dd3838ec63517b5afb624ffbe25019c9fa341a937819dde958325f30e76a4` |
| `reproducibility/results/sigreg_alignment_phase_20261003/reader_facing_tables_audit.json` | `113936a28cea60fa912787e41917137b9cd1b527e35b46f54691ce85f8eb9fac` |
| `reproducibility/results/sigreg_alignment_phase_20261003/manuscript_not_report_audit.json` | `113936a28cea60fa912787e41917137b9cd1b527e35b46f54691ce85f8eb9fac` |
| `reproducibility/code/external_frozen_dynamics_audit/summarize_forward_dynamics_deep.py` | `d67c944194c0c690f6c6b14a269433a1358829f66bf07436a3dbb3f21b8bb178` |
| `reproducibility/code/external_frozen_dynamics_audit/test_summarize_forward_dynamics_deep.py` | `1a7dc958a7a1b049bb8ae43044377281a4c6eadb9aa833c90592e88f74b95cd7` |
| `reproducibility/results/temporal_world_models/external_frozen_dynamics_audit/forward_readout_20261004/forward_dynamics_deep_summary_20261004.json` | `68006f705c76a0de4e97736da07e94afe34a720f3f5be94e7e7e67ad9ef90af1` |
| `reproducibility/results/temporal_world_models/external_frozen_dynamics_audit/forward_readout_20261004/forward_dynamics_deep_summary_20261004.md` | `9d3fd61c7d7fa010c34cb519fb4eff26ba050beabc9c95f85b9dfe89da581d31` |
| `reproducibility/results/temporal_world_models/external_frozen_dynamics_audit/forward_readout_20261004/external_physical_dynamics_probe_public_w8.json` | `be9de4f9e75d8081d428d3f84c8bb541e9f5e310dee3ed4ab8c65813f0941c8d` |
| `reproducibility/results/temporal_world_models/external_frozen_dynamics_audit/forward_readout_20261004/audit_manuscript.json` | `a37004e5d40d36476e83fc6ece0d06e64f59e9286364a7d274c84d8238360245` |
| `reproducibility/results/temporal_world_models/external_frozen_dynamics_audit/forward_readout_20261004/audit_manuscript.md` | `be22033e0d3b4e877a3c08dcf3ca6da0db5a1046389d48f795d3c6b8e4cad9e3` |
| `reproducibility/results/temporal_world_models/external_frozen_dynamics_audit/forward_readout_20261004/audit_tables.json` | `a37004e5d40d36476e83fc6ece0d06e64f59e9286364a7d274c84d8238360245` |
| `reproducibility/results/temporal_world_models/external_frozen_dynamics_audit/forward_readout_20261004/audit_tables.md` | `03b0af74928ac78c7b7ddab31bd311005348832fb0dd39c56f6c4a3841efdf36` |
| `previews/main.pdf` | `24a4c79868543a4205573488fdf888377402133d50e2fa29d8097f8980dcc62e` |
| `previews/supplement.pdf` | `82bde38cb7f83c5119c715778fcee9c24ab05abbb16c3af45301003097f1db4c` |
| `reproducibility/code/contact_predictor_repair/run_pusht_contact_predictor_teacher_forced_repair.py` | `9c7b4eb154eb1d05a43bbf73494d414f330f38b8999fae15d9a1a46dcd0ec979` |
| `reproducibility/code/contact_predictor_repair/summarize_pusht_contact_predictor_repairs.py` | `58faff0e0b953d38dc8ef59d079f25c5768f41df689181c77c936afaef8a24ff` |
| `reproducibility/results/contact_predictor_teacher_forced_20261004/contact_predictor_teacher_forced_summary.json` | `dbebdfc8ffb8b7bb98be1974a21b7e4de21306b05b72b5b724e7ef61295ed46c` |
| `reproducibility/results/contact_predictor_teacher_forced_20261004/contact_predictor_teacher_forced_summary.md` | `3096bbca55bf8406f15ec349d3c9649594e27b340a194f57bfd7f374938d72e3` |
| `reproducibility/code/contact_predictor_repair/test_pusht_contact_predictor_teacher_forced_repair.py` | `29bccfd7b24a94514aa107873c817fa7703a25e105b04840a02e50e938becdb6` |
| `reproducibility/code/contact_predictor_repair/summarize_pusht_contact_predictor_deep.py` | `3e38383ee1b23d031efbe6da35cbce924813ea56e3e8e3fa889f64a2a5eeba7d` |
| `reproducibility/code/contact_predictor_repair/summarize_pusht_contact_predictor_ablation.py` | `1db79d744f977fa69ff190a11bc415beaf053af131ed3eb55b8c67b43bbe378c` |
| `reproducibility/results/contact_predictor_deep_20261005_seed8421/contact_predictor_teacher_forced_repair.json` | `4c30e65c02dd7c775250d207c425b67526730a5af15195fb5d3d8b4fe7029587` |
| `reproducibility/results/contact_predictor_deep_20261005_seed9271/contact_predictor_teacher_forced_repair.json` | `4b6686e8374834f2d2d6d85be2b43b6d3eade9e0dcd595b1b2c1e6a565bd5c37` |
| `reproducibility/results/contact_predictor_deep_summary_20261005/contact_predictor_deep_summary.json` | `72b67e27c20081c16d049cf97cffa4d6873f9216bc84f3f62528be00fa28621b` |
| `reproducibility/results/contact_predictor_deep_summary_20261005/contact_predictor_deep_summary.md` | `8f3ed1898c4336e2ada35b4e3d4c0712ede42a5b392778cfb1b43dd1824ce427` |
| `reproducibility/results/contact_predictor_ablation_summary_20261005/contact_predictor_ablation_summary.json` | `17a8c910606ea824aed7b6b53e7fa4510bb0c6cdeb40ebaae65220287e11ce65` |
| `reproducibility/results/contact_predictor_ablation_summary_20261005/contact_predictor_ablation_summary.md` | `44696cf2276ef6fabfc357218a6068a1d588a4fc5fba9772f3793a7f516a4e13` |
| `reproducibility/results/contact_predictor_deep96_20261005_seed8421/contact_predictor_teacher_forced_repair.json` | `06c76d7af43b04970c31abf34f771cefdbf9b4940dcc69fccd0a979aff7770ca` |
| `reproducibility/results/contact_predictor_deep96_20261005_seed8421/contact_predictor_teacher_forced_repair.md` | `9e7d55d1084c9bfa9fa2ae03078f3f45b59fb4acde3aaaf3d41201d97847f4d2` |
| `reproducibility/results/contact_predictor_deep96_20261005_seed8421/run.log` | `5c8ec062f1415b637dbff52514325bdc31e630c86ace3ea5439fa793ef3fa284` |
| `reproducibility/results/contact_predictor_deep96_20261005_seed9271/contact_predictor_teacher_forced_repair.json` | `0124c6cb818bf36b44e68cd816a96166f417f83e5929263eeace826f9c6ee114` |
| `reproducibility/results/contact_predictor_deep96_20261005_seed9271/contact_predictor_teacher_forced_repair.md` | `1425f0b9b52863f49de446cf64e0169bb0ecb80e8792c3c53596d61512efdca8` |
| `reproducibility/results/contact_predictor_deep96_20261005_seed9271/run.log` | `0aa3feb1fb0bc4bb686a1637786d3368456878701322f0f64ebfadbf19995217` |
| `reproducibility/results/contact_predictor_moredata_20261005_seed8421/contact_predictor_teacher_forced_repair.json` | `f71bd81d23e43f2e5afe0bb672c7143da684bd3257321e3bb26d42a7ce74af6f` |
| `reproducibility/results/contact_predictor_moredata_20261005_seed8421/contact_predictor_teacher_forced_repair.md` | `758374b7d409ee27c40ba8513f26a47dd5018c16b59ce795ebbabca066e779bf` |
| `reproducibility/results/contact_predictor_moredata_20261005_seed9271/contact_predictor_teacher_forced_repair.json` | `fe00f49a3347c37ed1ec27bd2f55f40b71a243a7b07cdb8264a13c790c7fa1a0` |
| `reproducibility/results/contact_predictor_moredata_20261005_seed9271/contact_predictor_teacher_forced_repair.md` | `6c63dd737b5f9983735fc101e226dd22fe13ddebe187492ef22207b20cda3fb0` |

## 当前验证

- `main.pdf`：本次复核后页数以当前构建为准，使用 ICLR 2027 官方样式；摘要与引言已重构为正式学术论证，Related Work 和 Discussion 已加入 seq-JEPA 的正交定位。
- `main.pdf`：当前构建为 16 页；`supplement.pdf`：当前构建为 42 页，均使用 ICLR 2027 官方样式。附录新增 teacher-forced 三臂、observed-latent upper-bound、calibration-size、training-strength ablation、96/24 enlarged replication 与 192/48 contact-mode coverage scaling 表；这些结果只承担冻结 checkpoint 的 targeted repair 诊断，不扩展为原始 LeWM 的 physical causality、online planning 或 task success 结论。
- 主文定义、读出量、核心定理与证明公式统一编号为 `(1)--(10)`，补充材料沿用独立连续编号 `(S1)--(S17)`；本次新增 rank-target estimand 为 `(S17)`，正文交叉引用均已解析。
- 新增冻结外部动作代数证据位于 `reproducibility/results/temporal_world_models/lewm_action_algebra_20261004/`，包含 public、validation z-score 和 raw/100 三个 normalization artifact；对应文件已加入 `SHA256SUMS_关键文件.txt`。
- 本次进一步修正了 action-algebra direct horizon map 的 `(window, start)` 特征对齐，并新增针对该错位的合成回归测试。修正后的 2-window 三 normalization artifact 已覆盖原结果；public normalization 的 up-to-8-window replication 位于 `reproducibility/results/temporal_world_models/lewm_action_algebra_20261004_w8/`，同时保留 runner 和 targeted test。论文只将其解释为 action--state alignment diagnostic，不将其升级为 exact composition、causal control 或 online planning 证据。
- 新增冻结外部 conditional local-operator 证据位于 `reproducibility/results/temporal_world_models/lewm_conditional_operator_geometry_20261004/`，包含修正后的 2,000-replicate bootstrap、三种 action normalization，以及 public normalization 下 4/8/16 state-strata sensitivity；对应脚本、汇总器和 JSON/Markdown artifact 均已加入 `SHA256SUMS_关键文件.txt`。
- 新增冻结外部 dynamics directionality 证据位于 `reproducibility/results/temporal_world_models/external_frozen_dynamics_audit/`，包含 temporal adapter 跨归一化汇总、future-vs-past/deranged-action audit、源代码与可复现 JSON/Markdown；严格双重 CI 判据在 36 个单元中支持数为 0，故只作为 claim-boundary 结果写入补充材料。
- 新增 leakage-controlled forward-readout 深度汇总位于 `reproducibility/results/temporal_world_models/external_frozen_dynamics_audit/forward_readout_20261004/`，报告 $h\in\{1,4,8\}$ horizon profile、state/action/temporal controls、matched middle-band readout、full-latent negative control，以及 physical-target negative control；该结果只支持 source-defined band-level encoded-latent predictive diagnostic。
- 两份 LaTeX 均已成功编译；第二遍构建后无 undefined reference 或 fatal LaTeX error。日志仍有既有 underfull box 排版提示，但无 overfull box。
- teacher-forced/contact 相关 targeted regression：27 passed（10 个新增/动作规范化测试 + 17 个既有 PushT closure 测试）。
- manuscript report morphology audit：通过。
- reader-facing table audit：通过；新增 repair 表已完成渲染检查。
- contact-mode coverage scaling：两组独立 192/48 runs 完成；h=8 的 repaired-calibrated $+a$ gain 分别为 `+41.412 [+22.044,+62.088]` 和 `+39.654 [+23.293,+60.053]`，$-a$/zero matched-arm gains 均为正。该结果支持 targeted repair 的 contact-mode coverage 解释，不升级为原始 frozen checkpoint 的因果或规划能力。
