# 图表—数据—脚本对应表

| 论文对象 | 读者看到什么 | 主要数据 | 生成脚本 |
|---|---|---|---|
| Figure 1 | LeWorldModel 中 latent covariance 与 predictor operator 如何形成 operator pair | `figures/operator_pair_intro_figure.pdf` | `code/make_operator_pair_intro_figure.py` |
| Figure 2 | 固定 predictor spectrum、旋转 eigenspace 后 commutator 与 transfer readout 的变化 | `results/theory_operator_alignment_sweep/` | `code/make_operator_pair_main_figure.py` |
| Table 1 | permutation-preserving 和 random-basis null 与 oblique rotation 的差别 | `results/theory_operator_alignment_null_sweep/` | `code/run_operator_alignment_null_sweep.py` |
| Table 2 | alignment angle 与 spectral anisotropy 的分离 | `results/theory_operator_alignment_decoupling/` | `code/run_operator_alignment_decoupling.py` |
| LeWorldModel PushT | frozen latent 的 top-band dynamics、residual 和 transition overlap | `results/theory_empirical_bridge/` | `code/run_operator_alignment_bridge.py` |
| Frozen PushT conditioned action readout | 三个 seed 的 state-conditioned held-out action-to-dynamics $R^2$ 与 action-permutation null | `results/conditioned_controllability_pilot_20261003/` | `code/verify_conditioned_controllability.py` |
| Supplement Table 22 | trainable pixel encoder 下的 global/conditional alignment intervention 与 one-step、composition、rollout 边界 | `results/pixel_world_alignment_20261003/` | `code/alignment_interventions/run_pixel_world_model_synthetic.py` |
| Supplement Table 23 | controlled vector world 中 latent/predictor equivariance 的分离消融，以及 geometry 与 rollout/composition 的边界 | `results/predictor_equivariance_synthetic_20261003/` | `code/equivariance_interventions/run_predictor_equivariance_synthetic.py` |
| Supplement Table 24 | equivariance intervention 在 action OOD、初始状态偏移、h=8 rollout 与 candidate-set planning proxy 下的功能性 trade-off | `results/equivariance_functional_synthetic_20261003/` | `code/equivariance_interventions/run_equivariance_functional_synthetic.py` |
| Supplement Table 34 | global 与 action-conditional alignment 在随机 action-sequence rollout、ordered composition 和 candidate-set planning proxy 下的功能性后果 | `results/alignment_functional_synthetic_20261003/` | `code/alignment_interventions/run_alignment_functional_synthetic.py` |
| Supplement Table 35 / Figure S-phase | SIGReg 与 alignment 的交叉相位图；stable rank 用于检查 covariance flattening 与 alignment diagnostic 的混淆 | `results/sigreg_alignment_phase_20261003/` | `code/alignment_interventions/run_sigreg_alignment_phase_synthetic.py` |
| Supplement Table 35 / Table 36 / Supplementary Figure 2 | 全部 binary action sequences 的 exact operator-product error 与同 protocol 的 ID/OOD random rollout 关联 | `results/action_sequence_composition_20261003/` | `code/alignment_interventions/run_action_sequence_composition_synthetic.py` |
| Figure 3 / Table 25 | I-JEPA view displacement 与 source covariance eigenspace 的 overlap/lift | `results/jepa_feature_measurements/` | `code/run_ijepa_feature_anchor.py` |
| JEPA-WM temporal | held-out latent MSE、commutator 和 spectrum-preserving null | `results/temporal_world_models/` | `code/run_jepa_wm_temporal_eval.py`、`code/run_jepa_wm_pointmaze_temporal_eval.py` |
| DINO-WM temporal | 第二个实现和两个环境上的相同测量协议 | `results/temporal_world_models/` | `code/run_dino_wm_pointmaze_temporal_eval.py`、`code/run_dino_wm_pusht_temporal_eval.py` |
| Supplement Table S-cross-horizon | source horizon 到 target horizon 的冻结 latent readout transfer；dynamics/residual target specificity 与 covariance/random controls | `results/external_cross_horizon_readout_20261004/` | `experiments/empirical_spectral_audit/run_external_cross_horizon_readout.py` |
| Supplement Table S-cross-horizon-robustness | 三种 action normalization × 三种窗口密度下的 rank-8 paired cross-horizon contrasts | `../le-wm-main/results/pusht/lewm/external_cross_horizon_readout_20261004/cross_horizon_normalization_density_robustness_20261004.{json,md}` | `../le-wm-main/experiments/empirical_spectral_audit/summarize_external_cross_horizon_robustness.py` |
| Supplement Table S-cross-horizon-source-split | 两个独立 source splits × 三种 action normalization × 两种窗口密度的 rank-8 paired cross-horizon contrasts | `../le-wm-main/results/pusht/lewm/external_cross_horizon_source_split_robustness_20261004/source_split_cross_horizon_robustness_20261004.{json,md}` | `../le-wm-main/experiments/empirical_spectral_audit/summarize_external_cross_horizon_source_split_robustness.py` |

## 读表原则

表格中的 `Commutator` 是 covariance 与 predictor 的关系量，不是模型准确率；`Spectrum-null mean` 是保持 predictor 特征值不变、随机旋转 basis 后的参考值；`Latent MSE` 只是预测误差，不能替代 transfer geometry。
