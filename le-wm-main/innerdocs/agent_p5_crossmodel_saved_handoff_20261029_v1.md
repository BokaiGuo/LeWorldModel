# P5 cross-model / multi-environment saved handoff

2026-10-29；仅本新文件。capture v1/v2、MAIN文件及旧结果均未编辑。没有新神经、物理、下载、checkpoint反序列化、训练或拟合。本文件不是新生产协议/ready manifest/科学准入。

## 可复用闭包：不要重复实现

已有 `experiments/spectral_reachability/p5_jepa_panel_audit_20261027_v1.py` 完整saved-only入口：`prepare`返回stdout JSON，**不写manifest**；`audit --manifest PATH`读取现成manifest。其固定三个seed2026102702/03/04，原生0/5/10历史、真实15标签，四固定action arms；不是当前LeWM164面板、不是同根跨架构控制比较。源码显式核原collector、journal、数组及模型收据；不运行模型或环境。

现成saved receipt已在本轮读取并对全文重算SHA（不是本轮重新运行全部consumer）：

| 已存在文件（相对inner根） | 文件SHA256 | 读取到的既存状态 |
| --- | --- | --- |
| results/p5_jepa_panel_audit_20261027_v1/fidelity_audit_saved_v1.json | 09656b9a08df6437971d3128ebd98ba56ef533d855593b9c78fbfb1d8b1617c2 | saved_panel_fidelity_audited；all_checks=true，1205项；C2/P5=false |
| results/p5_jepa_panel_audit_20261027_v1/physics_audit_saved_v1.json | 15b25541625555ea611a2bebb888cef72a40a006c0f7e37d14ccebbeb77eb13b | saved_panel_physics_audited；all_checks=true，1168项；C2/P5=false |
| results/p5_jepa_panel_audit_20261027_v1/manifest_fidelity_frozen.json | 51de60b227ce5f66bd7c86c613e8fa15c9771d5f7fbec69633e8f00efddce37c | 已有fidelity seals，不能覆盖 |
| results/p5_jepa_fresh_fidelity_main_20261027_v12/manifest_frozen.json | 41ef9bf8f5c5ae5c4ff8890a25e6a83137b7e637058a8033281a5eabc9cea07f | 上游manifest磁盘SHA |
| results/p5_jepa_fresh_fidelity_main_20261027_v12/cuda_main_v1/report.json | d8667a4e8537a5e005d13507d4d05b8d7907996549889decc8a00e1db873eda6 | fresh_native_panel_fidelity_recorded |

注意report的manifest_sha256字段是原semantic digest，不等于上表manifest文件SHA；不可互换。模型/normalizer跨模型身份也不能以同名字段推定相同。

## 立即可执行入口（无需新实验）

cwd `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
.venv/bin/python -m experiments.spectral_reachability.p5_jepa_panel_audit_20261027_v1 audit --manifest results/p5_jepa_panel_audit_20261027_v1/manifest_fidelity_frozen.json
```

该入口仅stdout，不持久化新回执；不要覆盖旧receipt。若源已漂移，现有consumer应拒绝，而非重seal。MAIN如确需fresh复验可执行；本worker未重复运行已完成1205项消费，也没有重新prepare。

## 多环境实物存在，但不能偷换成同协议复制

已读取以下实际JSON并计算全文SHA，未加载其数组/数据集或执行模型：

- `results/enriched_jepa_pointmaze_observed_cuda_fixed_20261011/metrics.json`：4cf4a352b2e41a53dbbf65cbeecf9c085f3614313aa56bd554f4c7b309fce0af；包含protocol/evidence_boundary/enriched_records及operator_fit_vectors字段。
- `results/theory_operator_alignment_iclr_oral_20260829/wall_temporal_gate/jepa_wm_wall_observed_metrics.json`：5c2af67a2e8e690af6f1bde012d4bc32d19640bf9de7c8247328affd8a3b5314；包含protocol/evidence_boundary/operator_fit_vectors。

对应既存入口为 `experiments/theory_spectral_transfer/run_jepa_wm_pointmaze_temporal_eval.py` 和 `run_jepa_wm_wall_temporal_eval.py`。它们是历史temporal评估程序，**不是当前可授权的P5 native raster复现runner**。例如PointMaze入口有device auto/cpu及visual calibration/operator fitting路径；不能本轮直接启动或把其统计当冻结Q真实控制收益。此处不发伪装可直接部署的CUDA命令。

`results/p5_architecture_provenance_20261027_v1.json` 当前SHA为6c169f0685bb227913ae6deb821d74093e15eed29f063584e919bae892cde7b0，已有三任务JEPA权重/配置、source、normalizer及学习成本来源说明。三份任务checkpoint属于同一JEPA架构，不能计为三种独立架构；该旧合同仍显式不完整，不为本轮回证loaded bytes。

## 真正缺失的前置及最小下一实现

1. **跨模型同协议raster输入桥尚未在本次固定资产集合中绑定**：需要MAIN预声明使用哪个已完成物理面板、全部root/condition/packet身份与同一观测序列，以及JEPA所需agent proprio权限。当前LeWM164与JEPA三seed不能按row序或相似seed拼成matched pair。复用物理RGB可以避免重跑物理，但新的JEPA encoder/predictor证据只能由MAIN后续实际CUDA产生，不能由saved-only重命名得到。
2. **跨环境缺同一问题的原生任务合同**：PointMaze/Wall历史temporal指标没有在上述P5面板下封存真实own-action→successor→native reward/success及reset/action meters。必须先选定既存合法collector/数据源及原native成功定义、动作packet语义、观察权限，再实现对应输入适配；本轮没有授权擅定新根、配额或成功门。
3. **架构/来源边界独立保留**：JEPA RGB+proprio与LeWM RGB的输入权限不同；各原normalizer/权重/模型输出维度、学习成本来源须分栏，不直接比较384patch与192latent的MSE数值或把动作arm差分称物理反事实真值。外部来源与许可、normalizer统计lineage等unknown仍沿旧合同披露，不要求P1通过才准备接口。

最小有意义的新实现应是MAIN指定固定面板后的一次JEPA输入adapter，复用原原生loader和非零历史、逐episode绑定RGB/proprio/动作；所有预测先封后读successor标签。这里不提前发明该新协议或预算，不重复旧registry/consumer。当前安全可新增工作止于这份准确交接；不能仅用现有saved结果补出跨模型/多环境机制或真实收益认证。

## 本轮验证范围

实际运行：现成源码/CLI只读定位、上述7份JSON的metadata读取和SHA计算；没有调用prepare/audit/NN/physics。未新增源码，因此没有新增单测；文档所列入口只经源码核对，未声称本轮fresh strict通过。所有科学admission保持原false。capture v2正在MAIN生产，不读其partial来推断终态。
