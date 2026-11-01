# HiJEPA high rollout 精确layout诊断 v2

仅新增本stem source/tests/doc，不改冻结v1、loader、已失败report或任何旧结果。使用experiment-code固定来源/有界测试；worker只saved JSON/text/NPY算术和RAM合成view，不import torch、不运行神经/物理/网络，无CPU神经fallback/训练/拟合。测试不构成神经证据。

## 已保存失败与认证源码

绑定MAIN v1实际prepare SHA`879296f84ab3d09218a9699bd0eca47095fd8e0f1b3ded3e8158d1178a0756b9`、实际export verify receipt SHA`afec1d335e5380d740930a5e8ea9f9af4bfb445847d3821ad97c54dbd6c28069`、实际失败report SHA`3ebe0f5932aa80aff19f052b9d1da42a3a926b6397a6ea2a38a8b675d1583077`。prepare只复核这些现存saved身份，读取全部3对输出复算原失败摘要，不重做850数组export验证、不修改旧prepare。direct两项exact、rollout非exact；旧失败继续失败，v2不会追认其成功。

官方jepa.py文字SHA`bb5338bbb4f9e76b0bd15b94767624168883c994ab4c2ad8209337a3d70022a2`，`rollout_high`循环339..341：逐步`act_k=latent_actions[:,:,k,:].reshape(bs,d_l)`、`act_hist.append(act_k)`、`act_seq=torch.stack(act_hist,dim=1)`，再quantize identity/context slice。代码AST验证这三条准确语句和顺序。latent_action.py文字SHA`d06aa5955c5fdc6e61ec2f7a06317d024d8c4ad6ec6ceba7802e3d203e518e58`的quantize_latents只返回latents。source/IR/module.py由冻结loader.audit认证。官方**文字布局**支持append+stack；v1 reference的prefix slice+reshape仅代数等值，不是逐字layout忠实。原checkpoint历史bytecode身份仍未认证，不能因此宣称官方模型parity。

## 固定干预，不预判原因

仅原固定B2,S2,T5,z192,macro32，完全复用seed3 dyadic输入。不挑新样本、不换阈值、不恢复cost项目或宣称v1全四通过。

先记录frozen root原生rollout；再独立运行reference source-stack与legacy-slice各5步，保存每步z/action/full-context输出和全部layout（shape、stride、storage_offset、contiguous、dtype、device）。各自历史由各自产出递推，定位累计误差；不是只看终点。原生route与prior v1 owned、legacy route与prior v1 reference完整bytes对照，明确是否实际重现旧失败。

共同历史上每步固定五路：owned_source、reference_source、reference_legacy、reference_legacy_contiguous、reference_source_restrided。五路输入**值**必须同一固定动作/同一owned历史；只有owned_source输出喂下一步，不让legacy误差污染因果对照。reverse干预用清零storage+as_strided+copy重建legacy的stride和storage_offset，不改变任何源/模型。每路另实际调用同权重macro_to_condition functional linear保存projection，以区分首个可观测差异在projection还是之后；不是声称已定位所有内部层。完整每context输出、输入、projection都保存，不只hash/末步。

summary按实际数组逐字节复算：首个controlled legacy分歧步、每步different_elements/first_index/max_abs_error，source exact、legacy contiguous是否回归、reverse stride是否复现legacy、projection差异、三autonomous/手工source fullroute比较。`layout_intervention_support`必须同时满足同值/同历史、source reference等于owned、legacy不等、连续化回归、reverse-layout复现legacy；否则false，**绝不预填“已证明原因”**。即便支持也只限当前固定输入/库/设备，no_global_causal_proof=true。数值差异是诊断数据，不当工程异常中途停掉剩余固定对照；实际异常则partial保留，strict拒绝。任何路径都不放宽float32 bytes exact门。

## 实际计量、保存验收与边界

fixed高predictor owned10+reference30=40 API，batch-sequences160，context-token positions448；额外诊断projection25 API，与predictor或时间不能相加冒称统一预算。原生完整rollout顶层1dispatch、独立ref10、controlled预测25、额外projection25，共61顶层dispatch（上限65）。owned高predictor真实pre/post hooks、reference冻结函数meter、顶层attempt/return全记录，异常不退款、finally移除2hooks。pixel0/macro0/AD0/physics0。copy425独立数据装载，model_before/after全部原export值hash强等。

HOST CUDA only，CUBLAS workspace=:4096:8、deterministic true、matmul/cuDNN TF32=false。固定32MiB bundle/单saved数组1MiB/余量200MiB、600s边界wall（不代替MAIN外层watchdog）。所有写排他O_EXCL/NOFOLLOW/fsync，新独占目录；失败/负结果不删除。saved consumer读取全部descriptor并核文件完整性、float32 finite、layout contiguity算术、全部输入配方、5步历史/action/order/fullcontext/join、projection/输出完整分母、meter/来源/model/设备与summary复算，不运行CPU模型；它不能凭JSON独立证明历史调用。

不重做direct/CEM/架构/encoder/macro研究；不训练，不读取新outcome择根。`diagnostic_complete_not_release`仅表示固定诊断完整保存，不等于exact fidelity通过。`official_parity=false, baseline_runnable=false, scientific_admission=false`恒定，v1失败不释放。

## MAIN命令

cwd `/home/gbk/code/LeWorldModel/le-wm-main`。worker未prepare/CUDA。prepare输出真实manifest SHA，run输出真实report SHA，两个变量需MAIN填实际返回值，无未来hash默认。

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_hilewm_trusted_cuda_validation_20261028_v2.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_hilewm_trusted_cuda_validation_20261028_v2 prepare --output results/agent_hilewm_trusted_cuda_validation_20261028_v2_prepare_MAIN01
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_hilewm_trusted_cuda_validation_20261028_v2 verify --manifest results/agent_hilewm_trusted_cuda_validation_20261028_v2_prepare_MAIN01/manifest.json --manifest-sha256 "${HILEWM_LAYOUT_PREPARE_SHA:?MAIN actual prepare SHA required}"
CUBLAS_WORKSPACE_CONFIG=:4096:8 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_hilewm_trusted_cuda_validation_20261028_v2 run --manifest results/agent_hilewm_trusted_cuda_validation_20261028_v2_prepare_MAIN01/manifest.json --manifest-sha256 "${HILEWM_LAYOUT_PREPARE_SHA:?MAIN actual prepare SHA required}" --output results/agent_hilewm_trusted_cuda_validation_20261028_v2_cuda_MAIN01 --main-authorized
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_hilewm_trusted_cuda_validation_20261028_v2 consume --manifest results/agent_hilewm_trusted_cuda_validation_20261028_v2_prepare_MAIN01/manifest.json --manifest-sha256 "${HILEWM_LAYOUT_PREPARE_SHA:?MAIN actual prepare SHA required}" --report results/agent_hilewm_trusted_cuda_validation_20261028_v2_cuda_MAIN01/report.json --report-sha256 "${HILEWM_LAYOUT_REPORT_SHA:?MAIN actual report SHA required}"
```

本三件套冻结后修复必须新版本，不改运行依赖或原失败。
