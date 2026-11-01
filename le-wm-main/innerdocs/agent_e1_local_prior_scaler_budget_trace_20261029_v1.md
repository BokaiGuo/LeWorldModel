# E1 本地 FIRST / scaler / 预算追踪：只读结果

仅新增本文；不修改旧文件，不运行生产、神经、物理、fullstrict，不拟合或训练。范围是已明确的 known2/off E1 与 Hi 已存资产；不是全磁盘、全部历史 revision 或所有作者资产的穷尽搜索。下列 SHA 是本轮直接对文件字节运行 `sha256sum`，receipt 是 MAIN 既有完整消费的持久证据；本轮没有重新执行其 fullstrict。

## 1. 可实际接线的增量：两个同 blind 输入的 FIRST

旧 E1 controller prepare 的 `/slots/0` 和 `/slots/1` 的 `blind_sha256`，与 Hi FIRST integration manifest 的 `/slots/0` 和 `/slots/2`（均为 off）逐字相同；相应 NPZ 文件本轮逐字节 hash 也相同。Hi manifest SHA `a3255d38e9cf718f9560c1b9e381abe38750edd9e91d3f04e5ad332cf414d7e9`，Hi report SHA `083d0d9aa190b561e6ccf865b09d7224b97d404b16ba38c792bd8af386f5622a`，持久 receipt SHA `40f28414682c9540e9e195b361a36dfc514f816358adbeacb379fedc460e4ef0`。

| 原槽 | E1 task 文件 SHA | 两者共同 blind NPZ SHA | Hi report 中 FIRST 文件及**文件** SHA | receipt 的 FIRST **raw tensor data** SHA |
|---|---|---|---|---|
| root0/off | `a42bfbb1ea692782355e2572a5df2cd048b72cf3e6c41fee40eb12503674959b` | `3c749e224e075061353dee590aff5883b202854451ef95f9bbf951510e0f6f0c` | `array_00215.npy`: `54db4544b0cc9a9723c4d89a3e283a419cb22cbb00d6b6957404db560a8912a6` | `966b13008bc83b46b2b35258cc22a40583a13f005967c0b8f03d62f7aed4332f` |
| root1/off | `101add21a7a40591167a6ab8e8b22c3c62cb1fd700c741fd36cd2073e2d5df74` | `4e9aeb8504db653552c4a9924cbc0c3ddfc53f1e6f3a56e5b73fc6c8f1b3be39` | `array_00697.npy`: `5a6b0416eb9ac212c777fa3684ab93e1611c57ff50103501d8847e3f22b5c1d2` | `a070915118c5a8b98a7fefc8f0265793880392489658f12a8fa946427564c53c` |

完整路径相对于 `/home/gbk/code/LeWorldModel/le-wm-main`：

* E1 prepare：`results/agent_experiment1_common_controller_20261028_v1_prepared_MAIN01.json`，SHA `8c472485d44c40ce3d95f8043d2c58f9ec2694fef74786a840d1699d5a1a86bd`；task 分别为 `results/p1_constructive_moving_history_encode_cuda_20261027_v1_cuda_main01/task_00.json.gz` 与 `task_01.json.gz`。
* 两个 blind 为 `results/p1_moving_target_overlay_source_20261027_v1_main01/s0_off_blind_inputs.npz` 与 `s1_off_blind_inputs.npz`。
* Hi manifest：`results/agent_hilewm_rgb_high_low_20261028_v1_prepare_MAIN01/manifest.json`；report 与两 FIRST 文件在 `results/agent_hilewm_rgb_high_low_20261028_v1_cuda_MAIN01/`；receipt 为 `results/agent_hilewm_rgb_high_low_20261028_v1_consume_MAIN01.json`。

report 的 `/rows/0/high_target` 和 `/rows/2/high_target` 引用上述文件，`/arrays/<文件名>` 描述 `<f4`、`[1,192]`、896 文件字节及两个独立 SHA；receipt 的对应 `/rows/0` 与 `/rows/2` 记录 `target_delivered_exact=true` 且 high target 和 low goal 的 data SHA 同上。这里核了保存文件 SHA、manifest↔E1 的 blind 字节身份和 receipt 声明，**未独立解 NPY 重算 data SHA、未重做全部原 strict**。不把 off/on 混合或按结果改槽，也不据两个 target 的存在宣布已完成 E1 三臂。

`results/agent_p2_hilewm_computation_parity_20261029_v1_consume_MAIN01.json` SHA `3555696c02329062084d791c0c40958084d7352e6f7680938bade0068afe9480` 对固定 known2 输入记录 `all_values_exact=true`，同时 `normalizer_pairing=false`、`raw_deployment=false`；不是全部输入/全架构等价证书。`results/agent_p2_history3_tracker_cuda_20261029_v1_consume_MAIN01.json` SHA `080fe1173895621ea4c0565671509a267ba456135fedfac313ddac7894f98138` 为实际 normalized history3 tracker：24 predict API、108 predictor samples、0 新 encoder/AD/physics/high，raw inverse=false；其目标和调用合同须逐项核，不能把该次 tracker 改标为这两个 FIRST 的完整共同控制实验。

## 2. 实际 raw/scaler 数据与仍不能证明的事

task_00 原始导出 `/task/action_mean` 为 float32 数值 `[-0.008700000122189522,0.006800000090152025]`，`/task/action_std` 为 `[0.20190000534057617,0.20020000636577606]`，`/task/root_raw_action_history` 有 10 个原始二维动作。task_01 的实际文件 SHA见上表；未来接线须对两文件分别解析、检查字段及动作完整来源，不能仅从这段文字复制统计。源 `experiments/run_pusht_counterfactual.py` SHA `0a4eb4592951139e596b35becede17d175df7b48a2fde16679ab0aac7c4dce34` 给 P2 公共工程常数；该合同不证明它是 Hi 原 sklearn process。原 Hi 官方 eval 在运行时对 `pusht_expert_train` 非 NaN action 列拟合 scaler；`cache_dir:null` 没有给当时 resolved cache、数据 revision、保存的 process bytes 或数值。

本轮只核了以下明确候选路径不存在：`/home/gbk/.cache/stable-worldmodel`、`/home/gbk/.cache/stablewm`、`/home/gbk/code/LeWorldModel/le-wm-main/data/stablewm` 和 `.../data/pusht_expert_train.h5`。它们是定向候选检查，**不声称它们是实际 resolved cache，也不证明其他地方没有原始资产**。已封存镜像 tree 只显示大数据 archive 和小配置，不包含外置 scaler sidecar；非官方历史穷尽证明。`results/agent_e1_hilewm_time_scaler_contract_20261029_v1_prepare_MAIN01.json` SHA `c5ceb908e95a092519dcb9626d3ea9d07bca1b387add25b3410edc612b342e7c` 已证实 source 对象流但非历史数值。

原 inverse 的可认证输入仍需当时已保存 process 的数据导出、原 bytes SHA、`mean_/scale_/var_`、flags、dtype、feature 顺序和单位、实际数据/版本及该 checkpoint/eval 的生成与使用证据。不得将 P2 常数称为它，也不得在本任务重新 fit。若 MAIN 事前单独批准全三臂共同采用 P2 工程归一化，这是一个**新的** fixed-domain 模型校准协议；仍需检查各臂实际 source→normalize-once→model token→raw domain 的字节和返值、逐个付费重放，不会恢复原 Hi 历史 scaler 或证明训练分布一致。

## 3. 可落实的 known2/H10 新共同接口及预算边界

不要求把上述 Hi FIRST 假称有固定 physical cp5/H50 时间，也不强推新的 D50 数据。最小候选是事前冻结 root0/1 × arbitrary / learned-FIRST / compiled 六槽，所有臂使用相同 task、history3/两真实 incoming token、goal、P2 冻结 backbone/原 Q、H2/10 raw 控制、同 controller/同 seed 规则；只将 FIRST 作为无时间标注的 target。以既有 receipt 的已付 FIRST 作为候选输入时，须把上表 manifest/report/receipt、两组文件/data hash 和此前 high/pixel 费用作为 inherited 成本列清，不能把两个已付 target 声称新生成免费。COMMON controller 的 `ALL targets before model loading and any controller dispatch` 需要新的六槽 seal；原源码仅枚举四分支，必须新实现并审阅，不能靠旧 CLI 偷加 learned。

若 MAIN **明确预先批准完整六槽均重新执行原 3×24 H2 controller 配方**，来源 `experiments/spectral_reachability/agent_experiment1_common_controller_20261028_v1.py:59–79` 给每槽 72 candidates ×2 packets = 144 predict API、144 predictor samples、144 action encoder、144 pred projection，六槽分别 **864**；不是旧 576 加免费旧执行。仅在另行明确批准保留旧四槽结果并接入两新槽、且完成同源同控制消费时，增量两槽对应 **288**，历史四槽的 576 另列。两种设计不能混称：旧四槽不是同一新 job 的重新执行，proposal 支出并不匹配；旧 controller 未 replay winner，若要求新 whole/segmented/实际 raw token replay，要另行逐轴增预算，不从 864 里假装已付。

继承的 compiled search 在旧 controller 合同里是每根 144 predict API/samples，原 arbitrary 生成 20×192 正态候选为 CPU 数值搜索、无 model calls。Hi integration 全四槽合计 encoder 8、high_cost 12、high_predictor 56（input rows 392）、selected high rollout 4、low_predictor 8、physics 0；这是全四槽实际旧账，不可未经逐事件核对擅自除二当两 off 槽的账。高/低 predictor 不加总成等 FLOPs。完整新 job 的 high/proposal、whole/seg replay、实际 token、encoder、真实 batch/module、AD、native constructor/reset/warmup/render/close、磁盘/内存和墙钟边界尚未事前固定；旧四槽 64MiB/900s 的存储证明也不能外推六槽。**因此当前没有可认证的完整三臂端到端硬预算或生产命令。** 缺项必须按新 runner 实际 shape/journal census 和各 proposal 算法预先核定，失败不退款，不可把未知填0。

旧 E1 物理四槽 full-horizon receipt 的 80 public/816 physics/88 PNG 是当次实际开销，不是六槽新预算证明。新协议如需物理，必须新 adapter 并分别核原生 reset/step/早停计数、动作域、颜色/RGB 和 renderer，保留六分母；动作见证、模型 tracking 和实际原生到达仍分开，旧 P1/science gate 继续 false。

## 4. MAIN 最小下一步

先选择并事前封存上述 **全六重新执行** 或 **两槽增量**，以及是否采用明确命名的 P2 工程归一化（不是 Hi 历史 scaler）；若不接受该新协议，提供可认证的原始 scaler export。随后审新六槽 source/计量和完整 byte census：两 FIRST 文件逐 NPY/header/data/target/low-input 验证，全部 target 先封，再同一 controller 算，returned raw 合法和 roundtrip 后额外付 replay，最后另行授权物理。不得从旧结果挑根、动作或预算；不得把模型目标当已物理到达。此文只定位资产和缺口，没有实现/授权上述运行。
