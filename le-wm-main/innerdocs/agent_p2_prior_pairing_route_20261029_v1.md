# P2 / Hi learned-prior 配对：两条最短接线路线

日期：2026-10-29。本文唯一新增文件；代码路径相对 `/home/gbk/code/LeWorldModel/le-wm-main`。只读已明确的小型源码/config/metadata，不扫描资产目录，不加载权重/NPY/pickle，不联网，不运行 NN/physics/full strict。以下“复用”指源码接口可复用，不表示新接线已通过。原封存文件和 `agent_p2_subgoal_gap_protocol_20261029_v1` 均不改。

## 结论与选择顺序

**先做现有 metadata 上的 embedded-low↔P2 配对核对，再由 MAIN 决定路线，不必先下载新 checkpoint。** Hi 原训练 config 声明从 LeWM 引入并冻结 encoder/low predictor/action encoder/projector/low projection；这使逐项配对值得做，但不是 tensor 相同的证据。当前不能断言“两者必然不同”，也不能凭同为192维宣布相同。

若需要最短、无跨坐标映射的真实模型子目标实验，路线 A（三臂统一到 Hi）更直接：已有真实 high-FIRST 和 normalized low-CEM，先接**模型坐标下**的三臂 target delivery / predicted tracking；它不需要 raw inverse，但还不是物理可执行性或 E1 成功实验。完整 native 三臂实验仍受原始 scaler 资产约束。路线 B 保留 P2 必须额外证明坐标/计算合同一致；不得用新训练 mapper、线性 readout、单位阵或192维 cast 代替。

原提案 `pasted-text-1.txt` §7（527–570行）要求 arbitrary 是直接 latent 优化、prior 是已有训练 proposal、compiled 是动作响应导出的中间状态；还要求低层实现偏差和 native 任务成功。因此两条路线都不能拿现成 final goal/random vector 充 arbitrary，或只报 latent 距离就称完整验证。所有下一实验均固定原两 known/off roots，不是独立确认；科学 P1/P2 gate 继续 false。

## 1. 当前可以直接定位的接口及 shape

| 组件 | 精确 source / 函数 | 实际合同与复用边界 |
|---|---|---|
| Hi 数据与可信构造 | `agent_hilewm_storage_npy_export_20261028_v1.py::state_dict_paths`；`agent_hilewm_trusted_loader_20261028_v1.py::graph_spec/check_export/authorities/construct_cuda`（均在 `experiments/spectral_reachability/`） | 425 storage 的安全导出与 literal key/alias inventory 已有；构造只允许 MAIN CUDA。模块是自建受信实现，不是 foreign pickle/class parity。loader v1 自身拒绝的 pixel forward 不可直接调用 |
| Hi pixel | `agent_hilewm_rgb_high_low_20261028_v1.py::config_contract/image_stats/preprocess/owned_pixel` | uint8 `[224,224,3]` → float32 `[1,3,224,224]`，先 /255、ImageNet Normalize、224 identity Resize；patch14、257 tokens、192维、CLS→projector `[1,192]`。原 pixel-equation 实现独立于 loader 的 blocked forward |
| Hi high proposal | 同文件 `high_search(t,CEM,Box,root,z,goal,trace,row,index)` | `z/goal=[1,192]`，high CEM init `[1,2,64]`，macro32×block2；输出首 receding block `[1,2,32]`，调用 `root.rollout_high`，**FIRST** `roll[:,0,0,:]=[1,192]` 实际送低层，非末 waypoint |
| Hi low controller kernel | `agent_hilewm_low_cem_20261029_v1.py::low_shape/OwnedLow.get_cost/search` | solver前 info 三张量 `[1,1,192] / [1,1,10] / [1,192]`，metadata `planner_level='low'`；sample扩展后 `[B,S,1,192] / [B,S,1,10] / [B,S,192]`，本实现 B=1，S≤64，H≤16，candidates `[B,S,H,10]`。先追加 outgoing action→action encoder→ctx=min(3,…)→low predictor/proj；输出完整 `[B,S,H,192]`，terminal squared **sum** cost |
| Hi solver metadata桥 | 上述 `LowBatchBridge`、`search` | installed CEM 只接 tensor info；在 cost 边界补 metadata，不能将字符串 `low/high` 作为 batch 切片字段。已有 low search 固定 S8/H2/iter3/topk3，不是通用三臂/H3 runner，不能原封不动称本次预算 |
| P2 root/context | `agent_p2_contextual_composition_runtime_20261028_v1.py::task_bindings`；`p1_constructive_moving_history_encode_cuda_20261027_v1.py::decode_blind/check_packet` | blind RGB `[3,224,224,3]`、真实 raw warm10 `[10,2]`、goal RGB；P2 context latent `[3,192]`、between-action `[2,10]`。原 Q=identity192，terminal **mean** MSE |
| P2 model/transition | `agent_p2_equal144_baseline_cuda_20261028_v1.py::loaded_model`；`pusht_full_packet_compiler_20261027.py::make_transition` | CUDA `[1,3,192]`+`[1,2,10]`+raw `[5,2]`→滚动 full context；normalization一次，三模块原输入/输出全部可封存。该 transition **不能**直接塞入 Hi 的 single-z/zero-history ABI |
| P2 compiled | `agent_p2_three_program_20261028_v1.py::plan/HostBackend._rollout/jacobian/model_plan/consume` | 原42候选、H3 raw15、whole+segmented replay和三条完整 context 边均已有；`plan`硬绑定P2 backend/context与固定预算，迁Hi必须新适配，不能只换模型变量 |
| native / actual RGB | `agent_p2_three_program_physical_20261029_v1.py::episode/execute/validate_saved`；`agent_p2_three_program_rgb_20261029_v1.py::fixed_rows/process/analyze` | 原recipe/完整prefix与cp5/10/15来源/动作时间join可复用；旧physical release只接受旧compiled plans，不接受新三臂随意换绑。旧RGB编码器绑定P2模型，迁Hi只能复用其身份/分母结构，不能复用旧latent |

表内除首行外短文件名均位于 `experiments/spectral_reachability/`。低层入口精确签名为 `search(t,CEM,Box,root,trace,row,z,target)`；复用其算术/trace/consumer，不能借调用旧 `build/upstream` 获得新实验授权。旧 `upstream` 硬绑原四槽及已付 high target，不能将旧报告换绑到当前六分支。

## 2. 原始 normalizer：有精确“生成入口”，无已定位“原 stats 文件入口”

1. **Hi 真正生成位置**：`results/p1_hilewm_text_contract_20261027_v1/texts/package__code__h_le_wm__eval__hierarchical.py.source.txt:309–331`，`get_dataset` 根据 `cfg.cache_dir`/默认cache定位 `HDF5Dataset`；`build_process_map` 对 `cfg.dataset.keys_to_cache` 各非pixel列取全列、去任一NaN行、`StandardScaler().fit(col_data)`。505–509将 process 传给 policy。这是现场计算，不是读取 frozen stats。`package__code__config__eval__hi_pusht.yaml.source.txt:14–19,93–99` 给数据名 `pusht_expert_train`、action/proprio/state，但没有固定 dataset revision 或原数值文件名。
2. **消费位置**：`package__code__h_le_wm__planning__policies.py.source.txt:890–895`，primitive reshape 后 `process['action'].inverse_transform(action_np)`。需要原 mean_/scale_、flags、dtype、feature顺序、raw单位及 checkpoint/data/eval 配对；不能猜正负号或额外 /100。
3. **当前明确拒绝点**：`agent_hilewm_low_cem_20261029_v1.py::inverse_transform`（228行）对所有输入拒绝，`raw_actions=null`。新资产到位也须新版本实现并审查；caller自称 certified 不使旧版本放行。
4. **P2 确实有数值入口，但不是 Hi 原 scaler**：`experiments/run_pusht_counterfactual.py:34–35` 的 PUBLIC_ACTION_MEAN=`[-.0087,.0068]`、STD=`[.2019,.2002]`（float32）。source SHA `0a4eb4592951139e596b35becede17d175df7b48a2fde16679ab0aac7c4dce34`；`p1_constructive_blind_encode_cuda_20261027_v1.py::native_metadata` AST读取并绑定，显式 `training_provenance_verified=False`；`pusht_d2p_packet_contract.py::normalize_live_packet` 保留“public constants are not checkpoint truth”。它是原P2工程协议，不得改标成 Hi/sklearn 原始精度 stats。
5. **已保存定向资产调查**：`innerdocs/agent_hilewm_normalizer_asset_route_20261029_v2.md`及两个固定目录 `results/agent_hilewm_normalizer_asset_route_20261029_v2_{mirror01,small_metadata01}/report.json`。实际 tree只列dataset大训练包和model weights/config/README，无独立process sidecar；小config无 scaler。报告SHA分别为 `f8eeae02c04ef66449243f687c199799c117fb438da9922451bdd1853d6d2256`、`1de9c82ac42613de2973d0d5cdb4d83b34239612dfab6b92aba85f889f423034`。这些是镜像快照，不穷尽官方历史；本轮无新探针。

**当前答复：原stats的实际文件路径/成员名/SHA仍 unknown。** 不提供猜测的 `normalizer.pkl` 或默认cache路径当可用输入。最小外部输入是作者当时保存的 data-only process 导出与上述 provenance；若当时未保存须明确，不能在本授权内重新fit。现有 `BatchNorm1d` 是神经层，不是 scaler。

## 3. 路线 A：三臂统一到 Hi backbone（最短模型内接线）

### 必须一起迁移的量

仍用原两known/off root的 blind RGB/goal和原native recipe，但所有臂共享 Hi pixel、projector、low predictor/projection、action encoder、预处理、raw inverse、观察/反馈规则与controller。不沿用P2 latent/Q终点数值；新identity192 metric需明确属于Hi坐标。旧已付Hi四槽末帧与同一source确实逐字节匹配时可作继承输入，历史成本保留；不能把别slot或旧计算称新预算免费工作。

最短版本采用**原Hi policy单末帧z+zero grouped a_hist**，三臂全部相同；这改变了P2三帧causal协议，命名为Hi-backbone校准，不称原P2复现。若要三帧history，须新实现并认证Hi context delivery，不能调用目前 `low_shape`（硬拒绝history>1）或靠reshape蒙混。

### 最短实现顺序（尚未实现；不是已有CLI）

1. 新source复用 `preprocess/owned_pixel/high_search`、可信构造及已有 trace/store：接受实际external prepare/report/whole-receipt SHA；先重建完整上游 strict，再按固定root索引交付 z/goal。worker本轮未运行此步骤。保全高层原macro、完整rollout和FIRST到controller的byte equality；不重新选择target index。
2. 在同一Hi low kernel上新增**H3 normalized transition + whole/segmented replay接缝**。`OwnedLow.get_cost`已容许H3，但只返回cost、保存rollout，不是现成differentiable full-context transition；把其append/ctx/AE/predict/proj链抽成新受审函数，保存z_seq/a_seq和每步数组，不patch旧root。不能直接用带inference_mode的官方rollout做编译Jacobian。新纯函数与原kernel须MAIN固定输入逐步核对后才用于D2P编译。
3. 实现三种proposal：arbitrary显式优化 `[3,192]`，事前固定非退化目标/初始化/迭代/可行域（不能只min到goal的距离而三点全等goal）；compiled在同Hi transition上编译三包并保全见证；learned保持已训练high-FIRST。FIRST不自动等于cp5/10/15三个target：采用每包真实RGB后重规划一个FIRST的共同执行协议，或另行预声明三target生成规则，必须MAIN先选，不能把高层macro索引冒充确定物理5步。两种实验不要混表。
4. 所有臂经同一个low tracking controller，不给compiled raw witness warmstart。复用 `LowBatchBridge/search/validate_native/validate_cem` 的安装CEM算术与全attempt账，但新预算新config，保存返回mean的独立付费replay（旧search明确final_mean_evaluated=false，不能冒认旧elite-average为其cost）。先只验证normalized轨迹和target-delivery，**不输出raw/physical可执行率**。
5. 原stats与raw inverse认证后，才实现normalized `[H,10]`→time-major `[5H,2]`→原raw的单次逆变换，并新三臂native handoff/同encoder cp反馈。若同域要求native `[-1,1]`，须事前定义共同raw参数化/可行域，不能把CEM的Box元数据当clamp、事后clip或按物理结果修动作。全6分支/18cp、earlystop缺失与native success保留；不借旧4plan终态。

这条路线消除了跨backbone mapper需求，**没有消除stats、真正arbitrary optimizer、共同controller与compiled新transition的实现工作**。能立即推进的最小源码slice是第2步的normalized Hi transition/replay与RAM shape/时序测试；MAIN随后做CUDA方程接线检查，而非提前做物理发布。

## 4. 路线 B：保留原P2 backbone，认证 embedded-low 对齐

证据起点是 `.../texts/package__checkpoints__pusht__main__config.yaml.source.txt:59–73` 的 `pretrained_low_level`（path=`data/stablewm/pusht/lewm_object.ckpt`、freeze flags）及 `package__code__h_le_wm__models__jepa.py.source.txt::freeze_low_level/train/_encode_pixels_sequence`。它声明意图，不证明实际权重未变或就是当前checkpoint。

1. **先metadata、后数值，不先启动模型**。本机 P2 config精确文件 `/home/gbk/.stable_worldmodel/checkpoints/pusht/lewm/config.json` SHA=`2564086e961e7b5c7c04dffc451091115b389a590645ff19653c64fd0bc16e09`，与已存镜像 `metadata_02.bin`字节相等。旧P2 source绑定weights SHA=`48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607`，本轮未读weights。这仅关联候选发布资产；config相等不证明Hi embedded tensors相等。
2. 下一最小认证实现是固定五组mapping：Hi `encoder→encoder`、`projector→projector`、`action_encoder→action_encoder`、`low_predictor→predictor`、`low_pred_proj→pred_proj`。用安全export `state_dict_paths`、loader `graph_spec/check_export` 列出所有参数/buffer、dtype/shape/stride/alias与nonpersistent flags；对照P2确切key inventory（如无data-only版须另受审导出，不能worker load weights）。ViT legacy key变换只沿 `experiments/checkpoint_compat.py::remap_legacy_vit_keys`；不能猜QKV拆合。逐key原值字节/hash需真实等值检查；prefix名称对不上、缺buffer、数值不同即停，不能平均/fit/放宽阈值。这里尚无已完成配对certificate。
3. 即使五组逐tensor相同，还要认证两个**计算接口**：Hi owned_pixel vs P2原encode的预处理、CLS/projector、精度/SDPA/layout；以及low状态/动作时序。P2固定 `[1,3,192]+[1,2,10]`，Hi官方low起于单z/零history；source rollout_low:354–441可接受明确4D `[B,S,3,192]`和`[B,S,2,10]`，但当前OwnedLow seam未实现该入口，必须新context adapter，不能认为source“接受”即已认证运行。固定真实causal输入由MAIN逐层核对，保存差异；若不等，不能称当前P2同backbone。
4. 高层只见同一个当前末帧/goal；三帧history仍完整交给共同P2低层。这是公开披露的高层输入选择，不把single-frame称history3；必须证明high输出与P2目标同坐标，并实际byte delivery。高层训练依赖的action scaler与P2公共常数仍需provenance一致性，tensor等值本身不认证stats；否则只可报告“候选坐标配对通过，原raw统计配对unknown”，不能签完整route。
5. 认证闭合后，新接口把high FIRST接原P2 `make_transition`上的统一tracking controller；原P2 compiled kernel可复用，但旧138/936预算产物不是新三臂等预算输入。仍需arbitrary optimizer及共同预算重生成。若五组不等，此路线不在本授权内靠mapper补救，转路线A或等待真实compatible prior。

## 5. 同预算、度量与实际停点

沿已冻结subgoal协议，由MAIN结果盲固定 `predict_API_calls_matched` 或 `predictor_forward_samples_matched`、cap/batch和精确low callable。计入proposal/controller/AD primal/whole+segmented replay/失败attempt，失败不退款、不padding；已付source编码按共享cohort一次列明。Hi high predictor API/sample/token另列，不能与low API相加后声称同world-model调用；需要匹配全部模型边界时须另预声明分配。encoder、projector、AE、backward/J/VJP/SVD、physics、CPUcapture及wall各列，FLOPs unknown。

固定cp5/10/15 actual encoded-to-delivered-target误差、predicted-to-target误差、native指标独立列；Hi low原cost是192维sum，P2 Q是mean，不能混列或隐式把sum当原Q；若新共同metric取mean需三臂一致事前声明、原cost仍保存。realizability阈值缺失则仅连续偏差、比例null；不能按本次追踪分布设阈值。compiled witness证明模型动作生成，不保证物理reachability；不能将known-root工程校准称独立确认或P2机制成功。

**MAIN可以立即分派的最小工作**：A的normalized full-context transition/replay；或B的五组data-only tensor/config inventory join。两者不需先下载新模型，不应先写更多泛manifest。若最终目的必须native gap，原始stats仍是独立外部输入；本轮未定位到合法文件，故没有宣称完整三臂可运行。

## 6. 本轮检查与复核命令（仅保存文件/源码）

本轮实读上述源码、原提案§7、小config和两份已存mirror报告，复核下列SHA；未读weights/NPY/archive，未调用任何consumer。没有新增测试代码或生产CLI。本doc不是运行证据；旧39项fixture无需重跑来认证配对。

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
sha256sum experiments/run_pusht_counterfactual.py /home/gbk/.stable_worldmodel/checkpoints/pusht/lewm/config.json results/agent_hilewm_normalizer_asset_route_20261029_v2_small_metadata01/metadata_02.bin
cmp -s /home/gbk/.stable_worldmodel/checkpoints/pusht/lewm/config.json results/agent_hilewm_normalizer_asset_route_20261029_v2_small_metadata01/metadata_02.bin
rg -n '^def (low_shape|search|inverse_transform)|^class (OwnedLow|LowBatchBridge)' experiments/spectral_reachability/agent_hilewm_low_cem_20261029_v1.py
git diff --check -- innerdocs/agent_p2_prior_pairing_route_20261029_v1.md
```

没有可合法复用的现成三臂HOST CUDA完整命令；不提供会误运行旧四槽实验的别名。上面是现在真实可执行的只读核对，路线A/B编号步骤是下一实现交接，不是已经执行或已获批准的实验。
