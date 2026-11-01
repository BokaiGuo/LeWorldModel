# P2 / Hi frozen pairing inventory

2026-10-29；只新增本文。路径除绝对路径外相对 `/home/gbk/code/LeWorldModel/le-wm-main`；短源码名位于 `experiments/spectral_reachability/`。沿用 `agent_p2_prior_pairing_route_20261029_v1.md` 的路线，不另造实验或认证。此次仅有界读取既定源码、小config、export JSON及loader receipt字段；没有读取NPY/大型weights、执行full strict/NN/physics或重新fit。

## 1. 当前结论

五组均**尚未证明 Hi embedded-low 与 P2 参数/buffer逐项相同**，也没有证据据此断言不同。同192维只说明shape相容；不是同坐标、同函数或可直接交付prior的证书。现有Hi config声明载入并冻结LeWM低层，这是可追踪线索，不是权重身份。

MAIN告知 probe v3 实际报告 SHA `ff4d92b73d5f11bb40953acb0b6e90c60bd97e6c991894066b51ab716e1e60c1`，三个route forward/grad逐位及固定FD通过；本文未运行消费。saved消费因construction的loader tests/doc没有纳入manifest.sources而由MAIN另处理，不能在此写成全strict已经通过。该probe仍仅是Hi自身 single-z/zero-history/S1/H3低层接缝检查，不等于Hi↔P2配对。两份未纳入before/after manifest的nonexecuted metadata限制也不能被当前文件hash追溯抹去。

## 2. 已有证据的精确入口

|证据|已有路径 / 函数|能证明 / 不能证明|
|---|---|---|
|Hi原低层来源意图|`results/p1_hilewm_text_contract_20261027_v1/texts/package__checkpoints__pusht__main__config.yaml.source.txt:55–73`|`pretrained_low_level.enabled=true`，explicit `data/stablewm/pusht/lewm_object.ckpt`，epoch1，五组freeze；不能证明这个训练期路径等于本机P2文件|
|Hi安全tensor目录|`results/agent_hilewm_storage_npy_export_20261028_v1_MAIN01/report.json`，SHA `3cd7dd6783129893f397ddef63bd253e447fbef93582dd5f4dc8d7144310efd3`|`state_dict_paths`、`rows[index].source_entry`、`rows[index].array`已给key/persistence/shape/stride/dtype/value SHA；无需读NPY即可先做目录join，不代表原foreign runtime state_dict parity|
|Hi构造/配对接口|`agent_hilewm_trusted_loader_20261028_v1.py::graph_spec/check_export/construct_cuda`|全425非None值和alias/persistence已有独立来源；必须保留原export实际验收，不把本文当新许可|
|P2当前config|`/home/gbk/.stable_worldmodel/checkpoints/pusht/lewm/config.json`，SHA `2564086e961e7b5c7c04dffc451091115b389a590645ff19653c64fd0bc16e09`|与已存候选镜像 `results/agent_hilewm_normalizer_asset_route_20261029_v2_small_metadata01/metadata_02.bin` 字节SHA相同；不证明Hi实际embedded weights相同|
|P2真实loader收据|`results/agent_p2_three_program_20261028_v1_cuda_MAIN01/loader.json`，SHA `2401302225f042755bd9a8e8a06874ebafd367dee5937434b442c739b45b4660`|本轮核其1263bytes字段：files/same-buffer、remap、missing/unexpected、aggregate `tensor_hash_before`；没有逐key value hash目录，不能用一个aggregate跨不同key前缀graph直接证明五组配对|
|P2加载实现|`agent_p2_equal144_baseline_cuda_20261028_v1.py::loaded_model/validate_loader`|同一immutable文件buffer验SHA后加载、严格检查missing/unexpected、eval/frozen；不是Hi embedding复制认证|

## 3. 五组逐项核验表

计数由上述export JSON按 `source_entry.module_path[0]` 且 `array != null` 有界复算；不执行模型。共303非None项=297参数+6buffer；其余高层不属于这五组，但原全425冻结守卫不能删除。None条目、nonpersistent标记与alias应在真正join保留，不用0张量代替。

|Hi → P2 literal前缀|Hi已存非None项|关键state/config与计算配对要求|当前精确缺口|
|---|---:|---|---|
|encoder → encoder|198参数|CLS `[1,1,192]`、position `[1,257,192]`、ViT patch14/image224/tiny；全部Q/K/V、LN/MLP参数及原encoder配置、eps/激活/SDPA、CLS选取|缺P2逐key/dtype/shape/value hash及配置逐字段join；shape相同仍不认证pixel输出|
|projector → projector|6参数+3buffer|`net.0.weight=[2048,192]`，BN running_mean/var `[2048]`、num_batches_tracked int64 scalar；MLP192→2048→192，eval BN/激活/eps|缺P2六参数和三buffer逐项值；不得只配Linear或忽略BN统计|
|action_encoder → action_encoder|6参数|patch_embed weight `[10,10,1]`、bias、embed两Linear；source Embedder float→permute→conv1d→permute→embed，原token time-major width10|缺P2逐key/原value；另缺action scaler身份。literal Conv1d结构解释要采用v2，不调用旧inference节点|
|low_predictor → predictor|81参数|position `[1,3,192]`；P2 config depth6/heads16/dim_head64/mlp2048、输入/hidden/output192、num_frames3、dropout .1而eval关闭、emb_dropout0；AdaLN/causal attention、ctx和layout|缺完整P2参数/constructor与Hi literal逐字段join；不能因两者都是AR就忽略ConditionalBlock/投影/注意力实现差异|
|low_pred_proj → pred_proj|6参数+3buffer|同样MLP192→2048→192及BN三buffer；在predict后按 `[B*T,192]` flatten再投影/reshape|缺P2逐项值和flatten/eval计算配对；单独latent终值相近不能代替|

合法key变换仅先做五组前缀映射，再复用 `experiments/checkpoint_compat.py::remap_legacy_vit_keys` 的七个已明示字符串替换（legacy encoder.layer→layers、Q/K/V、attention output、intermediate/output dense）。它是names-only、碰撞拒绝；不做QKV拆合、transpose、dtype转换、数值拟合或凭suffix模糊匹配。若实物需要额外规则，先列出具体key/shape与源码证据，再由MAIN决定；本文不预先授权。

## 4. 参数相等之外仍须闭合的合同

1. **配置与source实现**：读取确切P2config并与Hi `graph_spec` literal逐字段对照。安装P2 `stable_worldmodel/wm/lewm/module.py::Embedder/MLP/Predictor` 可直接读；Hi原 `module.py`/官方jepa文本及owned facade source也已有。state值相等不自动证明不同Python方程、eval/dropout/精度或layout相等。不能使用未核默认值填配置空白。
2. **像素/latent坐标**：Hi `agent_hilewm_rgb_high_low_20261028_v1.py::config_contract/preprocess/owned_pixel` 与P2 `experiments/run_pusht_counterfactual.py::image_transform` 都有f32缩放、ImageNet Normalize与224输入线索。仍须同一byte RGB上的实际逐层CLS→projector比对；禁止沿用P2 latent冒充Hi输出。
3. **causal context/计算时序**：Hi原policy single-z+零incoming动作；P2 `pusht_full_packet_compiler_20261027.py::make_transition` 是 `[1,3,192]+[1,2,10]`，raw `[5,2]`规范化一次后追加。新normalized seam虽可显式接history，已报告probe只证single-z，尚无P2 history3同输入接线证据。必须用真实causal输入认证ctx=min3、outgoing先append、AE全history、flatten projection和末帧追加，不重启/裁剪接缝。
4. **raw scaler独立前置**：Hi官方 `.../texts/package__code__h_le_wm__eval__hierarchical.py.source.txt:309–331::build_process_map` 对完整数据去NaN后现场 `StandardScaler.fit`；policy `get_action:890–895` 用 `process['action'].inverse_transform`。已保存镜像调查没有定位到原stats sidecar的确切文件/member/SHA；仍unknown，不是证明全球不存在。需要当时保存的data-only mean_/scale_、flags/dtype、feature顺序/raw单位、dataset revision及checkpoint/eval配对，不能本轮refit。
5. **P2工程常数不补Hi缺项**：`run_pusht_counterfactual.py:34–35` mean `[-.0087,.0068]` / std `[.2019,.2002]` float32；`p1_constructive_moving_history_encode_cuda_20261027_v1.py::normalization` 明确training provenance未证实。BatchNorm统计不是action scaler。旧 `agent_hilewm_low_cem_20261029_v1.py::inverse_transform` 拒绝所有raw发布，不能用公共常数追认。Hi cost sum与P2 identity-Q mean也不是相同标量合同。

## 5. 下一项最小真实核验与顺序

**先补P2逐key目录，不再启动新神经forward。** 当前明确不足的是已定位loader收据只保存aggregate，而非不存在checkpoint。MAIN可在独占受审的data-only导出步骤，使用原冻结P2加载边界和实际checkpoint SHA，输出完整参数/buffer的original key、remapped key、dtype/shape/stride/offset、persistence/alias、C-index原值bytes SHA及config/loader/runtime/source绑定；不需要instantiate forward、训练或refit。若有既存完整目录，应先复用；本轮只检查已明确P2 loader.json，未搜索全库，不能声称全仓库都无目录。

随后在RAM/saved-only做固定五前缀一对一join，对Hi现成metadata逐项检查key集合、dtype/shape、参数/buffer角色、None/persistence/alias和原value SHA；统计missing/unexpected/mismatch完整保留。格式/端序不同不能直接hash对比时先报告具体障碍，不能默默coerce。任何数值不同不能用mapper、refit或容差宣布同backbone。若全部通过，仅签“五组已存tensor/config目录配对”而不是完整baseline可运行。

最后才由MAIN另审固定CUDA source-native parity：同RGB encoder/projector、同normalized history3动作AE/predict/proj，原before/after权重守卫、全部中间数组与layout保存；原single-z probe不重跑充新证据。原raw stats仍独立阻断native三臂发布。路线A统一Hi可继续normalized模型内实验，不等待跨backbone等值，但没有原stats仍不得宣称物理可执行率。

这些是两known roots的工程核验，不是新fresh panel；原P1/P2科学gate不变。API/sample、AE/encoder/projection、J/VJP/backward/AD、replay/capture与wall各自计量；权重配对或同192维不导出equal-FLOPs。

## 6. 现在可执行的只读复核

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
sha256sum results/agent_hilewm_storage_npy_export_20261028_v1_MAIN01/report.json results/agent_p2_three_program_20261028_v1_cuda_MAIN01/loader.json /home/gbk/.stable_worldmodel/checkpoints/pusht/lewm/config.json
jq '{schema,device,remapped_keys,missing,unexpected,frozen_eval,tensor_hash_before}' results/agent_p2_three_program_20261028_v1_cuda_MAIN01/loader.json
jq '[.rows[] | select(.array != null) | .source_entry.module_path[0]] | group_by(.) | map({group:.[0],non_null_entries:length})' results/agent_hilewm_storage_npy_export_20261028_v1_MAIN01/report.json
```

本轮已实际运行对应JSON字段/五组census及上述三个SHA检查；未调用consumer，无新增测试代码，文档检查不是配对科学证据。未完成：P2逐key权威目录、五组逐值/config join、跨实现RGB/history3 parity、原scaler文件/provenance及MAIN新saved probe消费结论；没有伪造通过或未来SHA。
