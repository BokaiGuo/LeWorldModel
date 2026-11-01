# P2 同 controller 三臂：配对后的下一最小实现与 scaler authority

日期2026-10-29。仅新增本文；路径相对 `/home/gbk/code/LeWorldModel/le-wm-main`。短源码均位于 `experiments/spectral_reachability/`。不改旧source/metadata，不加载weights/NPY，不运行consumer/fullstrict/神经/physics，不联网、fit、换Q或挑根。本轮仅读取已指定源码与小型保存JSON，复核字节SHA。

## 1. 已闭合什么，不再重复什么

MAIN已完成五组data-only目录导出/消费/join：encoder、projector、action_encoder、low_predictor↔predictor、low_pred_proj↔pred_proj。目录333项（含predictor的30个None），五组原值相等，无layout differences。相关固定来源在冻结 `agent_p2_hilewm_computation_parity_20261029_v1.py:29–37`。

本次实际计算report：

- `results/agent_p2_hilewm_computation_parity_20261029_v1_cuda_MAIN01/report.json`
- 实测953261bytes，SHA `12fe43cc4a796fffc18b98fcc867283f8534c22110d794729652ac112aedead8`；本轮已独立重读字节核SHA。
- MAIN报告19.8788s、786数组、errors=[]；本轮只核report保存字段，`diagnostics.all_values_exact=true`，没有重新打开全部数组或重建strict。**完整消费正在MAIN执行；本文不编造receipt路径或SHA。**

范围是原known root0/off、root1/off的observed0/5/10与goal、真实history3/a_history2、固定S1/H3动作及完整30维梯度；不是所有输入/批量/layout的全域证明。两条Hi路径的若干AE输出stride与P2不同，值相等已分别记录，不能改称layout全相同。原paid history另列，也相等。不再以“同192维但尚无参数/计算证据”阻塞这个固定slice；也不重跑旧single-z或本次parity冒新证据。

**仍未认证：**Hi原action scaler、训练数据与实际eval process配对、任意目标优化器、共同三臂controller交付、三臂完整原生实验。低层计算相同不推出高层prior的训练action分布与P2公共工程常数相同。known2不是独立fresh，P1/P2科学准入仍false。

## 2. 原 scaler：确定入口与未知资产必须分开

| 事实 | 当前可定位证据 | 可推出/不能推出 |
|---|---|---|
| 原stats生成代码 | `results/p1_hilewm_text_contract_20261027_v1/texts/package__code__h_le_wm__eval__hierarchical.py.source.txt:309–331`，`get_dataset/build_process_map` | HDF5Dataset取非pixels全列，删除任一NaN行后现场`StandardScaler().fit`；不是加载既有stats sidecar |
| 数据声明 | 同texts下 `package__code__config__eval__hi_pusht.yaml.source.txt:4,14–18,93–99` | `cache_dir:null`、action/proprio/state、`pusht_expert_train`；没有当时resolved cache路径、dataset revision或stats文件SHA |
| inverse真正消费 | `package__code__h_le_wm__planning__policies.py.source.txt:890–895` | action reshape成env action_space后，对action列执行`process['action'].inverse_transform`；不能对10维block随意广播未知stats |
| 可见dataset候选 | 已存mirror report的datasets repository | revision `655cd446b9929369d7d406001da85c15d1457850`，`pusht_expert_train.h5.zst`，声明13136247974bytes/LFS SHA `7cfbd6d90fa2f27876379a5ff169715a36ed82edbda64f9e5b5bfa34d212f318`；只是镜像声明，不认证原eval实际用过它 |
| 可见model候选 | 同report的models repository | revision `22b330c28c27ead4bfd1888615af1340e3fe9052`，config/weights/README；tree无独立scaler sidecar，config无stats。不能据此否定未列历史/私有附件或大文件内部 |
| P2工程常数 | `experiments/run_pusht_counterfactual.py:34–35`；`pusht_d2p_packet_contract.py::normalize_live_packet` | mean `[-.0087,.0068]`、std `[.2019,.2002]` float32，原P2工程协议可溯源；不是Hi当时sklearn process认证，不能借数值接近或tensor相等追认 |

上述官方文本本轮实核SHA：hierarchical `0a2c2b2688b73a388ff6011c4b32081704a7641b31efa1645a47b2f1d34eab3c`；policies `d6e7d476b45e47c5015494775b4d96727b8ebb8c801f719a100ba32878f288d1`；hi_pusht config `35b02a4f6e22dd9fecf034d664e799b2c5c85f157750ba53e0d69ba0be1091cc`。

两个定向调查的原report本轮重核：

- `results/agent_hilewm_normalizer_asset_route_20261029_v2_mirror01/report.json`：`f8eeae02c04ef66449243f687c199799c117fb438da9922451bdd1853d6d2256`。
- `results/agent_hilewm_normalizer_asset_route_20261029_v2_small_metadata01/report.json`：`1de9c82ac42613de2973d0d5cdb4d83b34239612dfab6b92aba85f889f423034`。

镜像tree实际仅dataset的`.gitattributes/README.md/pusht_expert_train.h5.zst`、model的`.gitattributes/README.md/config.json/weights.pt`。本轮没有重新探网、扫描cache/大目录或读取大训练包。**原冻结stats实际文件/member/SHA仍unknown，明确生成代码不是资产authority。**

最小可解除raw配对阻点的外部输入是当时已保存process的data-only导出：mean_/scale_/var_（原有者）、with_mean/with_std、n_features_in_、n_samples_seen_（实际标量/数组）、dtype及feature顺序/XY单位；并给实际文件SHA、checkpoint身份、eval config/数据版本与生成来源。字段未记录保unknown，不造数字。只有证明当时确用该对象才是历史认证；新fit不是找回旧对象。本授权不重新fit；若原对象未保存，MAIN可另提新协议，但不能改写历史或偷偷借P2常数。

## 3. 现在能接的真实函数，不复用旧固定plans授权

| 角色 | 可复用实现 / 实际接口 | 新接口必须补齐 |
|---|---|---|
| 共同causal transition | `agent_hilewm_normalized_transition_20261029_v1.py::cuda_kernels/rollout/segmented_rollout` + `agent_hilewm_differentiable_low_20261029_v2.py::wrap(graph,torch,main_authorized=True)` | 显式z `[B,S,3,192]`、between actions `[B,S,2,10]`、future `[B,S,H,10]`；不调用旧inference-onlynode.forward，不改旧root |
| P2原参考 | `pusht_full_packet_compiler_20261027.py::make_transition`；冻结parity的`p2_low` | 原函数含raw normalize；normalized controller不得再调用它做第二次normalize。参考/诊断独立保存，不拿parity的固定future充proposal |
| learned FIRST | `agent_hilewm_rgb_high_low_20261028_v1.py::high_search` 318–368行 | actual macro→selected rollout→`roll[:,0,0,:]`，float32 `[1,192]`逐位交付；保留high候选/成本/首target与完整rollout，不以末帧/goal替FIRST。旧`build/upstream`与四槽receipt不可换绑新六槽 |
| CEM/controller算术 | `agent_hilewm_low_cem_20261029_v1.py::OwnedLow.get_cost/LowBatchBridge/search` | 只能复用算术/见证设计。`low_shape`硬绑singlecontext；`search`硬绑H2、3iter×8samples、零history，不能直接当history3三臂controller。新版本接上面seam，不patch globals，不改CEM更新数学 |
| raw release | 同low_cem `inverse_transform` 228–235行 | 冻结版无条件拒绝；不能靠caller布尔或新stats字典绕过。任何真实inverse要新版本与authority消费，留原拒绝证据 |

`high_search`可在同一P2末帧/goal编码坐标上接线；history3完整交给低层，高层只见当前末帧/goal，明确是新组合校准，不冒称原Hi policy复现。high learned权重已有真实资产，不伪造prior；但原高层macro时长**不等于**物理packet5，不拿macro索引当cp5/10/15。

## 4. 下一最小实现：共同 normalized target-tracker，而非再造manifest

建议独占新source实现一个真实HOST CUDA controller内核，先与proposal解耦。下列是待实现接口，不是现成可运行CLI，不自授权生产：

```text
track(context_z3, context_a2, delivered_target192,
      frozen_low, registered_search, budget_meter)
  -> normalized_packets_Hx10, full_rollout_Hx192,
     selected_whole_and_segmented_replays, delivery_receipt,
     all_attempts_and_partial_state
```

所有臂进同一callable/precision/config，无compiled witness warmstart；初始mean、seed映射、可行域、H、候选数/迭代、topk、early-stop规则均由MAIN结果盲声明。目标按同一个预先声明identity192 mean-MSE计算；原Hi sum成本如需对照另列，不能混称原Q。target与最终返回mean的独立付费replay须封存，旧search的elite成本不能冒认为返回mean成本。物理raw输出关闭，不妨碍normalized实际搜索/追踪诊断。

可以先做固定一个delivered target的完整controller/H3 replay与actual gradient/成本计账，验证内核，再接三proposal；这不是三臂实验完成。不能为了立即出数以goal/random vector充arbitrary。arbitrary direct-latent目标函数、初始化、约束和迭代仍须事前固定，避免仅min目标距离导致全部点等goal；compiled须新生成对应动作响应target，保全fullH见证，**只向controller交target**；learned保全FIRST。三者proposal预算与controller预算分别固定并总计。

完整三臂还需MAIN选择时间语义：共同每packet5 actualRGB重规划FIRST，还是一次交三target。优先前者能直接复用FIRST；但必须新合同明确其与原一次三target协议不同，不能把高层宏时间当物理时间。固定known2×3arms、共15public/每5反馈的完整6分支/18cp不删减；没有认证raw资产前只交模型内结果，不能报告物理可执行率或成功率。旧物理结果不换绑新controller。

预算不能沿旧1902/450s或parity资源默认追认：MAIN预声明匹配low predict API或forward samples的哪一轴，包含proposal、controller、ADprimal、whole/segmented/selected replay；失败attempt不退款，不padding、不隐去partial。high predictor/macro/proj、encoder/AE、J/VJP/backward/SVD、CPUcapture/sync及wall独立列，FLOPs=null。完整真实输出允许未耗尽共同cap；不是actual all-axis equal。

因此最短可执行代码工作是**history3 normalized共同controller及其严格saved消费**，不是重做参数目录/跨runtimeparity或写另一个blocked清单；三proposal策略声明与原raw process authority分别推进，互不伪造。现无已认证完整三臂HOST命令，本文不把旧实验CLI包装成新实验。

## 5. 本轮验证与交付边界

已实际：读取上述官方文本/固定源码；重核3份report SHA、3份官方文本SHA，检查mirror原file清单；未重建任何producer指标/梯度，也未运行fullstrict。只新增文档，无新增程序测试；旧35项RAM通过由MAIN复核，本轮不重跑当新证据。

可直接执行的复核（只读，非实验）：

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
sha256sum results/agent_p2_hilewm_computation_parity_20261029_v1_cuda_MAIN01/report.json results/agent_hilewm_normalizer_asset_route_20261029_v2_mirror01/report.json results/agent_hilewm_normalizer_asset_route_20261029_v2_small_metadata01/report.json
sed -n '309,331p' results/p1_hilewm_text_contract_20261027_v1/texts/package__code__h_le_wm__eval__hierarchical.py.source.txt
sed -n '890,895p' results/p1_hilewm_text_contract_20261027_v1/texts/package__code__h_le_wm__planning__policies.py.source.txt
```

待MAIN实际strict回执到位才把本次计算配对标为完整消费通过；本文不改冻结metadata，也不追加未来hash。原scaler仍unknown，三臂尚未运行，known-root/科学准入界限不变。
