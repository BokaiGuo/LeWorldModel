# Parent R8：RAM callback → NativeTree 最小生产 slice

本文件仅接线设计；不选择生产 bank、不更改已冻结 kernel/NativeTree/policy，不运行 NN/fullstrict/physics。当前 kernel 仍仅接受 RAM_FIXTURE，不能把 HOST_CUDA 伪标 RAM。下一生产版本应独占实现，而非 monkeypatch 或放宽旧版本检查。本设计不签 P1/full-policy/C3。

## 1. raw10 AD J 可以复用什么

可以直接复用 `agent_p3_local_signal_cuda_20261028_v1.NativeSignal.signal`（:362），通过现有 `agent_p3_parent_native_cuda_20261029_v1.NativeTree` 继承获得。它并非 logits/sketch J：

- 输入严格等于 `core.request(local_task,{context:local_task.root_context},operation='compile_finite_horizon_J',horizon=1)`；raw tensor 10维、零点、float32、requires_grad。
- 每5×2 raw 先 `(raw-task.action_mean)/task.action_std`，再 time-major reshape10，拼 outgoing3；完整 latent3/between2 经原 action_encoder/predict/predictor/pred_proj 滚动。
- 独立 nominal 与 AD forward 逐位相等；192个 one-hot VJP、192次 grad 和 raw hook，保存 J192×10 和 native IO。不是一次 vectorized Jacobian 假记192回调。
- 原 source 记录 origin（含 mean/std/raw）、packet、boundary、nominal、AD_terminal、direction、grad/raw_backward/J。沿用现有 source/model/precision before-after 闭包，不新增 encoder/projector。

因此无需新写 AD 方程，也无需把已有 signal 的 raw导数除/乘 std。normalizer 已在 AD graph 内；再归一化 J 会错误缩放。必须建立每个 postcontext 的 local_task，不能直接对 outer-parent/root backend 调 signal 后改 SHA。

## 2. chart、norm、tanh 的唯一合法顺序

新 RAM variant 已定义 suffix10独立 logits、零点 J、identity192、原 bounded_ridge damping .002/cap sqrt(5)*.75。零点 `d tanh(u)/du=1`，所以原 raw-zero J 可直接用于该 logits 零点线性化；这只是零点相等，不宣称非零 raw/logits J 一样。

`ridge(J,goal-nominal)` 返回 float64 params → NumPy float64 tanh → 一次 float32 cast → reshape(5,2) 原 raw → NativeTree 内按 task mean/std 归一化一次。不得先 normalize 再 tanh、重复标准化、把 L2 cap 当逐坐标 raw clipping，或把 old sketch Q 变成 identity 而不声明 variant。

edge bank 已经是原始 raw，只查长度5h、float32 bits与[-1,1]，不对 edge 再 tanh。suffix H1仅lookahead，不加进可执行 edge/raw prefix。保存 params、raw、normalized token与全 context；实际 bounded suffix nonlinear cost 与 ridge reach 分栏，不用 nonlinear cost 替换资格。

## 3. 最小生产 adapter 调用图

复用 `parent_native_adapter.parent_task(root,context)`（:30）建立局部视图；它更新 context binding但不是新 root certificate。每个 invocation 保存 outer root_task、outer parent、local_task 与 request 的完整 mapping，而非只有两个 hash。

1. 验证已冻结 bank authority、完整8 raw edge和独立zero，完成九条预算及最终验证 reserve 的原子检查；通过才调用模型。八队列/原144分母保持，zero不是第九个外部候选。
2. 顺序 zero、0..7：对 outer-parent local_task 建 `NativeTree(model,device,task,sink)`，`replay(parent,edge_raw,False,ledger)` 返回全部h个boundary；保存整条 path并checked_path。edge不会通过 signal 生成，也不会标 progress。
3. 用 edge终点建立 post_task；在该 task 的 NativeTree 上 `signal(exact_H1_request,ledger)`。复用旧 `RecordingBackend` 的非递归 attempt/return/failure 语义，给每个调用独立call_id。九个 post_task逐项保存，不按latent相近跨槽借J，也不借已付 outer-parent H1/H3。即使两条raw相同，本variant也逐槽付费。
4. signal response 只含 nominal192/J，RAM callback要求 nominal_path。**从本次 H1 nominal phase 已保存的 boundary 解码出完整context**（带incoming后两token），认证与 response.nominal逐位一致，再转新callback payload。不要为了补nominal_path额外replay一次，也不要只凭 terminal192捏造context。新strict须把该 boundary 的call_id/phase/step/数组pointer逐项绑定。
5. 内部调用原 bounded_ridge，计一次array_SVD/solve，保存全部输入/输出；再对同post_task执行 bounded suffix replay。严格资格与稳定index规则沿用kernel，完整8槽，不复制winner/不fallback。
6. 新own-schema orchestration将合格edge接回原modelQ beam，不能改旧adapter的progress-only edge白名单。最终从root做whole/segmented replay；不把诊断suffix拼进edge。生产saved consumer重放已存调用和数值选择，不执行神经，拒绝unknown event/漏调用/错context/错source。

实现建议：新生产 evaluator复用已冻结数值合同/输入校验/收费公式，显式接受真正 HOST_CUDA callback；不要直接调用RAM evaluate假标kind。kernel tariff与native ledger是同一次计算的两种视图，须逐调用对账，**不能二次收费相加**。native signal外层需要signal_compile包装，replay外层需要replay计费，compile/metric/solve/array_SVD同样明确包装；底层NativeTree/NativeSignal只负责其实际嵌套调用，不能假设所有外层已收费。

## 4. 九 postcontext 来源闭包与资源

每槽映射至少保留：bank source/authentication/bytes SHA、candidate或zero身份、outer root/parent完整上下文、原prefix/path、edge raw/path、post_task及binding、H1 request/response、nominal/AD boundary与J所有pointer、solver输入/输出、suffix raw/path、资格两项不等式和各operation的ledger区间。原模型实际installed class源码、model权重/精度guards前后仍由现有生产closure统一绑定，不把九次local_task当九个新模型/新根。

单bank最低费用（尚不含 outer parent决策已付H1/H3及最终验证）：

| 资源 | h=1 | h=3 |
| --- | ---: | ---: |
| edge predict | 9 | 27 |
| post H1 nominal+AD predict | 18 | 18 |
| bounded H1 suffix predict | 9 | 9 |
| predict API / sample rows | 36 / 36 | 54 / 54 |
| J / AD primal forward / J元素 | 9 / 9 / 17280 | 9 / 9 / 17280 |
| VJP / grad API / raw hook | 1728 / 1728 / 1728 | 1728 / 1728 / 1728 |
| array SVD / ridge solve | 9 / 9 | 9 / 9 |
| edge compile / metric / replay | 9 / 18 / 18 | 9 / 18 / 18 |

每predict还有各1 action_encoder、predictor、pred_proj调用，context/sample token轴为3倍；pred_proj_rows为3倍。signal_compile9、encoder/projector0、model_SVD0是该明确调用图的预期，不是已经取得的生产计量。AD_primal/J元素须由实际phase和J数组复算，旧ledger无这两个轴时用新显式扩展/派生字段，不能修改旧AXES或未知补0。FLOPs/backward engine/CUDA时间无测量仍unknown。

最终reserve至少两次root→selected完整可执行链的predict：`2*(old_prefix_tokens+h)`，replay_calls2及全部嵌套forward轴同步预留。若后续search还要扩展，另按原队列合同预留，不能把bank局部最低数说成整棵树上界。原114/562 hardcaps可保持但允许原子拒绝；不能宣称R完整树必定容纳。九条J是新增费用，parent已付pair仍单列，不退款、不padding。

## 5. MAIN需要决定的最小集合（不从outcome选择）

| 决策 | 原proposal能推出的默认/不能推出的部分 |
| --- | --- |
| **生产bank来源/参数化/八索引** | 原严格R用uniform含zero；full-packet有8个spectral proposal另加zero/root-DLS，共10。可讨论以原8 spectral作外bank、zero独立控制，但去掉root-DLS/扩到H3是新variant，不能自动认证。当前必须由MAIN冻结八条raw的生成规则、seed/chart/support、原始来源及认证；本doc不选bank |
| **edge与lookahead时域** | RAMvariant已固定edge5h、h∈{1,3}、诊断suffixH1；建议生产保持原choose返回h、不按remaining-fit缩短。它不是旧prefix1+suffix1的H2程序，亦不宣称同预测终点。MAIN需确认沿用这一variant，而非重新解释suffix为执行动作 |
| **八槽资格如何进入beam、无witness如何结束** | 严格原proposal推出 immediate>zero且reach<zero、无epsilon和None；RAM已经全保留并按(reach,index)给diagnostic winner。但原旧compiler返回一个prefix，原tree按modelQ选beam：MAIN须固定是全部eligible真实edge进入原modelQ beam，还是仅严格winner继续，其余显式未采用；无eligible应停该分支no_witness，绝不progress fallback |

identity192/原goal/normalizer、原damping/cap、suffix tanh顺序、原chi/gain阈值/宽8/depth/144都无需重新调参；改变它们是另一个协议，不是本最小slice。硬budget若不够应停；增加budget是单独资源设计决定，不能隐含在适配里。

## 6. 下一代码验收的具体最小测试

RAM映射测试首先证明edge→post_task→H1 request是九个真实身份链；native boundary能无新增predict转成nominal_path；原norm/tanh/raw bits一致；每一call tariff与native嵌套events一一对应；严格eligibility/无witness和8槽partial不丢。拒绝伪RAM生产、错phase boundary、旧root J、重复normalization、漏control收费、任一资源不足、source/class变化。最后由MAIN审读后才生产prepare/run/strict；不重跑旧合成CEM或progress作替代证据。

本轮只读源码、写本独占文档；未运行测试/模型/fullstrict/physics，未改任何冻结文件。源码锚点：NativeTree/RecordingBackend见 `agent_p3_parent_native_cuda_20261029_v1.py:114/144`；post-task见 `agent_p3_parent_native_adapter_20261029_v1.py:30`；原signal见 `agent_p3_local_signal_cuda_20261028_v1.py:362`；新variant数值/预算见 `agent_p3_parent_r_bank_evaluator_20261029_v1.py`。

本轮只读核对SHA256（均为 `experiments/spectral_reachability/` 下）：

- `agent_p3_parent_native_cuda_20261029_v1.py`：`eeea5710d4171b7c43f99d1f2b5ad2412e03a25ba14e8fc5589ea5ef73f57b3b`
- `agent_p3_local_signal_cuda_20261028_v1.py`：`6ce7722eca40cdd5f8958ba9590d8dc3d76d38bd10895d5ca494c1db3b6d100b`
- `agent_p3_parent_r_bank_evaluator_20261029_v1.py`：`71e4e178eb180a0d9b740697c98538d6f9afd0ba50d35efb03aa23f91378e04d`
