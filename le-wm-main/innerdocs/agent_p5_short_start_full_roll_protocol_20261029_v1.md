# P5 prospective short-start full causal roll — protocol v1

仅设计文档；未实现或授权656生产运行。固定已有164枝/80对、两个known roots、off/on全保留，不按既有或未来outcome挑根/尺度/坐标。使用新短启动协议，非补认证旧实验，非科学门修复。

## 1. 已有依据与尚缺授权

项目根目录：`/home/gbk/code/LeWorldModel/le-wm-main`，下列路径相对此目录。

冻结短启动probe三件套保持不变，source/test/doc SHA分别为：

```
56e4e9919b0e5b21fbeca11230758c999854739482b91214f688cb91066fcfe6
8746b62d714332fbc0a49740ed4e3eea0f7f0b0dba56e96a76118eb9ae4424d8
3172e905507fea972d7edbb585a0eea5c63a1da0eeb18f3913401114afdbaf8c
```

MAIN报告`agent_p5_short_start_cuda_probe_20261029_v1`主机T1/T2已完成，report SHA `2f1f60a2f34bf329dcfcb7e5ea378dd3dbf22283be15aa640301ea1ed62974d8`，status `branch0_T1_T2_native_ABI_recorded`。这是MAIN通知，本设计没有执行生产consume；其strict尚在进行，确切report路径、prepare SHA及strict路径/SHA须由MAIN交接并在未来prepare显式绑定，禁止猜路径/未来SHA。原probe是branch0工程证据，不证明其余163枝或完整四次链已通过。

来源catalog：`results/agent_p5_short_start_native_abi_20261029_v1_catalog_MAIN01.json`，SHA `1c49b5eba91619a6798a9d387c524b6e7d8658c196a9ac75f1ea9a65226c2678`。逐枝复用冻结ABI resolver的六NPY交叉核验，不按文件名猜身份；branch0保留三rawbytes SHA认证。

原H2：`results/agent_p5_raw164_jepa_h2_cuda_20261029_v2_cuda_MAIN01/report.json` SHA `e5a561215bfe7b6f1c1c224e7661211e3f237fb16458a6a409f5a1da0fc08579`；其prepare/strict分别为`results/agent_p5_raw164_jepa_h2_cuda_20261029_v2_prepare_MAIN01.json` / `results/agent_p5_raw164_jepa_h2_cuda_20261029_v2_consume_MAIN01.json`，SHA分别为`09070780c3ce782a71ae01492e0c4b801de976e9a36cdf9ee1bf1a53be98e924` / `783b65f67899791b5697bfe0819ccfb3231d57212b9e3e85aefc02419459c0ea`。

未来实现/测试/MAIN审读/prepare封存仍必需。本doc不是执行manifest，现无run CLI或新strict成功声明。

## 2. 时间轴、完整上下文与动作见证

每枝固定四次batch1调用，branch-major顺序0..163。每次完整保存三个实际输入及完整三元组输出；只用末token的visual和proprio追加历史，不将先前历史槽用本次输出整体覆盖。

| stage | T | predictor输入帧与来源 | action原packet索引 | literal raw steps | 新末token目标 |
| --- | --- | --- | --- | --- | --- |
| 0 | 1 | actual0 | [0] | [1..5] | pred5 |
| 1 | 2 | actual0, NEW pred5 | [0,1] | [1..5,6..10] | pred10 |
| 2 | 3 | actual0, NEW pred5, NEW pred10 | [0,1,2] | [1..5,6..10,11..15] | pred15 |
| 3 | 3 | NEW pred5, NEW pred10, NEW pred15 | [1,2,3] | [6..10,11..15,16..20] | pred20 |

raw1..10为原warmup，raw11..20为该枝原own raw10。取原H2 manifest完整20动作、原normalizer和pixel32 roundtrip，逐字节交叉核验原saved三个native action packets，第四packet由同一原20动作与同一统计量生成。不得改单位、缩放、裁剪、补零或用另一枝动作；每节点保存literal动作范围/packet索引/normalizer源与字节SHA。

对T∈{1,2,3}，输入shape固定为visual `(1,T,1,16,16,384)`、action `(1,T,10)`、proprio `(1,T,256,16)`；输出为 `(visual (1,T,256,384), None, proprio (1,T,256,16))`。全为finite float32、同一CUDA设备、frozen eval、inference。复用`agent_p5_roll_history_preconditions_20261029_v1.advance_history`无算术拼接ABI；它是纯数组守卫而非模型有效性认证。动作滑窗与特征帧同步，stage3不再含actual0。

NEW pred5/10必须来自本次stage0/1。不得复用旧H2 full3输出前两token：其pred10看到actual5，不能替代真正causal pred5→pred10。也不得把probe branch0输出当成本次零成本预测；本次所有164枝四次全付费，无额外best replay/重试/padding。

原输入帧0/5/10属于原warmup、在constructed-goal bind之前；plan5/10即全局15/20属于原goal bind之后。保留原off/on renderer与原goaloverlay历史时序，不重绘、不假称overlay跨bind恒定。本协议不向模型额外注入goal或未来场景。故15/20同时含原部署时序接缝，不能把其差异唯一归因于rollout累积。

## 3. 预算、计量与资源（事前固定）

每枝predict4 API、4batch samples、9token-frame samples；全164为**656 API / 656 samples / 1476 token-frame samples**。各stage全批分别为164/164/164、164/164/328、164/164/492、164/164/492。无新encoder、proprio_encoder顶层调用、AD、训练、物理或下载；predictor内部真实子模块成本不冒充零，也不与顶层API相加重复计账。

每token输入409,640B、输出409,600B，合819,240B；每枝完整native IO 7,373,160B，全批**1,209,198,240B**。这仅是三个输入+两输出的完整float32 tensor有效字节，不是GPU激活或总算力上界。另列每module实际intent/observed return/validated return、实际token/样本/IO字节、cuda peak allocated/reserved与host RSS，失败也记账。

未来runner固定：输出cap3GiB；tensor快照总cap2GiB；单NPY cap4MiB；单metadata cap32MiB；终态预留64MiB；磁盘floor200MiB且起跑要求cap+floor可用；初始CUDA free floor4GiB；完整job wall900s从verify前起，含load/dispatch/最终模型及source hash（终态durable cleanup单列真实超时，超时不认证complete）。不得因资源不足缩分母/downcast；prepare必须重算实际manifest字节并拒绝越界。

保守tensor存储估算：native IO约1.209GB，额外actual0三输入+四action packet的独立输入银行小于68MB，若完整另存旧label来源3帧+2帧patch特征至多322,437,120B；总计小于1.60GB，低于2GiB tensor cap。NPY头、164行/656calls、journal与source bindings仍按实际bytes计入3GiB输出cap；不复制模型权重进输出目录。保存不重复的label文件引用也必须保留原全文件SHA/EOF而非仅语义digest。

继承成本独立保留原H2 report实际模块收据及来源，而不是把复用features称为免费采集或本次新encoder。原完整H2的上限/完整返回轴为encoder328 API/820 image samples、proprio_encoder164 API/492 token frames、predictor328 API/984 token frames；新任务所复用是其saved tensors，不能按引用次数再计一遍全H2，也不得将不同算法成本称等FLOPs。

## 4. prepare / run / seal / labels / consume ABI

1. **prepare（无NN）**：固定164索引、原两root×off/on各41枝顺序、20原动作和80对schedule；绑定原physical→bridge→H2→catalog→probe strict完整源/模型文件closure。逐枝调用原actual0 resolver/materializer，抽出独立actual0三输入与四packet动作银行，保存其NPY SHA/rawbytes SHA及原report JSON pointers。这里只认证来源，可能读取含actual5/10的原完整NPY以交叉核验；不得宣称文件级从未读过这些字节。
2. **输入隔离**：未来run predictor adapter只接本次独立actual0输入银行和动作银行，不再解码/抽取旧actual5/10或successor RGB/state/proprio/features。label侧refs/评分代码与predictor ABI分开；loader的原authority metadata/hash核验不能被说成神经反馈。全future tensors即使离线已存在也不供预测。
3. **run**：复用冻结原bridge `load_host_cuda`、原model_receipt与本地checkpoint；加载前后记录原权重/config实际文件hash、strict loads、state receipt、所有实际import源SHA。新增只属于本runner的T=[1,2,3,3]守卫，不能修改旧probe“两次”ShapeMeter或旧full3guard，不monkeypatch。全部三模型装hooks，拒绝任何额外encoder/重试；每stage intent→完整inputs→observed return→完整outputs→stage seal后才能构建下一stage。
4. **全批seal**：须164枝×四stage全部完整且656次计量一致，保存所有full outputs、末token refs和整批seal/hashchain，之后才开放labels解码/读取与评分。失败保留164行/80对/656槽，不为幸存枝先解label并声称完整协议；缺项null，partial工程失败不是理论负结果。
5. **common label bank（零新编码）**：全seal后从原H2每枝`rows[i].context_patch`的原batch3 encoder输出抽取actual5/10（slots1/2）；从`rows[i].label_patch`的原batch2输出抽取actual15/20（slots0/1）。严格验证各ref与原native call、RGB来源、global frame、root/condition/task/selection及原inventory相同，不能仅按shape索引。15/20所有协议比较一律用这一份H2 label bank，不能混用H1 label编码结果。actual0可作固定persistence，不作为新标签反馈。
6. **saved consume**：不导入torch/physics，重建本prepare、全部manifest/SHA、四stage输入完整字节和动作滑窗，验证每一NEW pred5/10/15的因果来源，原输入和输出实际设备/shape/dtype/frozen元数据，所有事件字段/顺序、全batch seal先于label_read和score事件；独立复算算术、字节与完整分母。加载hash证据不得降格为仅metadata digest。只有完整工程记录才给新版本strict，不伪装成旧receipt。

HOSTCUDA-only、TF32off、请求MATH、沿probe保留observed fused-op拒绝及actual attention op清单；`requested_MATH`不是实际kernel证书。禁止CPU fallback，禁止为绕native内部backend选择更改模型。异常、超时、源漂移、坏返回/shape或禁止op全部封存，不重跑挑成功。

## 5. 固定80对与描述性比较

完整schedule沿冻结`pair_schedule()`：slot0..3依次(root0,off),(root0,on),(root1,off),(root1,on)；每slot nominal索引41s，h按(.01,.001)，j=0/1，coord k=0..9；minus=`41s+1+20j+2k`、plus=`41s+2+20j+2k`。每层20对，全80对，不按原56非零、符号或新结果筛选。164不是独立样本，root仅两个。

固定输出：每枝四时点(5,10,15,20)common-label全256patch×384feature float64 C-order MSE，及actual0 persistence MSE；80对每时点完整Δpred、Δobs（plus-minus）、两范数、dot、cosine，零范数cosine=null，正/负/精确零/null全保留，无epsilon阈值/显著性/新通过线。两h的central differences除2h另列，不能将raw差与导数混用。off/on、root、h、coord全部原身份分栏。

11..20才有原动作扰动；5/10是共同warmup阶段，属于固定重复输入/早期一致性检查，不冒充有非零物理响应的动作鉴别。逐枝列actual0/action0..1是否全byte同、pred5/10是否全byte同；若相同输入输出出现差异保留数值复现问题，不能按h调规则。

对原H2、原H1只在15/20各自存在且identity/模型/normalizer/动作/label确切相同的端点作描述性分栏；用共同H2 label数组重新计算其saved末token MSE与pair arithmetic，不借原异batch标签标量直接比较。新协议改变实际历史反馈/启动长度，不是单纯horizon差；wholehistory/latest已证实相同输入，不能当两项独立contrast。未来若要构造新history factorial，必须另行事前固定ABI并先验证early输入数值不同，不能在本轮按效果派生新角。

label5/10来自batch3而15/20来自batch2，既有label批次接缝是明确限制；本次零新编码不能证明跨时点绝对MSE差是纯累积误差。全部比较是模型内部factual fidelity，非混合context物理可达性、控制收益、任务native成功或C2/P1准入；不跨架构直接排名MSE，不拟合Q/权重/门槛。

## 6. 下一可执行步骤与交付边界

MAIN先提供probe actual strict路径/SHA与实际prepare/report路径；随后若授权实现，新独占runner须以本协议重建164输入银行及四阶段守卫，并用fixture验证T3两调用、全656seal门、batch label接缝、真实Store/hashchain、失败分母和cap。现在只交本doc：没有新runner、prepare、656输出或测试通过声明；不发不存在的生产CLI。旧probe与所有H1/H2封存文件不变。
