# actual0原生IO抽取与T1/T2最小ABI检查准备

只新增本stem source/tests/doc，不改旧H1/H2/full3 guard。无模型load/forward、physics、生产catalog/materialize、full strict或子智能体；不是656实验runner，不新增分数/训练/科学修复。这里只把上一份preconditions留下的actual0指针补成saved抽取接口和纯descriptor shape/meter fixtures。已有preconditions实际inspect SHA `1bd22a07c7a2966446bdb7d0e112063fa81361d1709092cb1a841b5bd6a0775f`为MAIN通知，本工作位未重跑；不把当时进行中的verify写成完成。

## 三个固定上游authority

以下路径均以 `/home/gbk/code/LeWorldModel/le-wm-main/` 为根：

| authority | 相对路径 | SHA256 |
|---|---|---|
| 实际H2 report | `results/agent_p5_raw164_jepa_h2_cuda_20261029_v2_cuda_MAIN01/report.json` | `e5a561215bfe7b6f1c1c224e7661211e3f237fb16458a6a409f5a1da0fc08579` |
| 实际H2 prepare | `results/agent_p5_raw164_jepa_h2_cuda_20261029_v2_prepare_MAIN01.json` | `09070780c3ce782a71ae01492e0c4b801de976e9a36cdf9ee1bf1a53be98e924` |
| 既有H2 strict receipt | `results/agent_p5_raw164_jepa_h2_cuda_20261029_v2_consume_MAIN01.json` | `783b65f67899791b5697bfe0819ccfb3231d57212b9e3e85aefc02419459c0ea` |

`catalog(...)`固定以上外部SHA，核report→prepare确切路径/SHA和strict→report字段。它解析全164 pointer，**不重建full strict**、不把status字符串当新物理/模型认证。既有strict的有效性是显式继承信任；本catalog只增加指针/身份核查。prepared/request、row的index/root/condition/task/selection必须相同；不按文件名推断，也不选结果。actual0是原pre-goal-bind warmup0，不是plan0/global10，也不是原H2预测15/20。全panel仍固定164/80；下述单branch工程probe不是缩分母的科学实验。

## 精确receipt接口与branch0实例

branch `i`首个native predictor为report `/calls/{4*i+2}`。其`module=predictor,phase=prediction,status=returned,units={api:1,batch_samples:1,frame_samples:3}`，输入树是`((visual,actions,proprio),{})`。

三个原完整输入的JSON pointer为：

```
/calls/{4*i+2}/inputs/$tuple/0/$tuple/0/$array
/calls/{4*i+2}/inputs/$tuple/0/$tuple/1/$array
/calls/{4*i+2}/inputs/$tuple/0/$tuple/2/$array
```

父shape依次`(1,3,1,16,16,384)`、`(1,3,10)`、`(1,3,256,16)`，时间为0/5/10。actual0仅为三者`[:,0:1].copy(order='C')`，shape依次`(1,1,1,16,16,384)`、`(1,1,10)`、`(1,1,256,16)`；原raw1..5动作作为outgoing packet，不补全/复制actual5或actual10。每份抽取receipt记录原数组SHA、准确report pointer、slice以及抽取后C-order byte SHA；后者worker未生产计算，不预填。

branch0原实际文件都位于report同目录：

| 角色 | 原文件 | 原NPY文件SHA256 |
|---|---|---|
| predictor视觉输入 | `tensor_00007.npy` | `5d4f00bec6c6e15c87f9bba64450353b011b5bfefa035b80cdcc4f0b1ee73829` |
| predictor动作输入 | `tensor_00008.npy` | `e1d00b43533afba1b4efbb5219aa17025a9790019ee874199b35aae5acd4c14e` |
| predictor本体feature输入 | `tensor_00009.npy` | `6a33f981f54c25f7a095433fb0757b2e925df6412646a5236e1fdcc3f4679703` |
| encoder原patch输出 | `tensor_00002.npy` | `0ecd8f5efd155b728fec7e2e44e4774283d8fe3d3cda575c11c6f28cce4501f8` |
| proprio encoder原归一化输入 | `tensor_00005.npy` | `65bbeb1091f9eee90f099b25d4184d30c9464358c2b19c275800d31887af720f` |
| proprio encoder原feature输出 | `tensor_00006.npy` | `ee3de30a0c82d6581c18e9378e39905ca4eb98210127afe7402ba577a74c1ef5` |

前三个文件file_bytes依次1179776、248、49280；这些SHA是完整原NPY文件，不是抽取后tensor byte SHA，不能混域。其余branch按report pointer解析，不沿文件编号猜测身份。

交叉核查的三个额外指针：

```
/calls/{4*i}/outputs/$dict/x_norm_patchtokens/$array
/calls/{4*i+1}/inputs/$tuple/0/$tuple/0/$array
/calls/{4*i+1}/outputs/$array
```

`resolve_actual0(...)`只解析上述六个refs并核每个rec与report inventory的SHA/file_bytes一致；`materialize_actual0(record)`才读取六份bounded NPY，核SHA/size/dtype/shape/finite/EOF。它证明整份predictor visual等于原encoder patch reshape，整份predictor proprio等于原proprio feature广播，原proprio encoder输入等于manifest的原stats归一化值，完整动作等于原raw1..20归一化packets前三组；然后才切actual0。不是重新forward encoder，也不将raw位置速度当embedding。

branch0原RGB0在prepare `/bridge/requests/0/images/0`：
`results/agent_observed_raw_coordinate_physical_20261028_v1_MAIN01/b0_warmup_00_0.png`，file SHA `b2accca6d8db2ba86aec1b0d41d75a626b9fa1bde30987eefb7829ff3b6b2cbb`，pixel SHA `ea1b47a1b3cee6f32bff6c4b560a37dfb3fe3edbc52bf6d9d16eb1bfefa0372d`。本工具不重新decode该PNG/重建预处理；RGB→原encoder输入的认证继承既有strict，不能称本工具新认证。原生agent位置/速度的额外权限保留，不称RGB-only或公平跨模型输入。

## 最小HOSTCUDA入口设计：仅两个工程dispatch，未实现/未执行生产runner

固定branch0/root0/off作为不评分的工程ABI probe，不挑结果；不是全部164的效果实验。必须先完成above saved抽取并封source SHA、actual0输入和原动作1..10。顺序固定：

1. 原actual0 T=1→NEW pred5，保存完整原生visual/None/proprio返回；不使用旧full3早期输出作替代。
2. actual0 + 刚封存NEW pred5的visual/proprio，T=2→NEW pred10；动作严格来自原raw1..5、6..10。不得actual5反馈、不得padding，不读取实际successor label或计算MSE。

形状以实际T派生：input visual`(1,T,1,16,16,384)`、action`(1,T,10)`、proprio`(1,T,256,16)`；output visual`(1,T,256,384)`、None、proprio`(1,T,256,16)`。不更改原full3 guard为宽松形状；将来须新local hooks只允许T1后T2的固定顺序。

可复用原桥`agent_p5_raw164_jepa_input_bridge_20261029_v1.load_host_cuda(original_bridge_prepared,original_bridge_prepare_sha256,'MAIN_P5_RAW164_JEPA_HOSTCUDA_ONLY')`，源SHA `d70352f2fc574ca8a226522fc3dda9f015305894c83a1d1826f05808c005da0f`。它原样调用冻结v8 loader与strict schema，并返回models、model_before、strict、sources、MATH factory；不复用其旧full3 forward/Trace，不monkeypatch globals。

MAIN未来实现入口必须：加载前绑定原bridge manifest/source/cache；实际加载后model_before与原H2三模块receipts完整相同、strict_loads一致、所有实际imported sources SHA正确；模型eval/frozen/CUDA、inference、TF32off/MATH。两次后和finally再次核模型/source，并保存加载/显存/失败。encoder模型可被loader构造但禁止encoder forward。上述是待实现加载/运行守卫，不是本source已经加载/认证模型。

`ShapeMeter`仅为未来hooks接口的纯descriptor模型：真实adapter必须从实际Tensor读取device/dtype/shape/requires_grad，不能信调用者预填声明。T1/T2 attempt在dispatch前计，完整合法return才累计returned，跨设备、encoder/nested/额外replay/重试、错误输出、CPU/AD均拒绝；失败保留pending和attempt不补跑。descriptor fixtures通过不等于真实HOSTCUDA ABI通过。

事前工程上界：predictor2 API、2batch samples、3token-frame samples；encoder/proprio encoder/AD/physics均0新调用。有效完整输入+输出tensor共2457720B，tensor cap4MiB、输出cap16MiB、floor200MiB、wall300s覆盖verify/load/dispatch/final hash；初始CUDA free4GiB仅scratch准入，不当作activation保证。durable inputs/intent、outputs/return、两槽完整seal、model/source前后收据均须由未来新入口实现；本工具没有该生产loop或假receipt。

仍未认证：实际materialize、真实加载一致性、HOSTCUDA T1/T2形状/数值/计量与模型不变；也未认证新history可达、控制改进或原误差原因。P1/P5 gates均false。若需要完整656实验必须另行授权和协议，不能由此两call probe外推。

## CLI与fixture

工作目录 `/home/gbk/code/LeWorldModel/le-wm-main`。worker仅运行pytest；下面metadata catalog由MAIN酌情执行，不启动模型：

```bash
.venv/bin/python -m pytest -q tests/test_agent_p5_short_start_native_abi_20261029_v1.py

.venv/bin/python -m experiments.spectral_reachability.agent_p5_short_start_native_abi_20261029_v1 \
 --report /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p5_raw164_jepa_h2_cuda_20261029_v2_cuda_MAIN01/report.json \
 --report-sha e5a561215bfe7b6f1c1c224e7661211e3f237fb16458a6a409f5a1da0fc08579 \
 --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p5_raw164_jepa_h2_cuda_20261029_v2_prepare_MAIN01.json \
 --prepare-sha 09070780c3ce782a71ae01492e0c4b801de976e9a36cdf9ee1bf1a53be98e924 \
 --strict-receipt /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p5_raw164_jepa_h2_cuda_20261029_v2_consume_MAIN01.json \
 --strict-sha 783b65f67899791b5697bfe0819ccfb3231d57212b9e3e85aefc02419459c0ea \
 --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p5_short_start_native_abi_20261029_v1_catalog_MAIN01.json
```

catalog只输出全164定位refs，status=`all164_actual0_IO_pointers_resolved_not_executed`，明确`arrays_decoded=false`、`upstream_full_strict_rebuilt=false`、`production_ready=false`。API materialize只返回RAM数组和局部抽取receipt，不持久生产、不调用模型；其真实执行留给MAIN。当前实现不是prepare/run/consume闭环，边界刻意保持为最小设计+fixture。
