# 后续P阶段统一资源合同（saved ABI / fixture only）

## 任务与复用边界

独占新source/tests/doc，不修改P1/P2/P3、GRASP、旧summary/runtime或封存结果。
实现仅依赖Python标准库，**不import producer，不存在full strict调用、模型/物理执行入口**。
用户授权范围内只读了原预算文档、源码及现有report/receipt的keys/计数字段ABI；
未对真实产物执行本组件的saved/build/validate CLI，未调用旧consumer。

已读的合同入口：

- `innerdocs/agent_p2_common_resource_grasp_spec_20261029_v1.md`
- `innerdocs/agent_p2_common_resource_grasp_cuda_20261029_v1.md`
- `innerdocs/agent_p2_common_resource_summary_20261029_v1.md`
- `innerdocs/agent_p3_common_budget_20261028_v1.md`
- `innerdocs/agent_p3_effective_schedule_20261029_v1.md`

P2同1902 predict API/sample上限、450秒同步搜索上限是资源向量盒，composition自然1062不padding；
GRASP532+100更新、632 scalar backward不是85J/16320 mathematical VJP的同义表达。
P3完整20步时每臂51predict，但commit15/feedback5分别1/3 solve、7/9 SVD等不同；
不能由相同predict宣称等实际总算力。全部old gate保留，所有新科学admission=false。

## 统一资源向量

`empty_vector / validate_vector / sum_vectors`固定完整AXES，每轴有：
`unit, attempted, completed, status, evidence`。

- encode/projector/predict：API、samples、frames分轴。
- AE/predictor/pred_proj实际嵌套module调用、samples/context frames/flattened rows分轴；
  不能把module调用和外层predict相加当独立forward总数。
- J request、requested J elements、mathematical VJP directions、显式grad API、raw backward callbacks、
  scalar loss backward、engine backward、独立JVP各自分轴。
- AD primal packets、solve、model/array SVD、Adam、virtual-noise分轴。
- native public steps、integration physics substeps、actual encoded RGB feedback、render/reset/goal bind分轴。
  public_step来自原全episode native计量，含warmup，不冒充仅20步控制成本。
- training保留一轴，不提供训练入口。

缺字段保持`status=unknown, attempted=null, completed=null`；**unknown不是0**。
attempt与complete是独立付费记录，完成不能超过尝试，失败不退款。
聚合时只要任何分项unknown，该轴合计unknown，不输出仅已知子集冒充完整成本。
`sum_vectors`只允许调用者明确不重叠的费用项；不能加入同一份cohort两次。
继承原编码cohort单列一次，按臂/两off根分摊unknown，不重复16/16或均摊伪计费。

FLOPs没有实际算子工作量instrumentation：总量及每个算子均字面`unknown`。
不按predict数、sample数、VJP方向、wall或求解次数套固定权重换算FLOPs。
不同算子的费用轴不相加生成“总算力分数”。

## 真实保存schema消费者

### P3：`p3_ledger(receipt)`、`p3_saved(report, receipt)`

独立实现原P3 delta元数据算术，不import `core.delta/audit_ledger/consume`：
根据resource/shape/directions重算全部原27 ledger轴，核delta、attempt/complete、cap与wall，
拒绝未知operator、非整数/布尔shape、非法blocked/failed生命周期及退款。
原Ledger.events按进入index排序，**不是return排序**；父signal_compile/program_compile/replay
会在子失败后unwind并记failed，数组中父failed在前、子entry在后完全合法。
本组件按entry时间/嵌套区间与最早failure/refusal边界核验，不因数组位置拒绝已发生子操作。
失败父子都保留attempt，成功子保留completed；失败后新started操作拒绝。
`complete=false/disposition=failed_or_blocked_nonrefunded`不冒称成功，shared-panel会block该trace。
scalar_backward和AD_primal_packets没有原ledger独立测量字段，均unknown，不由J/VJP/grad补0或推算。
这认证**保存元数据算术**，不认证其实际神经输入/梯度或原native全闭包。
原producer的显式grad和raw callbacks不冒称engine内部反传次数；J元素缺元数据则unknown。

`p3_saved`直接消费当前8episode/24node report，保留全部固定槽与未到达节点。
原native receipt的per_episode census作为另一个证据层导入，标记`not_rebuilt`；
不声称独立重计physics/RGB journal。compiler里视觉samples=0只说明compiler不采图，
真实RGB另付费用的batch/frame尺寸没有保存映射时unknown，不能复制该0。
实际encoder调用数可作该recipe的编码边界计数；其完整RGB/latent正确性仍属MAIN原strict。

P3 node wall有独立时钟：started/end同基准；inclusive含attempt sink、body及嵌套IO/check，
不含本操作return sink；外层signal仍含内层日志。
各operator inclusive不能求和冒充总wall，也不能相加不同node时间戳。
现有per-episode whole wall未独立instrument，明确unknown；whole-job wall原样另列。

### P2：`p2_vector(resource)`、`p2_saved(report, receipt)`

直接适配当前common-resource六槽strict receipt的`resources/accounts`与report行序，
不要求不存在的新schema。独立进行单位展开、完整轴映射、attempt/complete约束和固定batch1/full3
关系核对；**输入的计数来自原SHA绑定receipt，不声称本次从native事件重新计数**。
engine backward/JVP/FLOPs原为null则仍unknown。新schema提供该类数据需新版本明确适配，不猜。
保存账本算术重算与原strict认证分开；不因read+hash就宣称fullstrict成功。

`stage_wall`保留P2原timings；每槽search scope显式包含nominal/bank/AD/CPU solve/capture/
search IO/export，排除loader、offline strict、post-search persistence。
无分项实测的CPU/IO墙钟不通过相减推算。

## MAIN saved文件入口（本worker未执行生产）

`build_manifest(manifest)`读取外部SHA绑定report/receipt及producer source，验证已有schema与关联，
调用上述纯算术适配器；前后rehash。source必须包含本组件固定的真实producer源码SHA，
不能绑定任意无关源冒充。元数据单文件上限128MiB（真实P3 report约97MiB）。
不扫描整个results、不打开模型权重/数组/物理根、不重建fullstrict。

证据层级永远是`SHA_bound_saved_ABI_only_not_recertified`。
P2 receipt含report_sha256，强制与manifest report相同。
P3原receipt没有report SHA：不伪造签名关系，要求显式actual prepare SHA与report绑定、
核原counts/per_episode_neural相同，保留这只是ABI linkage的限制，绝不当独立报告认证。

MAIN可生成如下新输入manifest（这是示例文本，本worker未写生产JSON）：

```json
{
  "schema": "agent_downstream_resource_contract_20261029_v1.inputs",
  "kind": "P2_common",
  "report": {
    "path": "/home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_common_resource_grasp_cuda_20261029_v1_cuda_MAIN01/report.json",
    "sha256": "d781b493ecd624db846e3225ed0177bd3fe250d6ba9213d925b2a57956dd2242"
  },
  "receipt": {
    "path": "/home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_common_resource_grasp_cuda_20261029_v1_consume_MAIN01.json",
    "sha256": "d3ee465f2fa6d31345f369f9514165c4817c51a29c3d00ffdd2d681e5829cc1f"
  },
  "sources": [{
    "path": "/home/gbk/code/LeWorldModel/le-wm-main/experiments/spectral_reachability/agent_p2_common_resource_grasp_cuda_20261029_v1.py",
    "sha256": "ea763d96eee1040652a3ea02badcb5e8e37b5781530da721f73d2a63d9cb147d"
  }]
}
```

两SHA来自MAIN已交接并写入原文档的实际权威，本worker没有再生产consume认证它们。
MAIN可将上述完整JSON用apply_patch新增至：
`results/agent_downstream_resource_contract_20261029_v1_inputs_P2_MAIN01.json`。
本worker不新增生产manifest，只交付source/tests/doc；JSON的空白会影响manifest SHA，
请MAIN保存后实际sha256sum，再传CLI，不能用report SHA替代manifest SHA。

以下为第二份完整JSON，建议MAIN独占新路径：
`results/agent_downstream_resource_contract_20261029_v1_inputs_P3_MAIN01.json`。
本轮仅做文件SHA检查得到下列实际值，没有运行production saved/fullstrict：

```json
{
  "schema": "agent_downstream_resource_contract_20261029_v1.inputs",
  "kind": "P3_effective",
  "report": {
    "path": "/home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_effective_schedule_20261029_v1_MAIN01/report.json",
    "sha256": "840ef006b7d3be039ad84546d2ff8bfe903c3c776a6c6a187fad20c2d4fc7116"
  },
  "receipt": {
    "path": "/home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_effective_schedule_20261029_v1_consume_MAIN01.json",
    "sha256": "a48d63d272a4da1b4a270fa3dee60008d1e5d59251146b5ddd8143ba14632b5f"
  },
  "prepared": {
    "path": "/home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_effective_schedule_20261029_v1_prepare_MAIN01.json",
    "sha256": "8e0a80f9b907e7d8b61c605f28c73f295f53ab1447c973fe8b99ebbd241f01ee"
  },
  "sources": [{
    "path": "/home/gbk/code/LeWorldModel/le-wm-main/experiments/spectral_reachability/agent_p3_effective_schedule_20261029_v1.py",
    "sha256": "7cef2600e9844d981f97fdd6c385bff0c3ae9bdf55713ac8c92cebd28b003c23"
  }]
}
```

不要拿P2或P3旧结果作为新sharedpanel结果。

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
.venv/bin/python -m pytest -q tests/test_agent_downstream_resource_contract_20261029_v1.py

# MAIN审读后才执行；输入manifest及SHA由MAIN实际生成。
.venv/bin/python -m experiments.spectral_reachability.agent_downstream_resource_contract_20261029_v1 saved \
  --input ACTUAL_RESOURCE_INPUT_MANIFEST.json --input-sha ACTUAL_MANIFEST_SHA \
  --output results/agent_downstream_resource_contract_20261029_v1_saved_MAIN01.json
```

## 直接给新P3 shared-panel的消费接口

无需改旧producer，不是新必拒框架。新dispatcher在自己的预算/ledger接缝可直接调用：

```python
from experiments.spectral_reachability import agent_downstream_resource_contract_20261029_v1 as budget

# 每个node保存后立即验证其原生ledger的元数据算术。
node_resource = budget.p3_ledger(native_resource_receipt)

# 独立、非重叠node资源可合计；actual RGB/native应明确由新panel自身账本补入。
vector = budget.sum_vectors([node["vector"] for node in node_resources])

# 行至少包含slot_id/root_key/strategy/resources/wall/identity。
# identity六项都是实际预先绑定SHA：model/task/goal/normalizer/history/action_domain。
# 缺实际身份就None，不能拿任意占位SHA签通过。
decision = budget.validate_shared_panel(
    rows=all_fixed_rows,
    expected_slots=preregistered_order,
    compared_axes=["predict_calls", "predict_samples", "predict_frames"],
    caps=preregistered_axis_caps,
    wall_scope=preregistered_scope,
    wall_cap_seconds=preregistered_wall_cap,
)
```

`expected_slots/compared_axes/caps/scope`必须在新实验prepare前固定，不能由结果选择。
验证完整行序、同root各策略的六项输入身份、未知关键轴、实际attempt上限、选定轴的
attempt/complete相等与共同wall scope/cap。未知wall、不同scope或overrun均block。
失败不删除槽、不补计算凑数；返回blockers用于工程拒绝，不是科学负结果。
成功也仅为`selected_resource_axes_equal_only`，equal_total_compute/equal_FLOPs/equal_wall仍false。
同wall上限不代表实际wall相等，不同比例VJP/solve/encoder不能凭predict相同而忽略。
既有report适配器的identity默认None（未重建全部原history/task），必须由新panel真实prepare补入，
所以不能把旧P2/P3资源panel直接拿来冒充新实验准入。

也可将validate_shared_panel的参数保存为JSON后调用：

```bash
.venv/bin/python -m experiments.spectral_reachability.agent_downstream_resource_contract_20261029_v1 validate-panel \
  --input ACTUAL_SHARED_PANEL_RESOURCE_ARGS.json --input-sha ACTUAL_ARGS_SHA \
  --output results/agent_downstream_resource_contract_20261029_v1_sharedpanel_MAIN01.json
```

输出只允许本stem的results新文件，限128MiB、不覆盖，文件及目录fsync。
该saved工具无NN/physics/fullstrict预算，不承诺生产重建资源认证。

## 回归与未完成

35项fixture覆盖：独立27轴delta展开、未知operator/计数/shape/status/cap篡改、
failed不退款与blocked不收费、未知传播、P2真实ABI映射、P3全8/24与earlystop、
inherit cohort仅一次、布尔/NaN拒绝、同predict不同solve、身份/unknown/wall/cap拒绝、
实际临时JSON SHA/source pin以及冷import不导入producer/torch/physics。
首轮32passed/1failed捕获compiler encoder_samples0误带到实际视觉采集；修为unknown，未改旧件。
MAIN提出嵌套失败entry-order后，新增两项真正原Ledger RAM测试（父失败、成功子后另一子失败），
并拒绝伪造failure之后的新dispatch；最终35passed / 0.27s。

### 给Bohr/Pasteur的接口同步文本

`validate_shared_panel(rows, expected_slots, compared_axes, caps, wall_scope=None, wall_cap_seconds=None)`。
row的`resources`须完整AXES；未知scalar backward/AD primal等不得补0；
`identity`六SHA为model/task/goal/normalizer/history/action_domain；wall使用显式scope/seconds/status。
可先调用`p3_ledger(native_receipt)`独立映射原ledger；failure仍付费且complete=false，不能签成功。
调用合同/预算在prepare前固定；返回最多selected_resource_axes_equal_only，不签等总算力/FLOPs/科学门。
本stem由当前agent独占，其他agent仅import/call，不修改这三文件。
当前会话无可用thread/agent消息工具，不能直接送达；请MAIN转发此段，不伪称已通知。

未完成：MAIN对本source审读、真实saved manifest调用、新P3 sharedpanel自身真实身份和native
acquisition samples/frame级instrumentation、各算子独立FLOPs/engine内部backward/JVP测量。
本worker不运行生产saved build/fullstrict，不签任何科学gate。
api-and-interface-design技能只用于边界、未知语义和可直接调用接口；没有改动旧version。
