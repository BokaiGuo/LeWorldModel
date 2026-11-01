# E3已批准16格：输入resolver交付

本件是已批准协议 `agent_e3_frozen_model_task_matrix_20261029_v1.md`（SHA `51321bd5a139f0a41b24078df5a58e47548560859d6e00597ab966bcae36bf45`）的输入实现说明，不是新协议或另一套资格包装。实现固定known2×G0/G1×H1/H3×Dxy/Dx全部16格、独立own task identity、G1既有RGB/native7/latent精确接点及纯RAM mask/chart；没有optimizer/模型/physics/fullstrict/run命令。科学门和production_authorized始终false。

## 可用接口与证据边界

source：`experiments/spectral_reachability/agent_e3_frozen_model_task_resolver_20261029_v1.py`。

- `build(inputs_absolute_path, inputs_sha256)`：保存文件接线，返回完整16 cells、ready/blocked、全source/input字节bindings与固定预算。
- `verify(prepared_absolute_path, prepare_sha256)`：重新build并逐字canonical比较整份自身输出，包括own source/tests/doc；不调用任何上游fullstrict。
- `mask_chart(logits, K, domain)`、`mask_gradient(gradient, K, domain)`：K严格1/3，literal有限数组，无bool，f64原参数ball→原NumPy tanh(f32) chart；Dx参数必须已mask，raw_y明确正零。不会静默裁切旧incumbent、优化动作、改原历史或重估normalizer。

复用冻结 `agent_p3_actual_cp5_task_builder_20261029_v1.resolve_actual_cp5_tasks`，源SHA固定 `703f5f552c1911159b2144f72711c542ca48c408ed4e306b1fc17b019df889f1`。该接口只读saved三权威＋两个原export，验证原known2全部四分支与actual cp5 chronology/原raw10/一次normalize；没有调用producer/consumer。新resolver再次绑定所有实际读取输入及其source/tests/doc。原builder保留原任务，新resolver不修改任何旧文件、旧task_id或原goal。

新任务限定原两个root SHA和原模型tensor identity；不能通过CLI换根。G0沿原local goal/native goal；G1用原prepare.tasks[r].root_context.latent_history[-1]，原export task_contract.pixel_sha256[2]与物理cp0的pixel SHA精确连接。两条已认证H3 physical分支的rootstate、完整snapshot、cp0/cp5位串、task/root身份、prefix5、cp5 encoded observation与goal全部一致后方可借用。按原report/receipt绑定的**两份实际PNG**有界读取并解码224×224 RGB，验证file及pixel SHA，不选其中更优结果、不新编码。native_goal来源是同一cp0 state7，不猜goal_pose或从object XY捏造state。

原pixel→latent数值关系是继承的MAIN完整encoder/task-export证据，不声称本轮重新运行神经parity。PNG重新解码和f64 snapshot位串由本工具实际验证；未重hash全部旧模型权重、物理journal/PNG闭包。`full_upstream_rebuilt_here=False`。模型权重/normalizer训练ancestry、fresh资格、P阶段科学门均未因此改变。

真实ABI注意：blind export的`native_root_sha256`实际为null，这是原权限隔离，不是缺失待捏造字段。实现严格保留null，改核原blind/source SHA及task身份，经原task.binding.root_sha256与已完成physical receipt的同task/root绑定，再连实际cp0 RGB/native7；没有把blind export说成含native证据。真实保存数值/PNG的临时transport夹具实际捕获并修正了开发中误要求此字段非null的接线；未更改任何旧export或资格门。

另一个实际层级：physical `source_witness.task`是原control0任务，actual-cp5任务位于`source_witness.request.task`，不能混同。两者分别与原task和新cp5 context比，request完整digest还须精确匹配cp5 builder保存的对应bank分支。此接缝也由真实ABI临时夹具捕获，未放宽旧consumer或将original任务改标observed。

仅使用源prefix字段白名单：root/cp0/cp5、original goal、raw10/prefix5、原观测及context。虽完整JSON保存文件含未来suffix/outcome，本工具不根据它们生成目标、选根/预算/配置，不比较source_witness.result或future native metric；其opaque整文件SHA只作authority身份封存。测试改变未消费future字段后，16格planner输入/预算不变，authority identity随字节SHA改变是正常的。

planner拿到 `cell.task.body`，只有原context、goal192、projection、normalizer、模型/root身份hash、horizon/domain和新identity摘要；不含native7、native_goal、image/evaluator路径或原authority内容。`cell.evaluator`、`goal_evidence`和root_provenance必须留在MAIN/evaluator域，不将整个cell交给planner。新task_id与task_sha256绑定全部cell/goal/horizon/domain及来源，不能冒充旧task身份；没有改变旧cp5 builder的原root/action-domain hash。

## blocked、上限与预算

显式null authority、已识别的不支持/未完成planner/native receipt schema返回完整16 blocked；缺某root所需cp0/5元数据保留该root8格blocked；只有G1原图缺失则该root G1四格blocked，G0不捏图也不免费声明已解码。SHA冲突、非有限/bool、位串/像素/任务/前缀不一致属于损坏，failclosed抛错，不包装成合法missing。缺项不替补、不padding、不填零目标，不签可生产。

JSON caps：inputs64KiB，shared prepare16MiB/report256MiB/receipt32MiB，native receipt32MiB/report16MiB（实际原report12,145,714bytes），export压缩及解压各8MiB，单PNG256KiB；本reader unique实际读取累计512MiB。原builder自己的caps不变。新输出16MiB、floor200MiB，exclusive xb，文件仍open时flush/fsync，之后fsync目录。不会写生产results，除非MAIN明确调用prepare；fixture仅临时目录。

输出预算只是下一数值runner的预声明：16格/48nodes/1584candidate slots，6528predict（3456非AD＋3072AD primal）/1536scalar reverse，32新encoder/projector，480public/4864完整路径physics。资源计数分轴，不相加嵌套操作、不冒充equalFLOPs。**本resolver自身全部神经/物理/优化调用为0**；未来numerical runner完整tensor/事件运输shape证明仍未实现，`numerical_runner_transport_proven=False`，不得因输入ready宣布生产runner可用。

真实保存前缀ABI运输试验：只读取现成prepare、已存cp5 join与native receipt，全部known2/8原分支和4物理分支重绑到临时目录；shared-report的最小envelope由已认证保存的request/provenance重建为**测试夹具**，不是新的生产receipt，也没有跑原132MiB report/fullstrict。采用实际原latent/raw/native位串和4份原PNG，临时build→write→verify得到16ready/0blocked，最终版序列化3,741,477bytes（约3.57MiB，路径长度随tmp目录可能变动），低于固定16MiB。不能把这一输出当MAIN实际prepare；未来真正源读/闭包仍由MAIN调用上述CLI。测试还覆盖pure RAM bool/nonfinite/mask/chart、missing/null分母、schema blocked、SHA/PNG/native7/prefix/context篡改、own seal和真实fsync顺序/排他写。

## MAIN显式输入JSON与CLI

用apply_patch创建MAIN自己的新输入JSON；worker未创建生产输入或prepare：

```json
{
  "schema": "agent_e3_frozen_model_task_resolver_20261029_v1.inputs",
  "planner": {
    "prepared": {
      "path": "/home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_shared_panel_20261029_v1_prepare_MAIN01.json",
      "sha256": "7ca85050eeab4e47ed35cf84c284ed7e08fc64bce8064d7db9450074045b643a"
    },
    "report": {
      "path": "/home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_shared_panel_20261029_v1_MAIN01/report.json",
      "sha256": "668db35d79eb650419e4efa42dfc21a15cdf91725700b2f202e2a2ec975bcd6a"
    },
    "receipt": {
      "path": "/home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_shared_panel_20261029_v1_consume_MAIN01.json",
      "sha256": "d23ffa48bc954306bc27a15f924da8daafffc9b4bc76a91e59aa949ab1abbafd"
    }
  },
  "native_receipt": {
    "path": "/home/gbk/code/LeWorldModel/le-wm-main/results/agent_p3_h3_witness_physical_20261029_v1_consume_MAIN01.json",
    "sha256": "48a1a5ecadf7043e3162ca104d8f43b0d59baf88655a076f675c9f96d697b9f7"
  }
}
```

建议输入路径 `/home/gbk/code/LeWorldModel/le-wm-main/results/agent_e3_frozen_model_task_resolver_20261029_v1_inputs_MAIN01.json`；将实际sha256sum结果代入INPUT_SHA。所有文件参数绝对路径，无未来默认SHA。

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_e3_frozen_model_task_resolver_20261029_v1.py tests/test_agent_p3_actual_cp5_task_builder_20261029_v1.py
sha256sum /home/gbk/code/LeWorldModel/le-wm-main/results/agent_e3_frozen_model_task_resolver_20261029_v1_inputs_MAIN01.json
.venv/bin/python -B -m experiments.spectral_reachability.agent_e3_frozen_model_task_resolver_20261029_v1 prepare --inputs /home/gbk/code/LeWorldModel/le-wm-main/results/agent_e3_frozen_model_task_resolver_20261029_v1_inputs_MAIN01.json --inputs-sha INPUT_SHA --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_e3_frozen_model_task_resolver_20261029_v1_prepare_MAIN01.json
.venv/bin/python -B -m experiments.spectral_reachability.agent_e3_frozen_model_task_resolver_20261029_v1 verify --prepared /home/gbk/code/LeWorldModel/le-wm-main/results/agent_e3_frozen_model_task_resolver_20261029_v1_prepare_MAIN01.json --prepare-sha ACTUAL_PREPARE_SHA
```

prepare exit0可以输出blocked；MAIN必须读ready/blocked和全部16状态，不能只读exitcode。verify只验本输出整份重建，不神经/物理/fullstrict。测试结果与冻结三SHA在最终交付列出；尚无真实16格生产prepare或数值实验结果。
