# Model-only 五词库 selector ABI（新探索性算法）

仅新增本stem source/tests/本文。用户授权实现saved consumer，先交源码供MAIN审读，不执行生产saved analyze/consume，不运行GPU/CPU神经、物理、网络、训练/Q拟合，不改旧封存、MAIN文档或共享总账。使用experiment-code技能组织实现和RAM回归；用户边界优先，无图像或新实验产物。继承旧blocked P4 goal不覆盖，本个人有界交付为可审源码、RAM测试和冻结三SHA，不宣称研究goal完成。

## 规则与方法身份

固定四task×两role×五word，原顺序`teacher.LIBRARY=(zero,uZ,-uZ,uS,-uS)`，完整40原预测成本必须全部存在、有限且非负。每个原teacher context的五项按`min((model_Q,library_index))`稳定选择；精确相等以最小原index打破，包括+0/-0数值tie，不用epsilon。缺任何成本或槽位整体fail closed，不选子集、不回退、不去重重复raw词、不退款。输出恰好8个选择，task-major再Z/S-major。全部40候选与每组五成本保留。

这是现在固定的**新探索性算法**。原实现先按局部ridge生成并冻结uZ/uS，之后才评价共同库；原生成diagonal `Z:uZ,S:uS`不是共同库argmin策略。新见证逐组保留原diagonal的index、cost、原plan/prediction digest与是否相同，绝不改写原策略履历。规则在看过known-root结果后制定，不是fresh预注册验证；无新阈值、目标或Q，不宣称修复P1或选择器有效。

## 不从物理标签选择

直接输入原teacher compiler报告SHA `d5ab99d55bacd73c8d9cf0fb774819bb46af63100053198858e639d8fa746738`及原strict receipt SHA `d82109eb8de582e59a4f0485307c01c9b91f9f84551c218c61cca4da24335213`。不依赖support、task-metric、actual latent或physical末态报告。生产build完整重跑原teacher.consume的saved校验：源码/prepare/模型、8context、40库预测及replay、module/native数组/AD/ledger闭包；不调用其run，不产生神经/物理。原producer的prepare验证会读取历史acquisition身份/PNG及当时已验收receipt，它们只认证原输入，不进入选择算术。

纯接口`select40(candidates)`实行严格model-only字段白名单，候选/plan/prediction中额外physical success、actual Q或native reward字段均拒绝。每个prediction来自原evaluate输出：原native `[3,192]`和shifted context `[3,192]`只核最后一行endpoint，不比较前两行的不同语义。独立以float64原identity192和原goal重算Q，要求精确等于原plan/evaluate成本。封存native输出/shifted context的原receipt、端点/goal float32数组SHA及prediction digest；完整原plan digest独立重算。所有依赖artifact SHA纳入闭包，不能仅信日志hash。

同role五项必须同task/anchor/blind/context digest和prefix raw；两role同task共享相同五suffix buffers。zero须literal正0，inverse为原float32逐位符号翻转而非时间逆序。JSON commands必须精确表示float32值，不默默round/clip/tanh/重标度。

index12只缺真实末图latent，不缺模型预测；本算法不读取其物理缺项，也不据此过滤候选。合法模型成本若选中12，选择见证仍可存在，但不能由此把未知物理评价改为已知。

## 稳定接口与动作长度

`build()`返回完整selector；`consume(selector_path, selector_sha)`独立完整build再比较，返回`selected_plans`恰好8项，包含：

- task/role/root/condition、anchor_id、task_id、blind/context digest和原report/receipt SHA。
- selected_original_index、library_index/label、cost、selected_original_plan_digest、selected_prediction_digest和完整prediction witness。
- `commands`: **suffix5** JSON `[5,2]`，`commands_dtype='<f4'`、raw bits和数组SHA。这是原实际rawfloat32 buffer的部署拷贝，非重新优化动作。
- `required_original_prefix_commands`与`original_full10_commands/raw_bits`单独保留。部署前必须具备同一prefix完成后的真实history/context15，不能从root把suffix5冒full10，也不能把full10在prefix后再执行导致重复首段。
- 全五成本/原plan与prediction digests、精确minimum ties、selection_digest，原生成diagonal身份。selected原plan digest不是新selection digest，两者不得混淆。

本consumer**不授权dispatch或physical release**，也不匹配已有physics给它打分。MAIN需审读后决定是否能合法同位复算：逐项核原task/condition/role/prefix/history/start、原动作字节、原执行分母/失败/缺12与评分authority，不能按相同word名称或末态收益接旧结果。现有结果可否用于新的探索性选择评价不等于fresh确认，2known roots不膨胀为8/40独立样本。

资源继承原producer实际resources并明确不是免费新planner；本saved selector新增模型/物理0，total FLOPs未知null，无等总算力声称。无P1放行、无fresh资格、无科学成功结论。

## MAIN审读后可用命令（本次未执行生产CLI）

工作目录 `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p1_model_library_selector_20261028_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.agent_p1_model_library_selector_20261028_v1 analyze --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p1_model_library_selector_20261028_v1_MAIN01.json
.venv/bin/python -B -m experiments.spectral_reachability.agent_p1_model_library_selector_20261028_v1 verify --selector /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p1_model_library_selector_20261028_v1_MAIN01.json --selector-sha <实际SHA>
.venv/bin/python -B -m experiments.spectral_reachability.agent_p1_model_library_selector_20261028_v1 consume --selector /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p1_model_library_selector_20261028_v1_MAIN01.json --selector-sha <实际SHA>
```

analyze只独占新建JSON不覆盖，verify/consume不写旧结果，上限16MiB。每个selection另绑定独占schema与selector contract digest。RAM测试不是模型/物理或新算法科学证据。实际新增37项、联合common-library/support panel共113项RAM回归通过；没有执行真实saved CLI，没有生产结果SHA。最终文件冻结后三SHA随交付发出，后续修复必须新版本。
