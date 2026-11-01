# 参数化root来源与事件驱动反馈runtime适配接口

新增且仅修改本stem源码、tests、本文。旧runtime/attribution v2/所有旧source seal与结果不改。仅实现有限接口和小型mock验证，不加载模型/下载/调用CUDA或物理，不生成新批次或协议文件，不写共享进度文档。当前磁盘约375MiB，测试只用少量临时fixture；保留200MiB余量。

## 移除的是接口单根硬绑定，不是补造第二批来源

`bind_sources`接受显式root SHA和MAIN独立核对的capability report、response protocol、response prepare三个路径及文件SHA。无需root_index=1、四根固定顺序或全局ROOT常量。默认必须给出非空prior-root排除表，当前root不得与prior/excluded相同；root-disjoint不等于fresh holdout或科学准入。

prior-root表还必须逐项对应外部固定SHA的实际先前完整20步两臂报告（`prior_batch_refs`），本版接受原三轮runtime报告schema，并逐根核对两臂确为同一root。不能传入任意无关root列表绕过首批排除，也不能把两个arm当两批；后续其他schema的实际批次需独立版本明确接入。

本版支持既有root-specific v1 capability/response的数据格式。新的不同根仍须由独立、经审查的producer生成自己真实source pack；不能复制旧root1 JSON改root字段，亦不能复用旧root1物理RGB来冒充新根。不实现source producer，不将旧硬编码producer改造成新根生成器。

## capability→response→runtime身份与封存

读取/核验：capability外部固定SHA、原protocol SHA、同根qualified census、3role×41probe完整plan/actual trace、selection内容SHA；response外部SHA指向同一capability/root；response prepare SHA、所有producer source/archive/args seal、原输入/model binding字节均一致。selected prefix的预测向量、真实RGB和真实编码来自capability同根同role同step5 slot；root RGB3 stack与raw历史也对齐capability保存的context identity。验证native192上下文、goal/Q、float32归一化、实际RGB文件与像素SHA。

返回小型内存binding与`content_sha(binding)`，包含root、排除表、source refs、所有读取源SHA、model/goalQ/actual RGB/reset identity及本NEW runtime源码SHA。**函数不写protocol或seal，不会在缺包时生成占位。** MAIN后续若有真实新包，须在任何执行前将该binding与本adapter以及实际外部driver/encoder/compiler/guard代码封存，seal args至少含`binding_sha256`。构造`FeedbackAdapter(binding, expected_sha, arm, runtime_source_seal)`时检查绑定重建、runtime source seal及同root binding SHA；每次选择授权前再次验证source/seal，最终receipt复核binding。

上游保存trace/source真实性由原封存证据与MAIN审查支撑，本接口不重新执行原物理/encoder，也不证明隐藏模拟器状态可完整reset。mock来源仅存在pytest临时目录，明确不是真实第二批。

## 适配接口与状态机

```text
await_prefix --accept_prefix--> await_observation (step5)
await_observation --planning_request--> await_selection
await_selection --accept_selection + durable SHA ack--> await_execution
await_execution --record_execution完整5步--> await_observation (step10/15)
await_execution --record_execution最后5步--> completed20
completed20 --record_terminal_evaluation--> 仅保存评价，不产生新选择
任意错误 --> failed，禁止重试、退款、补步或重新选根
```

`accept_prefix`接收外部guarded五步完整trace、本臂实际prefix RGB、独立预测prefix context及root reset identity收据。检查原prefix像素/动作/预测向量与rolling history，不能跳过前缀就接入未来latent。

`planning_request`只接受step5/10/15，本臂`purpose=real_feedback`的实际三帧像素hash、两个已执行raw packet及其encoded native192 context；核对重叠latent（仅原始2e-5数值容差）与float32 action normalization。actual选择observed，imagined选择自身predicted；两臂都可记录真实编码，但control不得偷用它选优。

观测必须附`encoding_receipt`：同root/arm/boundary/binding、原model bindings、CUDA设备、三帧pixel SHA、raw-history SHA、context内容SHA、单次encoder/projector attempted=completed=1、encoded_frames=3、errors空。step5最后latent还必须与source pack的真实prefix编码见证相符。terminal评价收据使用单帧/原终图/`native192`内容SHA及`terminal_evaluation_only`。这些是外部实际driver应提供的可审计字段，本接口不把调用者自报或mock收据当作亲自执行了神经网络。

`accept_selection`核对root/arm/boundary/binding/context/goalQ，完整十候选的有限值/合法动作/原Q代价/stable argmin、successor滚动与independent replay完整context相等。只有外部`persist(sealed_selection)`返回同一内容SHA才给出一次五步dispatch授权；异常或坏ack永不授权。实际编译/预测/数值导数/资源预算与底层CUDA计数是MAIN外部driver责任，本有限接口不声称测量或匹配这些资源。

`record_execution`只接受该授权、本臂原始五步命令的完整guarded receipt、连续primitive-step trace和实际终图pixel SHA，成功才推进三帧/两packet及predicted successor。partial/abort/错根/错臂/错动作/重复回执均fail-closed。`guard_authorized`必须来自外部实际逐步RGB guard，本接口不替代物理guard或重算所有中间图的接触/可见性。

`record_terminal_evaluation`只允许完整step20之后，绑定本臂真实终图及`purpose=terminal_evaluation_only`，用native192复算原Q；其latent不得倒灌step15选择。完整接口receipt始终`P1_admission=false, P4_admission=false`，不会因mock、root-disjoint或外部自报guard成功升级为科学独立批次。

## 本次验证与MAIN命令

在`/home/gbk/code/LeWorldModel/le-wm-main`运行纯算术/临时保存文件mock：

```bash
.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p4_feedback_root_runtime_20261027_v1.py
```

命令行只有只读来源校验，无`run/prepare`或写文件命令；必须提供真实新root来源全部参数，缺失即拒绝：

```bash
.venv/bin/python -B -m experiments.spectral_reachability.p4_feedback_root_runtime_20261027_v1 --help
```

具体调用参数为`--root-sha256`、`--capability-path/--capability-sha256`、`--response-path/--response-sha256`、`--response-prepare-path/--response-prepare-sha256`、`--prior-root-sha256s`、逐项成对的`--prior-batch-path/--prior-batch-sha256`及可选`--excluded-root-sha256s`；不提供不存在的新包路径示例或假SHA。原首批root必须在prior表并有实际报告绑定，因此原包不能变成第二批。

剩余条件：真实不同根的capability/response完整封存来源；MAIN前瞻root roster及排除协议；NEW实际CUDA/compiler/guarded-physics driver接入、source closure封存和资源拦截测试；之后才由MAIN统一执行并核验真实第二批。独立P1机制/任务收益及P4科学效益仍未成立。本轮只有mock/接口证据，没有新增实际神经或物理结果。
