# Moving反馈执行资产与MAIN接线 v1

仅本stem source/tests/本文；旧driver、结果、共享文档完全不改。experiment-code技能用于
先固定执行入口/预算/失败分母，后RAM回归；不运行神经、physics或网络，不派生或训练。

## 有限核查结论：复用，不另造P4

真正可执行moving控制器已有：`p1_moving_live_feedback_physical_20261027_v1.run`及
`Handler.encode_observation/compile_suffix/verify_suffix_seal`、`native_factory`、`Meter/Sink`、
`Store`、`DispatchGate`、`native_transition`。源SHA固定6742a1b3…754ace9。已有12 episode
真实report SHA 990843e8…ca895e3：它是2个独立任务、四层×O/I/R，不是12根。
本版逐字节绑定原prepare/selection/backend/model/source闭包，输出原run直接接受的
native_prepared和prepare SHA；不复制编码/编译/物理算法，不重新执行历史实验。

原路径为warmup10、prefix5、唯一step5反馈、suffix5，R仅新RGB编码进入重编译，I用原想象
prefix，O继续原sealed suffix。terminal10编码只评价，原latent identity192不是物理收益。
原compiler请求白名单没有native状态/任务success；状态/goal7/snapshot用于恢复、拒绝和评价。
原goal bind保持原task goal，不改renderer goal_pose来制造图。with_target=false固定。

`p4_multiround_feedback_runtime_20261027_v1`已有固定旧root1三轮实现；
`p4_feedback_root_runtime_20261027_v1`已有root参数化状态机但物理/模型driver由外部提供。
它们不是本moving来源的通用三轮接口，不能换root字段或借图后称fresh moving P4。
因此本版只提供已存在单反馈路径的真实dispatch接线，不增加同类多轮框架。

## 新的可执行接缝

`prepare()`不导入旧执行器、torch、gym，不加载模型/环境；读取固定原prepare/report，核全部
依赖/模型字节，列原真实callback符号与行号、原预算及原native指标字段。该清单不是代码审计
或科学认证。status始终single_feedback_inputs_bound_NOT_P4_ready，不是metadata绿灯。
`verify(value, canonical_sha)`完整重建，不能prepare后改source/tests/doc或输入。

`dispatch_known(value, canonical_sha, NEW_OUTPUT, MAIN_AUTHORITY_PATH, AUTHORITY_BYTE_SHA)`是
实际生产接线：先verify和MAIN授权、写独占fsync launch intent，再import固定旧模块，核实际
module路径/SHA，调用原verify，最后调用原run，传原AUTH。不是把mock receipt当真实dispatch。
旧run自行执行CUDA allocation、模型冻结hash、source重核、native factory、逐步guard、
编码/编译meter、durable seal、原native指标和12分母失败报告。异常原样传播；不重试补根。
launch意图的actual_computation=null，不能由意图推算真实GPU/physics账。
失败launch保留，阻止同output再启动；不清理/覆盖现场。输出目录须直属inner results且用本stem。

MAIN外部authority精确字段：

```text
{schema:NAME.MAIN_authority,reviewer:MAIN,input_sha256:canonical prepared-input摘要,
 output:新绝对输出目录,scope:known_root_single_feedback_engineering,
 host_cuda_and_physics_authorized:true,confirmatory:false}
```

本worker不创建真实授权。不支持confirmatory/formal efficacy，即便自报P1通过也不能使用此
已知根路径解锁P4。授权只是MAIN工作流输入，不是密码学或科学真实性证书。

## 预算和任务指标

保留原600s/64MiB/200MiB floor/256KiB final/3GiB allocated/4GiB reserved，不改阈值。
原full-run上限public_step240（warmup120+control120）、predict40、encoder/projector各24、
J/SVD各8、数学VJP1536；其余reset/setter/render等逐轴原值直接透传。上限不是实际账。
attempt/return/pending从原Meter/Sink得出，不退款；FLOPs/device timing/inherited cost未知null。
O计算更便宜，不称三臂全资源等价；主反馈比较R−I不被R−O替代。
本wrapper launch记录额外<64KiB，位于原output之外，不冒充原64MiB内的实验数据；MAIN需要
完整whole-job quota时还须统一纳入launcher，不能把原driver配额扩称全pipeline硬保证。

原native eval_state成功位置是agent+object的前4维，原angle语义不改。object_position_error、
wrapped angle、native_distance、native_success分开；object_success原为null不填0，不由对象
投影非正推完整任务失败。新运行没发生时actual_new_work/null，native metrics未知/null。
源report只作已存在路径的来源证据，prepare不抽取有利outcome或编出新的native结果。

## MAIN命令与剩余缺口

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/agent_p4_runtime_assets_20261028_v1_tests.py
.venv/bin/python -B -m experiments.spectral_reachability.agent_p4_runtime_assets_20261028_v1 prepare
```

prepare完整JSON到stdout，MAIN可保存到自己新的本stem文件，取实际字节SHA后：
`verify --input ABS_INPUT --sha256 ACTUAL_BYTE_SHA`。API verify/authority使用canonical SHA，
不是stdout含换行的字节SHA。CLI run-known还需--output/--authority/--authority-sha256；
只有MAIN审读后明确决定工程复跑时才使用，不建议重复已有封存实验以凑结果。

未完成：独立P1/fresh root认证、moving多boundary实际driver与真实新源/编译回调接入、全资源
计量/等预算与prospective native efficacy协议。现有12episodes只单次反馈，不满足长期P4。
本实现不声称补齐上述科学缺口或物理放行。RAM回调测试仅验证本wrapper到原run接口的调用
顺序/参数/故障保留，不是原模型预测、物理行为或实验成功证据。

## 本轮实际验证

专属RAM回归 **21 passed in 0.13s**。真实固定prepare/report/model/source依赖的saved-only
`prepare()`及`verify(v,canonical_sha)`已实际返回正常，2 tasks/4 layers/12 episodes，
native prepare byte SHA为
`bc3fbc91d7a19ae256b24db22ad05b2526fcbe454e3b55d904136eba2f8fdff5`。
未创建真实MAIN authority/launch/output，也未调用旧native.verify/run或模型/环境。
因此没有本轮新实验、原native runtime重跑结果或科学效益结论。
