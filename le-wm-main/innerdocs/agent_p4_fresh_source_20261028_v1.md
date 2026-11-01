# 四槽原生 source adapter v1（MAIN审读后执行）

独占本stem source/tests/innerdoc。复用原moving-history `episode/warmup_commands/witness/snapshot`，
不调用其旧两槽build/execute，不patch旧slots/globals；复用frozen physical `metered_class/native_factory`，
逐操作实际计量在原生方法entry前持久化attempt、return后completed。无神经接口。

四槽/root/reset seed/sign完全继承已封存preregistration与原recipe。一次construction及两次独立replay，
每phase两reset、真实warmup10、一次预固定witness10。原goal由construction真实终态产生；
replay从原生reset重建而非setter恢复中间state，精确校验三帧/状态/snapshot/action。
原default outline goal_pose不改，不拿它制造taskgoal。合法native success早停保留。

## 模式与历史证书

`prepare --mode fresh_candidate`必须传`--history-cert /ABS/...json --history-sha REAL_SHA`：
采用已有`moving_fresh_source_exclusions.v1`及其MAIN外部coverage approval，不另造authority框架。
必须unresolved空、preconditions_met真、coverage content重算匹配、全部历史physical identity齐全，
MAIN approval外部文件SHA/内容/cutoff/scope匹配，并绑定历史evidence文件字节。
这仍是MAIN输入的权威声明，不是adapter能自行证明所有历史运行事实。
当前没有该合格证书，fresh prepare必须失败，不能产生伪freshness。

`prepare --mode calibration_only`不要求历史cert；缺证时historical_coverage=unknown，
所有row/result freshness=false、confirmed=false。允许真实工程采集，不要求不存在的全局访问日志，
不把新seed/新目录等同fresh。两模式都不开放P1/P4；confirmed恒false。
若提供cert，即使calibration模式也必须有效，不能静默忽略错证。

每个实际warmup10 guard通过后、witness动作之前，新GateStore从真实保存观测提取state7/pose/
snapshot/RGB3身份，做历史与批内去重；失败槽已观测root同样进入batch排除，不补根。
replay root/history必须精确与同槽construction一致。root-goal旧hash在construction返回后另核。
原XY/角度数值别名排除阈值1e-9 pixel/1e-12 rad不改。
blind严格只有rgb_history:uint8[3,224,224,3]、raw_action_history:float32[10,2]、goal_rgb:uint8[224,224,3]；
certificate/witness/state留在source输出，不授予部署selector。尚未接通四新槽encoder/planner/P4，
不能冒充旧两槽task或把原witness送入planner选动作。

## 预算、失败与持久化

全四槽最多24reset、48reset-only set_state、24goal_render、32goal_setter、8goal_bind、
240public_step、2448physics_step、252explicit_render；1native constructor（共享环境、每phase独立reset）。
wall最多1200秒含verify，output16MiB，final reserve512KiB，disk floor200MiB。
pre-dispatch检查wall/disk/cap；每张真实PNG/state先durable再guard。实际调用及pending保留，
`physics=2*reset+10*public`、set_state=2*reset、goal_setter=reset+goal_bind独立末尾核账。
原native每public10积分不改变；warmup仍原raw动作，无第二次/100或tanh。

固定四行始终保留。原生source guard/重复/early-stop拒绝标rejected_fixed_candidate并可继续后槽；
工程/资源/写盘异常停止后槽，留not_started，保留首错、close次错及pending，不重试补根。
全部4槽×3phase在启动前建行；未到达phase也保留not_started，失败phase为failed_pending。
sourcepack_complete表示工程批处理完，不表示4/4可达或科学成功；reachable_count必须另报k/4。
prepare绑定本三文件、协议/prereg、旧recipe闭包/native源码及cert资产；verify完整重建，run前后再核。
prepare后不得改闭包；修改只能新版本。MAIN须收到最终三SHA及冻结通知后才实际prepare。
新prepare还绑定实际已导入的experiment模块闭包，不使用当前Git/mtime假冒旧加载依据。

## MAIN命令（本worker不运行物理）

在 `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
env OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/agent_p4_fresh_source_20261028_v1_tests.py
.venv/bin/python -B -m experiments.spectral_reachability.agent_p4_fresh_source_20261028_v1 prepare --mode calibration_only --output results/agent_p4_fresh_source_20261028_v1_prepare_MAIN01.json
.venv/bin/python -B -m experiments.spectral_reachability.agent_p4_fresh_source_20261028_v1 verify --prepared results/agent_p4_fresh_source_20261028_v1_prepare_MAIN01.json --prepare-sha ACTUAL_PREPARE_SHA
.venv/bin/python -B -m experiments.spectral_reachability.agent_p4_fresh_source_20261028_v1 run --prepared results/agent_p4_fresh_source_20261028_v1_prepare_MAIN01.json --prepare-sha ACTUAL_PREPARE_SHA --output results/agent_p4_fresh_source_20261028_v1_MAIN01 --authorize-main-physics
```

fresh_candidate同prepare接口另传真实cert路径/SHA；缺失必须退出非零。没有未来SHA占位结果或假证书。
RAM测试只验证工程合同，不是物理/科学证据。没有运行CUDA/CPU神经/真实物理/网络/训练。
剩余：MAIN源码审读及实际native执行、合格历史certificate、四槽下游adapter和P1科学前置。

## 本worker实际验证

先后新测试25 passed、35 passed；新增RAM prepare/verify序列化及外部SHA、wall/cap提前拒绝、
witness commit先于dispatch、latest_action替换、goal_pose漂移测试后，与原moving-source联合回归：
**64 passed, 11 subtests passed in 19.62s**。全程只使用RAM fake native API、NumPy算术和sealed metadata。
history cert正向样例仅RAM语法fixture，不是真实MAIN证书或来源认证；变异包含有效重封checksum后
的state/pose、缺snapshot、缺source family和缺projector source依赖，仍须拒绝。
原source的slot数与build前后相等；新adapter绝不修改旧recipe/slots/globals。
prepare/verify输出序列化已在RAM接通；未实际生成prepare或sourcepack文件，未运行run。
CLI prepare/verify/run交MAIN审读调度；无全局访问日志不妨碍calibration_only工程采集。
