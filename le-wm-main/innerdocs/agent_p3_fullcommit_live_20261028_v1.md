# P3 known2 full-commit live calibration

独占source/tests/本文；旧MPC20、旧RAM合同和全部结果不改。已核已有results中无fullcommit/full_horizon实测文件。same-feedback报告cdf463…与v2 saved验收84a007…是每5步重规划、执行first5的不同协议，不借其outcome填本次结果。源前置直接复用已绑定P3 original tasks与P4原生live machinery，没有新增IPC或fresh证书要求。

## 固定协议与真实入口

2 known根×fixed/adaptive，共4 episodes、12反馈槽，顺序slot-major fixed/adaptive。每臂独立constructor→原reset两次→原placement→warm10，逐帧RGB/native7/body dynamics保存，warm0/5/10精确核原source。原GN固定prefix5从原layer.commands[:5]取得；两臂新鲜prefix终点PNG像素/state/snapshot/goal身份必须精确一致。禁止reset(poststate)、旧endpoint替代、借旧物理结果。

目标public20含共同prefix5，warm10另计。5/10/15各自实际RGB经原Handler编码，rolling context由原warm3加各自实际观察更新，raw history只取原warm+自身已返回动作最后10；state7/奖励/成功仅评估器使用，planner request白名单不含这些标签。20只终点评价编码。模型保持原projected CLS192/identity Q/normalizer，非P5 patch模型。

空闲节点复用原local H1/H3 signals、spectra、runtime_signal/choose（chi=.5、gain=.05），自然候选4/8，不采用旧equalquota12/4。使用原bounded_ridge解与i/(n−1)固定fraction，float64方向映射后只cast一次raw float32。每候选完整wholeH与独立segmentedH全部原生buffer严格一致、按原Q全H终点选stable index；另付一次selected完整H replay，不以prefix replay代替。

H3选中就承诺完整15动作；10/15仍新鲜RGB并付H1/H3信号，但不调度、不替换承诺。H1每包结束可重新调度。若新的fullH装不下剩余20−boundary，则整个节点refused，停止该臂，不截断、不强制H1、不补步、不重选。原unsupported reconfiguration/composition同样整节点拒绝。原done/earlystop原样保留；clean engineering/refusal不代表completed20。完整时域是否相同由实际returned控制步数重算。此比较**反馈观测频率相同、利用/决策频率不同**，不宣称只隔离一个horizon因素。

## 原生任务与资源合同

绑定`.venv/lib/python3.12/site-packages/stable_worldmodel/envs/pusht/env.py:347–355`的eval_state：success是joint XY<20且min(abs(angle差),2π−abs(angle差))<π/9；reward是负**原native7 Euclidean距离**，包含未wrap角度与agent速度，不能用wrapped angle或XY距离代替。state7=[agent_x,agent_y,block_x,block_y,block_angle_mod2pi,agent_vx,agent_vy]。保存body未wrap angle/全部float64 bits与goal/goalRGB/goal_pose；wrapped角误差只是分列诊断，不改变原成功门。本模块native_return_contract另独立复算原API并对齐kernel.metric，保留原source超过2π时可能负angle项的行为，不修旧科学阈值。

每反馈信号predict8（含nominal4及ADforward4）、J2、scalar VJP/grad/raw384、SVD2；程序H,n成本为predict=H(2n+1)、replay=2n+1、metric=n、SVD/solve/compile各1。每臂信号predict cap24，程序fixed27/adaptive51，总inclusive predict cap51/75；API/samples/context tokens、AE/predictor/pred_proj均分轴。每次新节点Ledger caps取该臂原上限减所有旧attempts，失败不退款，dispatch前拒绝。signal/program费用独立列，不把不同单位相加；inclusive wall允许嵌套，不能加成总wall/FLOPs。FLOPs、CUDA_ns、backward engine调用未知null。

真实native Meter hook覆盖constructor/reset/set_state/goal_render/goal_setter/goal_bind/public/Space.step/render/close及encoder/projector。完整4臂预期constructor4、close4、reset8、set_state16、goal_render8、goal_setter12、goal_bind4、public120、physics1216、render128、encoder16/projector16。reset工程cap12、physics1224等保留原hook上限，strict按实际原constructor无implicit reset逐臂核算，不能用cap当已完成数。每返回control额外eval_state1（总最多80），原step内reward计算不独立hook；metrics不藏作新模型调用。wall1800s、output512MiB（final reserve64MiB）、disk floor200MiB、CUDA allocated3GiB/reserved4GiB，失败清理close和partial日志保留；无重试/补步。

## 消费与边界

prepare/verify重建原task/goal/Q/normalizer/model/source闭包并绑定本三文件，不读新物理outcome选根。run只有MAIN授权且HOST CUDA存在才load frozen model，无CPU fallback；与已验证P4同进程原生接口，无全局函数monkeypatch。完整selected命令在dispatch前fsync，原生forward/AD/grad/J/AE/predictor/proj/fullH每边界数组原样journal。

consume仅saved NumPy/PIL：全12槽/4臂、部分失败原账、物理JSONL完整ABI及LIFO attempt/return、逐臂逐region资源、所有PNG/native7/bodybits、exact currentRGB→float32 preprocessing、原生模块/梯度全数组、自然候选/原Q/stable winner、完整selected replay、原prefix与所有commitment offset逐项复核。拒绝用first5或新动作替代H3后10；已拒绝节点后不得dispatch。工程complete与completed20分离；P1/P3/C3/freshness/scientific均false；无训练/fitting/新根认证。RAM测试只验证代码，不是模型/物理或研究证据。

## MAIN命令（worker未prepare/run）

在`/home/gbk/code/LeWorldModel/le-wm-main`，先冻结三文件，再执行：

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/agent_p3_fullcommit_live_20261028_v1_tests.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p3_fullcommit_live_20261028_v1 prepare --output results/agent_p3_fullcommit_live_20261028_v1_prepare_MAIN01.json
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p3_fullcommit_live_20261028_v1 verify --prepared results/agent_p3_fullcommit_live_20261028_v1_prepare_MAIN01.json --prepare-sha "$ACTUAL_PREPARE_SHA"
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p3_fullcommit_live_20261028_v1 run --prepared results/agent_p3_fullcommit_live_20261028_v1_prepare_MAIN01.json --prepare-sha "$ACTUAL_PREPARE_SHA" --output results/agent_p3_fullcommit_live_20261028_v1_MAIN01 --authorize-known-cuda-and-physics
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p3_fullcommit_live_20261028_v1 consume --prepared results/agent_p3_fullcommit_live_20261028_v1_prepare_MAIN01.json --prepare-sha "$ACTUAL_PREPARE_SHA" --output results/agent_p3_fullcommit_live_20261028_v1_MAIN01 --report-sha "$ACTUAL_REPORT_SHA" --receipt results/agent_p3_fullcommit_live_20261028_v1_consume_MAIN01.json
```

两个SHA变量必须取上一步实际文件，不提供未来SHA。所有输出exclusive xb，已有失败目录/manifest须保留，另MAIN02路径。生产未运行，是否full20/拒绝/数值精确重放以及任务收益均unknown，待MAIN审读与实际执行。
