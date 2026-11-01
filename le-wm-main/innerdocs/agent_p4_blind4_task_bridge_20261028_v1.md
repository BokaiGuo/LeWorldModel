# Future legal blind4 → task/planner bridge

本独占三件套只补现有四槽下游缺口；experiment-code用于固定输入/分母与RAM回归，不训练、不运行神经/物理、不采集或读取四个预留根。没有run/签approval/签freshness命令，不改旧source/consumer/MAIN或Pauli文件。

## 查重与当前实际进展

已存在source producer `agent_p4_fresh_source_20261028_v2.py` SHA `5d2103a883b58db85e9877fb25697ef835e317c529771691a3bdec701a582d3a`；严格consumer `agent_p4_fresh_source_saved_consume_20261028_v2.py` SHA `dcdd2e76559f084a87435e9aaf551eac1ff1d9affdba7eac09ed888a322d29d5`。producer三文件pins沿consumer固定，不借v1。authority原输入补证已有`agent_freshroot_authority_20261028_v1.py` request/validate，不重做审计/历史扫描。已有fresh_confirmation_prerequisites文档的“无producer”是早期快照，不据此重复实现。现有moving encoder/planner仍固定已知两槽及旧SHA；P4 runtime assets仍known两根，不是blind4桥。

本轮未取得真实完整历史cert/approval或合法四槽sourcepack；不会尝试prepare真实四槽，也不会用calibration_only启动。新seed不等fresh。已有工程补证接口不等补证已通过。

## 准备层与权限隔离

prepare先只读明确prepared JSON，强制最新v2 `fresh_candidate`，无条件拒绝calibration_only、v1或null cert；在此前拒绝路径不打开sourcepack/blind。复用既有证书schema与外部MAIN approval绑定，重核content SHA、明确完整覆盖/未决为空、相同历史身份和原绑定，不自签真实性/scope。此窄检查不替代原证据origin review。随后绑定v2 producer/consumer源码和source closure，精确核原recipe/witness/action/environment/float32 warmup，再直接调用已有v2完整saved consumer，核native telemetry/全数组/历史批内去重/账本，不重新实现其物理审计。

固定4槽×3phase，原reachable recipe prepare SHA `98ffb806018a270f5ef27ffa76be929393a74355c213bc274bf158c72d80b086`、原protocol SHA、RGB observed0/5/10、raw10前五[.1,0]后五[0,.1]、goalRGB、source证书SHA/state/pose/snapshot/history/raw/root-goal身份精确连接。失败/duplicate不删行，不补候补；合法source若k<4，prepare保存4行与null资产并标BLOCKED，不构造缩减planner请求。

admin含原sourcepack/cert/approval与完整consumer收据；deployment只含blind路径、RGB/raw hash和不可逆身份/source证书SHA，**不含实际state/goal7、witness或certificate路径**。部署不读sourcepack/证书决定动作。源计量保留在admin，new encoding实际费用null，不能把已有source warmup/physics费用写0。future caps16encoder+16projector/0predict/AD/physics仅该编码阶段计划预算，不是已执行或全pipeline总成本。

## 可调用真实数组接口及精确停点

`task_export(deployment, prepare_sha, slot_index, packet, values)`可直接接MAIN将来授权的encoder callback。packet必须通过原moving blind检查并逐像素/raw hash相等；values严格为四个有限float32[192]，顺序observed0/5/10/goal，无cast repair/去重。原identity192 Q、normalizer、完整3latent/2tokens、H2×5steps、packet logit cap、原冻结模型不变，返回真实core.validate_task/core.task_id封存的task，不造actions。history raw10只在原规则下归一化一次；不是零padding。

`planner_inputs(exports)`要求完整四个新schema exports及共同source/history/recipe cohort、固定slot-major；返回8条GN/D2P请求，含两算法实际可用gn_task/d2p_packet/q/task_id。拒绝额外witness/state字段、错Q/模型/raw/精度/goal或拼接cohort。**不把老moving loader的slot0/1约束猴补成4，不冒用老task/export SHA**。

本桥不生成/执行encoder或planner：MAIN仍需独立审计的四槽HOSTCUDA编码runner及真实16enc/16proj数组/账本收据，再调用本API；传入latent的float32/像素绑定不能独自证明来自真实冻结模型，所以`encoding_provenance_verified=false`。同理旧fixed-two planner build/prepare仍拒绝新四槽；真实count4 planner runner/物理downstream需后续新实现，不能把本8请求列表说成已支持P4 run。本任务限定桥实现，未扩写CUDA driver或authority框架。全部freshness/P1/P4/scientific/execution释放字段false，P1独立前置不被桥绕过。

## MAIN命令与测试

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p4_blind4_task_bridge_20261028_v1.py
# 仅未来合法资产齐备后；当前MAIN不会启动/打开四槽
.venv/bin/python -B -m experiments.spectral_reachability.agent_p4_blind4_task_bridge_20261028_v1 prepare --source-dir /ABS/LEGAL_V2_SOURCE --source-sha ACTUAL_SOURCEPACK_SHA --prepared /ABS/LEGAL_V2_PREPARE.json --prepare-sha ACTUAL_PREPARE_SHA --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_p4_blind4_task_bridge_20261028_v1_prepare_MAIN01.json
.venv/bin/python -B -m experiments.spectral_reachability.agent_p4_blind4_task_bridge_20261028_v1 verify --input /ABS/BRIDGE_PREPARE.json --sha256 ACTUAL_BRIDGE_PREPARE_SHA
```

只排他新本stem JSON、4MiB输出/200MiB余量，Reader继承单输入/aggregate限制；源码三文件与实际source/assets SHA一起封存，verify重建相等，prepare后源码漂移拒绝。测试只合成RAM RGB/NPZ/identity/cert，mock既有consumer接点；不是补造真实证书，也不证明freshness或物理结果。生产consumer本身已有独立v2 RAM回归，MAIN审后真实调用。当前没有可执行的真实prepare命令实参，不填未来SHA。

## 实际回归与冻结范围

本轮新桥41项；联合原moving encoder和authority接口测试实际136 passed in 1.70s，另实际运行CLI `--help`。命令：

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p4_blind4_task_bridge_20261028_v1.py tests/test_p1_constructive_moving_history_encode_cuda_20261027_v1.py tests/test_agent_freshroot_authority_20261028_v1.py
```

覆盖既有producer要求的历史projector源码绑定、recipe/raw/slot/telemetry、拒绝不完整consumer、prepare精确重建及source漂移、冷子进程禁止native/model导入、输入state/witness泄漏拒绝。保留开发失败记录：首轮30pass/2fail因RAM历史fixture未显式包含root_state；补闭包回归时133pass/3fail因fixture少原projector bindings；均只补齐fixture，不放宽生产合同。最终联合通过不等科学通过。

本轮未执行真实prepare/verify、编码、planner、物理或网络，未访问reserved root/sourcepack。冻结仅本stem源码/测试/innerdoc；后续修改新版本。精确未决依赖仍为完整合法历史cert及其外部MAIN approval、相应合法v2 source/strict闭包、MAIN另行审查的count4 CUDA编码与planner runner。可在将来授权后通过本桥的prepare/verify和数组API连接，当前failclosed不是补证成功。
