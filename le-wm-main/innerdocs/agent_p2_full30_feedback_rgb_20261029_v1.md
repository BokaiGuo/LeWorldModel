# 八条I/F full30真实RGB只编码桥

## 固定测量，不是规划

四base按job0..3顺序，每base I后F，每条cp20/25/30，共24槽。control时间20/25/30对应warm10后绝对30/35/40。source逐槽绑定自己的新物理off PNG、state/snapshot、原task、原goal192、raw30 bits、new witness、实际最后raw10及原float32 action normalizer。无Q计算/阈值/重选、无predict/AD/训练/physics、新task=0。
不因相同动作/像素/终态而去重。早停缺图保留null和原24分母；当前固定实际receipt全8full30，预期24新encoder+24projector。
原O及cp<=15仅由原完整RGB receipt和逐条prefix exact物理guard继承；绝不拿O后缀latent冒充I/F实际图。后续分析自行配对actualQ/modelQ，本桥不计算这些量。

## 权威及复用边界

source.PINS固定四个新own physical receipt：

|job|SHA256|
|---|---|
|0|565c432e08a7965079a10c709d90773edab4a967004ec07e43202b88a6c96d6a|
|1|fc08ef91da4f158218d78a17da640a2664a8d1e63ebdf6e3b6fe6b6df5c2ee5e|
|2|94157f79fd3ca2ea28e825914e56a9f364ba4f87e7871b6d09d9c9eeaa7b9c38|
|3|8a66e6bb41c716c4d0b01b405745a7d356fcd4cfc7b6e17178e4084c27199f2d|

绝对目录均为 `/home/gbk/code/LeWorldModel/le-wm-main/results/`，文件名 `agent_p2_full30_control15_feedback_physical_20261029_v1_consume_J{0..3}_MAIN01.json`。
joined_MAIN01 SHA2842de482d30ff5898dc52915f58ddca292d0c4958f84198377d8c852c9ae101；原O RGB receipt42164a…/report915b67…，完整SHA/path见PINS。
从receipt.bindings按实际prepare/report SHA唯一解析来源，不猜路径；逐项读取/rehash原prepare/report/completion/receipt/全部artifact/source/weight闭包；不复制物理目录进入新输出。
明确 `full_upstream_rebuilt_here=False`：继承MAIN已经完成的strict，不再次运行上游consume或物理重建。纯producer.validate_execution只核原动作计划ABI，不冒称重签原strict。
仅复用冻结五cp10桥的lossless codec、precision/class inventory primitive，以及原Trace/NoAD/native IO hooks。不复用其task构造器，不修改旧globals/schema。own prepare/run/consume均新schema。所有科学gate=false，known2非新独立根，FLOPs unknown。

## 精确运输证明

每图保存7数组，顺序：RGB uint8(224,224,3)、preprocessed f32(1,1,3,224,224)、encoder input f32(1,3,224,224)、hidden f32(1,257,192)、projector input f32(1,192)、projector output f32(1,192)、native latent f32(192)。
每图payload字节为150528/602112/602112/197376/768/768/768，共1,554,432。24图payload37,306,368 bytes；每个NPY1固定128字节header，共21,504字节。
每文件gzip上界对n=payload+128采用 `n+(n>>12)+(n>>14)+(n>>25)+31`（zlib compressBound加gzip wrapper）；168文件共 **37,344,384 bytes**，无需假设可压缩。运行codec逐数组实测header和压缩上界。
native module真正IO字节仅encoder input/output+projector input/output：24×801024=19,224,576；不将重复保存副本冒称新神经调用。
全部重复JSON（initial roster/event/native seal/report/completion/失败文件）aggregate硬cap16MiB，另留8MiB final reserve。总最坏运输 **62,510,208 < 67,108,864 bytes**；余4,598,656。reserve不是增额：正常write保持8MiB不可动，final只可在总64MiB内使用；metadata即使final仍受aggregate16MiB。
最大全返回168数组/241事件/414文件。逐文件写前检查类型、尺寸、组量、总量和floor200MiB；不删字段缩图扩cap。whole600s从verify前起至report fsync及全部artifact hash，completion外部SHA另绑定该边界；completion出版后再次检查wall，超时留terminal_failure拒绝成功。
GPU allocated3GiB/reserved4GiB，batch1，TF32/autocast off，MATH request加实际flags before/after，不冒称底层kernel parity。

## MAIN运行（全部canonical绝对路径）

在内层仓库执行；worker未执行生产prepare/run/consume。

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
module=experiments.spectral_reachability.agent_p2_full30_feedback_rgb_20261029_v1
prepared=/home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_full30_feedback_rgb_20261029_v1_prepare_MAIN01.json
output=/home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_full30_feedback_rgb_20261029_v1_cuda_MAIN01
receipt=/home/gbk/code/LeWorldModel/le-wm-main/results/agent_p2_full30_feedback_rgb_20261029_v1_consume_MAIN01.json
.venv/bin/python -m "$module" prepare --output "$prepared"
prepare_sha=$(sha256sum "$prepared" | cut -d' ' -f1)
.venv/bin/python -m "$module" verify --prepared "$prepared" --prepare-sha "$prepare_sha"
.venv/bin/python -m "$module" run --prepared "$prepared" --prepare-sha "$prepare_sha" --output "$output" --authorize-host-cuda
report_sha=$(sha256sum "$output/report.json" | cut -d' ' -f1)
completion_sha=$(sha256sum "$output/completion.json" | cut -d' ' -f1)
.venv/bin/python -m "$module" consume --output "$output" --report-sha "$report_sha" --completion-sha "$completion_sha" --receipt "$receipt"
```

每步成功才运行下一步；输出exclusive，不覆盖partial。消费返回24行identity+actual192/latent descriptors、所有count/IO/bytes/继承权威，不输出模型优劣结论。

## 验收边界

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
.venv/bin/python -m pytest -q tests/test_agent_p2_full30_feedback_rgb_20261029_v1.py
```

仅RAM/fake tensors/临时磁盘保存ABI。不加载真实模型、不执行神经/physics或生产fullstrict。完整build夹具须走真实磁盘pin读取、source/artifact join、prepare写入和verify，完整168数组/所有metadata消费并统计实际尺寸；这些不是CUDA或科学证据。

完整磁盘回归以真实已保存的四job plan/checkpoint/PNG身份为输入副本，临时重绑定receipt/report/prepare/completion、保留原source目录所有文件，走own build→write→verify→Store→consumer。仅本新模块的PINS/MODEL_PINS/model_metadata/source_files seam被夹具替换；不改旧模块globals。模型权重替身明确是从未加载的fixture bytes；RAM模块输出只是验证运输/IO接线，非神经数值结果。
该完整24图回归实测prepare约353,259 bytes，168数组gzip约295,744 bytes，全部重复metadata约1,066,419 bytes（浮点wall位数可能微变）；真CUDA module inventory不由RAM替身认证，仍须MAIN真实before/after检查和aggregate16MiB写前约束。
另用全168个噪声数组逐一NPY/gzip往返并实际填满16MiB metadata，验证不可靠低熵假图压缩过预算。完整saved consumer负例重算report/completion所有外层hash后，仍拒绝未知事件、orphan、缺array、precision/model变化、seal不一致、超wall，以及把新latent内容连同所有descriptor和seal一致重写但与原projector输出不一致的篡改。
冻结前曾由完整磁盘回归发现shape census内tuple/list JSON往返不同，已改为literal lists；非模型或科学负结果。
