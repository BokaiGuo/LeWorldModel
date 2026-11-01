# step20 两端点 CUDA 编码派生证据（prepare-only 交付）

仅一批真实20step/3round运行的imagined与actual两臂，不是两批、两个独立根或新物理执行。原报告及已封存attribution auditor v1均不改动。本次worker只运行prepare/verify和非神经测试；只有MAIN可执行显式run命令。

## 原始绑定

原目录：`results/p4_multiround_feedback_runtime_20261027_v1_prepare_v1`。
prepare读取并SHA绑定report、protocol、prepare_receipt、完整runtime.events.jsonl，重建唯一root/arm/step20关联，解码两张原始RGB PNG并验证文件SHA与连续uint8 HWC像素SHA。实际两端点的绝对路径、event index、文件/像素SHA写入新小protocol的endpoints字段；不复制PNG或旧大protocol。

冻结model binding取原protocol绑定的`policy/weights.pt`及`config.json`，流式核验SHA，不下载或复写。loader及原图像路径源码与历史seal manifest比较；新source seal另存本工具/tests/本文、loader、GPU preflight、原transform/_image_tensor、既有LeWM/MLP、ViT与torchvision变换实现等小源码。该源码seal不是全Python依赖供应链认证。

沿用`preflight_pusht_d2p_p0._image_tensor`及`run_pusht_counterfactual.image_transform`：uint8 HWC→CHW→float32 scale→原ImageNet Normalize→Resize224→CUDA [1,1,3,224,224]。不替换为带ToImage的其他transform。直接原model.encode(pixels-only)，其encoder CLS经过已有model.projector，保存native float32 [192]，不平均patch token、不拟合readout、不用预测latent代替观测。

goal/Q只保存原protocol路径、原字节SHA及规范数组内容SHA；run从已绑定文件读取原数组，用native192复算原Q cost。不复制约1.2MiB旧protocol或Q数组。

## 硬预算、错误及派生receipt

每臂单图单batch、encoder1+existing_projector1；全局各cap2。pre-dispatch检查CUDA设备/shape、全局与逐臂上限、encoder→projector顺序，failed attempt不退款不重试。每次返回后CUDA synchronize，异步错误不能计completed。predict/predictor/action_encoder/pred_proj及AD入口被拦截并计blocked；无模型预测、导数或物理步骤。加载checkpoint到CPU内存、CPU图像预处理、保存输出latent及hash不是CPU神经前向；所有神经模块必须CUDA，禁止CPU回退。

一份prepare只允许一次launch，任何setup/endpoint失败都停止后续dispatch，保留两槽位与first_error/后续错误。endpoint native数组先独立落盘fsync，再算Q及处理下一臂。完整/部分实际entry journal持久化attempt和completion/error；即使后续journal异常，已积累rows/entries不丢失。开始前实际预分配256KiB最终receipt，4MiB最小headroom，journal128KiB cap；完全不可写设备/进程硬杀仍可能阻止最终落盘，保留现场不可重用。

新输出：`imagined.terminal20.native192.npy`、`actual.terminal20.native192.npy`、`encoding.events.jsonl`、`execution_binding.json`、`receipt.json`。完整状态`two_terminal20_latents_saved`；否则`partial_or_failed`并非任务负结果。receipt含每模块attempted/completed、所有entry、blocked/forbidden attempts、完整errors/first_error、前后model state hash、model/file/pixel/goalQ SHA。P1/P4 admission及efficacy均false。

后续独立auditor v2按`source_report_sha256/root_sha256/arm/primitive_step=20`连接receipt；验证prepare/source/PNG/latent SHA与shape/dtype、完整entry预算及模型不变后才消费。此时可以补齐15→20预测/真实latent误差，但不能追改v1“当时没有保存step20 latent”的历史结论。

## MAIN命令

工作目录`/home/gbk/code/LeWorldModel/le-wm-main`。worker已生成新的`results/p4_terminal20_cuda_encoding_20261027_v1_prepare01`，prepare receipt SHA见交付回复。MAIN使用该精确SHA，不另prepare后冒用同一SHA。

```bash
.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p4_terminal20_cuda_encoding_20261027_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.p4_terminal20_cuda_encoding_20261027_v1 verify --prepared results/p4_terminal20_cuda_encoding_20261027_v1_prepare01 --prepare-sha <交付的prepare.json文件SHA>
.venv/bin/python -B -m experiments.spectral_reachability.p4_terminal20_cuda_encoding_20261027_v1 run --prepared results/p4_terminal20_cuda_encoding_20261027_v1_prepare01 --prepare-sha <同一SHA> --output results/p4_terminal20_cuda_encoding_20261027_v1_cuda_main01
```

prepare/test不证明CUDA成功，不预写派生latent。本工具未改共享docs、旧report或v1 sourceSHA绑定。若实现需要修订，使用独立新版本，不覆写已经封存的v1源码。
