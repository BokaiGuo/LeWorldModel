# Actual direct-H2 FD: independent raw-evaluation arithmetic notes

2026-10-27。新增 saved-only 审查，不修改封存 runner、prepare、eval、journal、analysis 或 receipt；不运行模型、AD、图像编码、训练、优化、物理。这里的测试通过指保存链和算术一致，**不是导数精度的科学准入**。

## 实际证据与绑定

实际目录：`results/p1_direct_fd_cuda_20261027_v1_cuda_main01`。MAIN 的一次真实 HOST CUDA 运行完成32次 wholeH2、64predict与64actionencoder，128条模块 entry 全部完成；errors=[]、first_error=null，0encoder/projector/AD/physics，模型前后 tensor hash 都为
`4b12c01f5859f4b46bea478f3a8d5e2d3f516df456a622a3feec31bde7e33546`。
本次只读重算文件/模型字节哈希及来源绑定，不声称重新计算了神经模型的 tensor hash。

|封存文件|SHA256|
|---|---|
|prepare.json（`results/p1_direct_fd_cuda_20261027_v1_prepared01`）|`fdb4c376c574ed61915e88b569980330edf6fa89dd974e8fa9033b9da9af149c`|
|report.json|`d77f7aadf3fd3b6bd7da8c32ce499e5dc486a6989b0c178622aba4ff4478eaa1`|
|analysis.json|`4091379f9f179a28338ca8e4b65f48b59e9ce2eeb5c76226eff662e271b064dd`|
|fd.events.jsonl|`1c96ccc2a324abf36cbe399cd2511cd7940866c9bee8c039186f2935fda0a0c8`|
|execution_binding.json|`83240fe696f534e8e6d7094353b0cad3c403abccc85f959ba4ef4fb42616761d`|

全部实际输出1,762,648bytes，低于原4MiB cap。本次只新增测试与本文，没有新增神经/物理实验，也不把继承成本称为零成本全pipeline。

## 独立复算范围

新增 `tests/test_p1_direct_fd_saved_audit_20261027_v1.py` 只导入 stdlib、NumPy、pytest；不导入 FD runner、NativeCuda、selfchecker 或模型包。直接读取全部32条 `eval_00.json`–`eval_31.json`，从原始终态向量计算：

`FD=(z_plus-z_minus)/(2*float32(eps))`，`Jv=原始192x20 logitJ @ 实际保存的float32方向`。

独立重建e0、e10、(e0+e10)/sqrt(2)、PCG64 seed20261027单位随机方向，核对 NumPy float32 logits/tanh rawcommands；每条完整H2的 retained latent history、两条 normalized action history 内容/layout逐项验证。按 root/direction/epsilon/sign身份和固定顺序配对，不按最小误差选择。

直接核对原 GN component 和原导数 journal 的192行 raw-control derivative、first joint logit-J、两个 AD forward packet及 nominal；检查当前 zero-point incumbent 对应 `direct_runs[0][0]`，导数存在 `direct_runs[0][1].derivative`，没有误取第一步优化后的J。zero点raw与logitJ精确相同。没有使用旧审计绿色结论代替这些数组。

逐项重算16组FD与Jv完整192向量、绝对L2/Linf/RMSE、两种相对误差、cosine、nominal midpoint drift；重算8组两epsilon间稳定性。核对全部384条 journal 事件：32evaluation attempts、128module attempts+128独立returns、64packet contexts、32evaluation-file hashes；和receipt逐模块计数一一对应。所有输入、原导数路径、源码、checkpoint字节SHA与prepare一致。

12tests通过：两项完整实际复算/范围回归，以及10项仅内存篡改反例（FD、Jv、relative error、epsilon stability、rawcommands、history、sign、task、模块完成记录、原J）。算术一致性 `rtol=1e-12,atol=1e-14` 只约束重复计算是否一致，绝不作为FD/J精度阈值。

## 全部16组误差，不挑方向或epsilon

根为两条新direct calibration任务，不是此前四根：

- root0：`02a139165f6e8465ec882926c2ceb13a33ca5547f736ed1e3817242902050d06`
- root1：`05844ca2c3ab8ba09846f9084edd71c8365ad74f7ab5961adaa5b0fded6289a7`

下表相对误差分母为 `||savedJ@v||₂`，无投影Q、无拟合缩放、无分母floor。本次所有分母非零，范围1.692088–11.628879。ε采用实际float32值（请求1e-3或1e-2），没有偷换扰动尺度。

|root|方向|ε=1e-3 绝对L2|ε=1e-3 相对L2|ε=1e-2 绝对L2|ε=1e-2 相对L2|ε稳定性L2|稳定性/||Jv|||
|---|---|---:|---:|---:|---:|---:|---:|
|0|packet0 e0|0.001453457|0.000812479|0.000750889|0.000419745|0.001156167|0.000646295|
|0|packet1 e0|0.001354661|0.000726916|0.000772197|0.000414364|0.001181479|0.000633986|
|0|equal packet e0|0.001375575|0.000536372|0.000773146|0.000301470|0.001231180|0.000480069|
|0|unit random|0.001393296|0.000823418|0.000682896|0.000403582|0.001184105|0.000699789|
|1|packet0 e0|0.002373350|0.000263357|0.002524351|0.000280113|0.001801673|0.000199922|
|1|packet1 e0|0.001878840|0.000249669|0.002727014|0.000362378|0.002369427|0.000314860|
|1|equal packet e0|0.002570069|0.000221007|0.002591731|0.000222870|0.002935074|0.000252395|
|1|unit random|0.001928779|0.000280890|0.001936055|0.000281950|0.001910508|0.000278229|

每档epsilon的分母都是8组root×direction，不是8个独立根。

|指标|ε=1e-3范围|ε=1e-2范围|
|---|---|---|
|绝对L2|0.001354661–0.002570069|0.000682896–0.002727014|
|绝对Linf|0.000297093–0.000625265|0.000140321–0.000618840|
|绝对RMSE|0.000097764–0.000185479|0.000049284–0.000196805|
|相对L2|0.000221007–0.000823418|0.000222870–0.000419745|
|cosine|0.999999664–0.999999976|0.999999921–0.999999978|
|midpoint漂移L2|4.118255e-6–1.441507e-5|0.000323610–0.001447517|

8组epsilon稳定性绝对L2范围0.001156167–0.002935074，相对Jv范围0.000199922–0.000699789。较大epsilon在root0的4方向绝对误差均更低，在root1的4方向均更高。没有一个epsilon可依据这些已开结果追认为统一“最佳”设置；也不以min误差取代完整表。

## 有限解读，不设post-hoc pass

这些真实数字显示，在这两个固定root的zero logits、四个预声明方向和两档epsilon上，中心FD与原saved Jv高度对齐；没有观察到这些方向上的整体反号、数量级失配或完全丢失第一packet响应。第一packet单独方向的Jv非零且FD跟随，提供了该点prefix→terminal依赖的数值证据。不支持仅凭此前direct优化表现，就笼统断言该zero点导数路径完全断链。

这不是整个Jacobian的正确性证明，更不是“通过精度门槛”：

- 每根仅4方向，含重复线性组合；既未穷尽20坐标，也未验证所有子空间或所有动作点。
- zero点tanh导数为1；漏乘/重复乘tanh链可能在zero点同样给1，故不能据此排除非零logits的链错误。也不能推广到后续GN iterations。
- 两档epsilon误差走势不同、较大epsilon的midpoint漂移更大，和有限步长曲率及float32减法误差共同存在相容；本实验没有唯一分离二者，不能据此选择唯一数值根因。
- FD和AD共享同一source-bound native forward。它们一致不证明共同的normalizer/model语义没有问题，更不证明冻结embedding的任务充分性或目标proxy正确。
- 仅两条calibration任务，历史仍为causal padding而非实际observed transitions；没有新增独立确认根。
- 没有新的Q、物理执行、控制效能或严格P1准入证据。科学阈值保持null、verdict保持not_assigned，不能将`cuda_fd_completed`解读为科学pass。

## 可重复命令（saved-only）

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_direct_fd_saved_audit_20261027_v1.py
# 可选：仅打印同一独立复算摘要，不写文件、不加载模型
.venv/bin/python -B tests/test_p1_direct_fd_saved_audit_20261027_v1.py
```

交付区分：实际CUDA测量来自MAIN封存run；本交付新增saved-array验证及10个篡改反例；无新actual模型/物理运行，无科学准入提升。
