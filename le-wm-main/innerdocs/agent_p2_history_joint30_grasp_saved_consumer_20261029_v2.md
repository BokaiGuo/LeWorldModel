# GRASP raw15 saved consumer v2：精确恢复空Adam快照

独占新source/tests/本文；不改v1三件套、prepare、actual run或任何旧结果。experiment-code仅用于定向修复与非神经回归，不执行通用训练/正结果模板。无新模型调用、NN/AD/physics、网络或训练。MAIN审读生产；本版没有prepare/run入口。

## 实际根因

绑定原report `results/agent_p2_history_joint30_grasp_cuda_20261029_v1_cuda_MAIN01/report.json` SHA `d92e24d9807a20f296811d4321b64cc87f2c8c73c04677a350f136debc1ce14a`。只读journal首个optimizer_return为index40，event SHA `d0980df2ee1e2ceac33fec1e2e2dca6608f90aaf2655c53be6eb41d7b5d9f6aa`。

`v_before.step=0`，m/q均384元素且0非零，各185个negative-zero，SHA均 `0b9c761b27a2e0b287c3fedf401af232fba0fae3783087b1f5da0b86568c17bc`。原producer `optimizer_snapshot`在空state分支用当前`p.detach()*0`；原consumer `blank(np.zeros)`是全positive-zero，因此原字节比较失败。原生产无此优化失败，不是科学负结果；保留MAIN首轮strict失败记录。

## 最小修复，不放宽moment

v2 `initial_memory(parameter)`用原dtype、当前参数逐元素乘0构造step0 m/q，精确保留零符号和hex/SHA。初始x/v、每次fresh sync x、每次sync后joint重建x/v都使用其**当时**参数，非只修第一轮virtual。已有step>0 moment仍按原descriptor/全部字节严格继承；不全局归一化signedzero、不容忍任意小nonzero、不改任何浮点界或model-parity条件。

`validate_root`除三类fresh初始化位置外与原函数文本相同，测试逐文本验证。原native/fullH3逐边保存检查、loss/梯度chain/Adam算术、900 backward/2706predict各根完整分母、全量event census、whole/segment及SHA闭包全部原样复用。未调用/改写v1.consume，也无module globals patch。消费原v1.report/loader/prepare自有schema；新receipt/model_plan用本版schema，标注原reportSHA，未伪造v1结果。

本版读前后绑定自己source/tests/doc，另硬pin v1三件套。旧prepare全量verify仍使用原frozen代码；不为v2重制旧prepare或要求NN重跑。任何后续实际不同错误须具体定位，不自动放宽。

### 完整saved消费揭示的另外两处算术参考错误

root0/joint2/step3的float64 action moment坐标28：old_m=.0014992054751739488、grad=-.01341571158008321、actual_m=7.713769648233314e-06。原`.1`与installed `1-.9`不逐位等价，原参考为7.713769648232853e-06，触发相消后的原界。二阶moment同样使用installed `1-.999`而非`.001`。未改beta值。

随后root0/joint10/step11的float32 virtual moment坐标176：old_m=.00025037897285073996、grad=-.002266007475554943、actual_m=-1.2596873375514406e-06，原NumPy两次舍入为-1.2597010936588049e-06。实际CUDA lerp的small-weight分支采用rounded(end-self)后RN fused add；新`native_lerp`用标准库Fraction精确计算乘加并round-to-nearest-even，float32额外检查相邻值避免double-rounding。两个实际错误坐标均逐位匹配精确重建值。

只纠正参考公式，原Adam64eps moment界、128eps proposal界及所有其他浮点界完全不变（测试逐函数文本验证差异仅want_m/want_q行）。**未扩大数值容差、未给神经输出加parity tolerance。** stdlib静态读取并硬pin installed `torch/optim/adam.py` SHA7efa850aab7342ed1e8675cf0f7a77df038ab532496434b0a7f2aa6866cb3a5c（行456、475）及`ATen/native/Lerp.h` SHA74cfc5466a751ad53c847c89cdcb0f8580318bd4a574c6963109cfcef372681d（行32–34）；不导入torch。

worker已完整saved-only stdout消费两根至末尾exit0：5412predict/1800backward完整、errors=[]、两plan，未写receipt、未NN/physics。开发期第一次v2尝试曾因漏upstream/BALL失败；补齐后暴露上述算术错误，deepcopy也已补齐，并加入全部函数全局依赖静态回归。失败记录不归科学负结果、不抹除v1失败。

## CLI

cwd `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p2_history_joint30_grasp_saved_consumer_20261029_v2.py tests/test_agent_p2_history_joint30_grasp_cuda_20261029_v1.py tests/test_agent_p2_grasp_raw15_prepare_20261029_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_history_joint30_grasp_saved_consumer_20261029_v2 --output results/agent_p2_history_joint30_grasp_cuda_20261029_v1_cuda_MAIN01 --report-sha d92e24d9807a20f296811d4321b64cc87f2c8c73c04677a350f136debc1ce14a --receipt-output results/agent_p2_history_joint30_grasp_saved_consumer_20261029_v2_consume_MAIN01.json
```

省略`--receipt-output`只stdout，不落新receipt；NEW receipt独占写入、<=2MiB、200MiB floor/fsync，不覆盖。实际fixture来自原journal前41事件，不构造新研究资产。完整saved消费结果/测试数/SHA见最终交付，不将synthetic/RAM测试等同科学证据。P1/P2/freshness/physical_release仍false；预算仍非1902等资源。

实际联合回归 **102 passed in 2.14s**（新16项+旧86项），包括原signedzero失败、两项实际相消/FMA错误的journal出处、fresh/reset signs、step>0严格字节比较、原校验界文本不变、源闭包及冷进程禁NN/network。最终版本完整saved-only stdout复验及冻结三SHA见交付；不生产新NN结果或覆盖首轮strict记录。
