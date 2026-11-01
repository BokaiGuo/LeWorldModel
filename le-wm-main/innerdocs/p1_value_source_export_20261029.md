# P1 固定来源只读 exporter

`fixed40` 调用既有 pair-margin 的完整 `build()` 和 `verify()`，重建并比对固定 SHA 的整个分析，再从已闭合的原 selector 中提取 raw float32 动作，逐槽核对动作 SHA 和原 plan。40 槽的 terminal 预测 MSE 和原 physical metrics 分列；实际编码 MSE 的原缺失槽保留 missing。固定来源未认证逐候选 prefix 即时评分与条件 Vreach，两个字段明确 missing。terminal 预测不是 Vreach，也不是物理评分。

`joint-prepare` 调用既有 prepare `verify()`，包括重建原 mixedmode winner authority。它只导出四个经认证的原 winner 身份和动作；prepare 不是 joint 运行结果，四类评分均 missing。它不与 fixed40 按槽号连接。

`joint-arm` 接受 SHA 指定的完整保存 bundle（键为 slot、arm、result、scalar_receipt、events），先完整验证 prepare，直接以 prepare 中的 request 调用已有单臂 `audit()`。仅通过后导出 20 候选的 stage_Q、terminal_Q 和 prefix 条件 own_reach_value；physical 保持 missing。单臂 audit 校验内部 native 事件和算法重放，但不认证 journal 的原始来源、模型精度闭合或 physical，bundle SHA 仅固定用户提供的字节，不能当成来源认证。此仓库目前未找到可用的已完成 joint arm，因此该路径没有真实生产验证。

调用均只读、输出到 stdout，不写结果文件；两个输入 SHA 固定于源代码。调用可耗时，因为原 strict consumer 会读取并检查完整 closure；不执行模型、训练或物理：

```bash
.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_value_source_export_20261029.py
.venv/bin/python -B -m experiments.spectral_reachability.p1_value_source_export_20261029 fixed40
.venv/bin/python -B -m experiments.spectral_reachability.p1_value_source_export_20261029 joint-prepare
.venv/bin/python -B -m experiments.spectral_reachability.p1_value_source_export_20261029 joint-arm --bundle /absolute/path.json --bundle-sha SHA256
```

该导出是带来源限定的结构化数据，不伪造原始 receipt，也不调用早期四 receipt 拼接器；各列的时间与目标函数必须按各自原合同理解。无科学 gate、阈值、候选选择或物理释放变更。

本轮只读实测：固定40完整重建及既有报告逐字节验证通过，导出40个预测终态、40个原物理返回、39个实际终态编码评分；joint prepare 完整重建通过，导出4个原 winner，joint Vreach 全部标记缺失。测试包含通过真实单臂 audit 的 RAM native-event fixture 及结果篡改拒绝；fixture 不是模型、CUDA 或物理证据。
