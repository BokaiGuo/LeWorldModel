# E1 封存事件到端到端 ledger 的只读消费

状态：六槽均为 `incomplete`；无等预算、排名或科学准入。没有运行生产、模型、GPU
或物理，不打开 P1 文件，不跟随保存文件中任何路径。

入口读取固定的 E1 six-target prepare、v2 prepare、v2 report、v2 receipt 及 v2
压缩 journal，核对代码内固定 SHA。检查 v2 prepare 到 six-target prepare、report
到 v2 prepare 的引用；对每个 target 解码并重算 data SHA。通过现有独立事件审计器
复核 journal SHA 和链、attempt/return 配对、每槽阶段计数与 receipt。六槽身份与
target prepare 按固定 root/arm 顺序连接；这继承上游 receipt 的完整消费结论，
不重新执行上游 fullstrict。

对每槽构造既有 `codex.e1.end_to_end_ledger.v1` 输入：只在 controller 和 replay
阶段填入 journal 实证的 `predict_api_calls`、`predictor_samples`、
`action_encoder_calls`、`pred_proj_calls` 四轴，采用派生的 event-set SHA 和
保存 receipt SHA；其余轴为 null。每槽 controller 四轴各 144/144、replay 各
4/4。module predictor call 不属于原 ledger 轴，未硬塞入 samples。各轴可以嵌套，
且 samples 和 predictor hook 引用同一批事件；摘要不同不能证明不重叠。

每槽 proposal 与 inherited 阶段 scope 为 `unknown`、receipt 为 null、所有测量为
null；历史共享费用没有分摊或归零。其余 encoder/native 等未证轴同样未知。
每个 ledger 均实际调用既有 `inspect`，其 blockers 包含 proposal 与 inherited
receipt 缺失。此次没有读取 Hi receipt 或重建旧费用，也没有声称相关成本不存在。

日志摘要只绑定固定保存日志中的记录，不证明 runtime 未漏记、日志外历史费用完整、
实际 FLOPs 或信息公平。输入 target 与事件关联依赖既有 prepare/report/receipt
和先前消费，不在此读取模型、数组或重建目标运算；`scope_sha256` 只是 v2
prepare bytes 的范围引用，不是全成本闭合证书。无比较、无排名，不改变科学门槛。
内部 `consume` 函数用于组合已校验对象；只有 CLI 固定来源入口先重算日志并验证
固定 SHA，不能把未经核验的自造对象传给该函数当成真实证据。

```sh
cd /home/gbk/code/LeWorldModel/le-wm-main
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -m experiments.spectral_reachability.codex_e1_event_ledger_consumer_20261030_v1
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_codex_e1_event_ledger_consumer_20261030_v1.py tests/test_codex_e1_sealed_event_axis_audit_20261030_v1.py tests/test_codex_e1_end_to_end_ledger_interface_20261030_v1.py
```

37 项通过，含真实来源、固定摘要、六槽 ledger 和 proposal/继承缺口、目标/槽/
链篡改及坏计数。测试不写既有结果文件。
