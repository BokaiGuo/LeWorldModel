# E1 保存 receipt 到资源账接口

本工具只读五个写死路径、写死 SHA256 的既有 JSON，输出 stdout；不运行模型、
物理或生产，不写或修改旧文件。六槽和全部事件维度无自动补零、无排名。

## 实际接线

- `six_slot_common_prepare`（SHA `cd96f6e28b25d3b7e3b6649c760b2f111747aec5c0ce9c8a41741318209df8a0`）
  提供六个 root/arm、task、blind、target；程序独立解码 target bytes 并复核 SHA。
- v2 prepare（`df548a6a9d8356f654e6aac823e7dd90383570da56631c38155b539bad8ae0dc`）
  引用上述 prepare 和目标 seal；v2 report
  （`cf3cb2a506ec1739ca53f58faa00bb07972c6e546cd75e02e0c609dccd7b7578`）
  引用 v2 prepare；v2 保存 receipt
  （`1ca8e949cc4b219b9b03adb3d34ee859e924e0078f9f81c519be2c03ed77be3d`）
  引用 report。程序校验六槽顺序及 receipt/report 的 selection 身份。
- Hi 保存 receipt
  （`40f28414682c9540e9e195b361a36dfc514f816358adbeacb379fedc460e4ef0`）
  的两个 off 行与 learned-FIRST 的 blind 和 target data SHA 对齐；其资源是四槽
  合计，只记 `shared_unallocated`，不分配给两个 off 槽。

六槽 receipt 逐槽给出 controller 和 replay 的四个 module hook 及 predictor sample
的 attempted/completed；本工具把能一一对应的轴作为 `receipt_counts_not_event_sets`
单独展示，并另列 predictor module calls，避免和 samples 混同。实际每槽 controller
144、replay 4；各轴嵌套，不能加总为 FLOPs。

## 为什么 ledger 仍不完整

既有接口每个非空计量要求真实的 `event_set_sha256`。所接 receipt 有计数，但此桥
未找到并验证相应逐轴事件集的内容身份，故**所有 ledger measurement 均为 null**，
不将 receipt SHA 或自行构造的字符串伪称事件集 SHA。controller/replay 阶段的
receipt 和 per-arm scope 仍有绑定；proposal 及其他继承成本不因此消失。该桥把六个
ledger 分别送入既有 `inspect`，均返回 incomplete。ledger `scope_sha256` 绑定
v2 prepare bytes，代表本工具的审计范围引用，不是外部认证的完整成本范围。

Hi 的四槽共享账单独展示 encoder、high cost、high predictor、selected high rollout
和 low predictor，包含 attempted、returned、input rows；它不是逐臂账，也不是
完整祖先成本。其他 proposal、原始目标生成、资源维度和相关事件集仍未知。FLOPs、
信息公平、账的完整覆盖和目标/资源全链真实性均未证明；不同中间目标不能混排。

此工具只复核有限身份、目标 bytes、计数格式和已保存链条引用，继承上游 receipt
的 fullstrict 结论；不重新执行 fullstrict、不读取事件 journal 或证明每个摘要是
事件全集。SHA 证明读取的字节与固定值一致，不单独证明其来源权威。保存 receipt
不等于新实验；不改变原科学门。

## 验证

```sh
cd /home/gbk/code/LeWorldModel/le-wm-main
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_codex_e1_saved_ledger_bridge_20261030_v1.py tests/test_codex_e1_end_to_end_ledger_interface_20261030_v1.py
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -m experiments.spectral_reachability.codex_e1_saved_ledger_bridge_20261030_v1
```

30 项通过，包含真实固定 receipt、目标/selection/链接篡改、坏计数和 SHA 拒绝。
真实只读运行输出 6 个 incomplete ledger；equal-budget、ranking 和科学准入均为 false。
