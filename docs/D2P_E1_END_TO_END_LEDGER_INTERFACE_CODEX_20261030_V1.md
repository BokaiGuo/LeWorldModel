# E1 端到端资源账：独立结构接口

状态：声明检查接口，未接入生产、未认证账目、未提供等预算结论。旧消费者和原科学门槛
均未改动。本次只运行合成与既有消费者测试。

## 已知断点

- 同目标四槽 controller receipt 记录每槽每轴 `144+2+4`、四槽每轴 600；
  此账不能代替强制候选六槽物理实验的逐臂完整成本。
- 六槽 common controller receipt 记录每槽 controller 144 和 replay 4，但不同臂
  的 proposal/继承开销未因此相等；目标也不能跨方法未经核对就混排。
- Hi integration 的 encoder 8、high_cost 12、high_predictor 56、input rows 392、
  selected high rollout 4、low_predictor 8 是既有四槽合计，缺逐事件分摊证据时
  不能除二或填入某一臂。arbitrary 的 CPU 搜索和 compiled 的既有搜索也需单独计账。
  不同模型模块的嵌套计数不能相加当作 FLOPs；wall、storage 与调用次数是不同维度。

## 接口

`codex_e1_end_to_end_ledger_interface_20261030_v1.py` 只读标准输入的 JSON，不打开
receipt、数据、模型或结果文件，不调用生产。CLI 接收唯一参数：标准输入原始字节的
小写 SHA256。输入顶层精确字段为 `schema`（值
`codex.e1.end_to_end_ledger.v1`）、`scope_sha256`、`arm`、`task_sha256`、
`target_sha256`、`phases`。这几个 SHA 都是声明，不提供身份真实性。

`phases` 必须恰含 `proposal`、`inherited`、`controller`、`replay`、`encoder`、
`native`。各项字段精确为 `scope`、`receipt_sha256`、`measurements`：

- `scope` 为 `per_arm`、`shared_unallocated` 或 `unknown`。后两种导致 incomplete；
  接口不会分摊共享成本。receipt SHA 缺失也阻断。
- `measurements` 必须完整包含代码中 `AXES` 的 19 个固定维度。每个维度为 null
  （未知并阻断），或为精确字段 `attempted`、`completed`、`event_set_sha256`。
  非负整数且 `completed <= attempted`；失败尝试不能退款。填零仍必须附事件集及
  receipt 摘要，且工具不会将其认证为真实零成本。
- 同一轴跨阶段重复使用完全相同的 event-set 摘要被拒绝。不同摘要不证明集合不
  相交；跨轴允许相同事件，因为调用可能嵌套，工具不跨轴或跨阶段求和。

结构缺失、非法计数或重复摘要退出 2；结构合法但有未知、共享未分摊或无 receipt
退出 3；完整声明退出 0。**退出 0 不代表通过科学门、证据真实、范围完整或等预算**：
所有返回均保持 `equal_budget_certified=false`、`ranking_authorized=false`。
接口既不读取证据内容也不检查签名、事件集合覆盖、单位语义、重叠、间接成本、
账与动作目标的真实绑定或公平资源分配。必须由独立审核者针对实际代码、全事件
日志与证据链核对这些事项；不能利用声明全填零绕过这一外部缺口。

资源轴列举是最小结构合同，并非声称穷尽所有工作或允许把其加为一个总成本；例如
selected high rollout 和硬件、内存、I/O 或单位定义仍须在外部完整 ledger 中明确。
proposal 与 inherited 的划分不能通过移动事件避免费用；当前接口无能力证明未遗漏
未声明事件。未来若要比较，须另行固定方法、目标、计量单位、真实预算上限和分摊
规则及其认证方式；本接口不提供比较或排名操作。

## 测试

```sh
cd /home/gbk/code/LeWorldModel/le-wm-main
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_codex_e1_end_to_end_ledger_interface_20261030_v1.py tests/test_codex_e1_strict_saved_consumer_20261030_v1.py
```

39 项通过，包含缺 proposal/inherited、未知与共享总账、失败尝试、相同事件集、
丢轴、坏 SHA 和非有限数。合成 fixture 的数值与摘要仅测代码，不是实际资源证据。
