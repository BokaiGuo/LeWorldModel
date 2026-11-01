# 四臂预算：持久事件和独立消费

本次仅新增四臂专用 driver、consumer、测试及本说明；不修改既有运行、P1 门、
封存代码，未执行 GPU、物理、模型推理或训练。

## 主线程接线入口

在入口外部先固定完整四臂 contract、canonical SHA256、各臂源码及配置、各预算点、
独立 run ID、费用归属和上界，并独立留存提交时间。主线程新 runner 可调用：

```python
from experiments.spectral_reachability import codex_four_arm_durable_20261030 as durable

ledger = durable.Driver(new_absolute_directory, contract, contract_sha256,
                        arm, level_id, run_id)
result = ledger.dispatch(phase, kind, complete_upper_bound_vector, callback)
receipt = ledger.finish()
```

对每个臂、每个预算点分别用全新目录；`arm` 为 `structured`、`native`、
`random`、`immediate_progress` 之一。参数格式和固定轴参照已有
`codex_four_arm_full_cost_20261030.py`。绝不可把不同预算点上一次实验剩余
额度当作新预算，也不可在看到 outcome 后选曲线点。`dispatch` 在调用 callback
前持久化并 fsync intent，预留指定非负向量；如果超 cap，则先记录 blocked，
不调用 callback；callback 异常或拒绝后 fail-stop，已预留成本不退。
独立预算上界是调用者声明，不是实测。

具体插入点为 encoder/action encoder/predict/backward 的实际入口、候选筛选、
Jacobian、solve/SVD、试算 replay、共同优化器及验证/physical 的对应操作。
Jacobian 外层事件和内部 predictor/VJP 等叶子事件必须分别、非嵌套记录；
不得重复记账或漏掉计算。当前 driver 禁止嵌套及并发调用；不能简单把含内部
hook 的整个优化器包成一个 callback。setup 和所有继承成本也要按事前固定的
standalone 规则计入。`finish()` 在完整结束或捕获异常后调用以保存终态；
如果操作中途进程崩溃，可能仅留下 intent，消费器必须拒绝当完整账本。
保存 `receipt["sha256"]` 到独立可信的外部位置，不能从待校验日志自身取得锚点。

独立消费入口（符号值由主线程提供，不是现成的生产命令）：

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -m \
 experiments.spectral_reachability.codex_four_arm_consumer_20261030 \
 --journal /absolute/new-run/events.jsonl --sha256 EXTERNAL_JOURNAL_SHA256 \
 --contract-sha256 EXTERNAL_CONTRACT_SHA256 --arm ARM --level LEVEL --run-id RUN_ID
```

消费器不调用 driver 或 callback，验证外部摘要、canonical 行、连续 index、
hash chain、合同和臂/预算/run 绑定，按事件顺序独立累计计费并核对 cap、终态和
汇总。截断、缺 terminal、重排、超额 intent、伪汇总、失败后继续等均拒绝；
有效失败日志返回 `failed_or_blocked`，不表示完成。合法空日志只返回
`closed_unverified`，不代表完整执行。

## 硬边界

摘要须由独立可信渠道固定；攻击者若能改写日志并替换外部摘要，可以重写自洽历史。
consumer 使用既有 contract 校验代码，并非对该代码或 runtime 的独立来源认证。
日志未自动绑定 driver/consumer/调用闭包源码；生产前须另行外部固定并校验完整源码。
持久化/磁盘满/fsync 错误可能留下不完整日志，甚至无法写失败终态；无容量预留和
恢复保证，此时不得视作完成。成功返回只是同步文件的 OS 请求成功，不保证底层硬件
或电源失效下的数据存活。不能证明未被绕过的 hook、callback 实际资源、CUDA 异步
错误、墙钟、FLOPs、峰值显存，亦不证明预注册时间、四臂完整分母或总成本公平。
无实际 optimizer、真实运行器接线和独立覆盖证据之前，不得作为生产放行条件。

## 验证

采用 TDD，先见缺模块导致测试收集失败，再实现并扩展回归。下面命令得到
`65 passed in 1.44s`，包含本轮测试及既有合同、向量和 H15 bridge 回归：

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -m pytest -q -p no:cacheprovider \
 tests/test_codex_four_arm_durable_20261030.py \
 tests/test_codex_four_arm_full_cost_20261030.py \
 tests/test_p2_preregistered_resource_vector_20261030_codex.py \
 tests/test_p2_h15_observable_budget_bridge_20261030_codex.py
```

覆盖预留前落盘、失败不退款、超 cap 无 callback、外部身份和摘要错误、截断、
缺终态、重排、重算 hash 链后的超额/汇总/顺序/终态篡改、fsync 故障、短写、
CLI 非零退出；均为临时目录与替身 callback，不是真实资源测量。
