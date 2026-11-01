# P4 预定四槽：物理采集前置与授权核查

本次只读源码、固定状态和包可发现性，另加独立 RAM 测试；未构造原生环境、reset、render、执行物理、访问四个保留根或使用 GPU。没有新建 sourcepack、prepare 或历史证书。

## 结论

四槽协议固定，现有 producer 能在预期执行路径上记录真实 state7、导出 body snapshot 与 observed0/5/10 的 RGB3 身份，但当前**不能安全启动这些槽位的采集**：`fresh_candidate` 需要满足完整历史证书与外部 MAIN approval；P4 v2 的封存交付说明明确禁止在四个保留槽使用 `calibration_only`。现有 blocked 证书快照无历史证书和 source run 授权。仅有用户对本轮 physical-only 核查的请求，不应被解释为改写该四槽保留协议、创建缺失证书或解除来源门槛。

源码中 `build('calibration_only')` 技术上可构造四槽计划，而文档禁止对保留槽实际执行该模式；CLI 的 `--authorize-main-physics` 是调用者提供的显式标志，不是独立认证或证据。不能把源码允许调用或 RAM 假环境测试看成授权。若要绕开 fresh 历史证书，仅做非 fresh 的身份采集，必须由用户/协议责任方**明确批准对这四个保留槽进行非 fresh 访问并事前封存其独立运行范围和记录规则**；这会改变现有访问边界，须先决定，不在本交付中默许。

## 实际身份链与限制

1. 每槽固定 root/reset seed 和单次 proposal，不替换；执行顺序 construction、replay0、replay1。每个 phase 先按 reset_seed reset，再按 seed0 + 显式请求 state 放置；seed0 不是新的随机根。
2. 每次 observation 在 PNG 与 `fresh_observation` 事件中记录 state、导出 snapshot、像素 SHA；guard 通过后，step 0/5/10 用于 RGB3，step10 的 state7 和 snapshot 构成 root identity。snapshot 覆盖两个 body 的位置、速度、角度、力等导出字段，**不涵盖物理求解器隐藏缓存**。像素 SHA 是像素数组 SHA，不是 PNG 文件 SHA。
3. 每步实际 latest_action 与预先规定的 float32 warmup 做严格比较；v2 的 `root_identity` 对固定 warmup bytes 求 hash。实际动作逐步检查后才进入 step10 gate，identity hash 本身不是独立的原始动作文件或历史来源证明。
4. construction 的 gate 在 witness dispatch 前比较已批准历史与前面槽的实际身份；碰撞被记录并拒绝，不补槽。replay 的身份需与 construction 相等。当前实现继续执行 witness 与两次 replay，并非纯粹“只捕获四个预 witness identity”的独立程序；失败槽位可拒绝，工程异常则保留未开始槽位，不保证四槽均实际成功取得身份。
5. 现有身份/排重索引不能证明未选、失败及未保存旧 reset 的完整范围。未来源即使成功取得 state7/snapshot/RGB3，也不能由此声称 global freshness、P1/P4 或历史闭包。

## 本机预检及测试

包发现检查：`numpy`、`gymnasium`、`stable_worldmodel`、`pymunk`、`pygame`、`PIL` 均可由项目虚拟环境找到；此检查没有 import 原生环境、初始化 SDL 或实际 render，因此不证明物理或渲染运行成功。工作区所在文件系统当时 `df -h` 可用约 30G；P4 v2 计划要求 16MiB 输出预算加 200MiB 空间底线。执行时仍须重新检查。物理 adapter 使用 `live.native_factory` 中的 PushT metered factory；该模块另有 CUDA 路径，但此次未调用它们。

项目冻结的 v2 RAM fake-native tests 以 `CUDA_VISIBLE_DEVICES=''` 执行 **45 passed**；新增 `tests/test_codex_p4_physical_preconditions_20261029.py` **5 passed**，检查固定四槽、缺历史证书拒绝、文档/代码模式区别、synthetic state/snapshot/RGB3 hash 与无授权拒绝。fake-native 与合成测试不是实际原生渲染或采集验证。

复现测试：

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/agent_p4_fresh_source_20261028_v2_tests.py tests/test_codex_p4_physical_preconditions_20261029.py
```

未创建新的可运行采集器：现有模式和历史 certificate 限制未获解除，额外写一个可以直接打开保留根的程序会绕过约束。若获得明确的协议变更/访问授权，需先单独固定四槽与完整失败分母、预 witness 停止位置、独占输出和日志、资源上限及非 fresh 标签，并实际检查原生 renderer，再进行授权范围内的采集；不得改写旧 source、历史结论或原协议。
