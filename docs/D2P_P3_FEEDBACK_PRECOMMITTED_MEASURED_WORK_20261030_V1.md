# P3 真实反馈：事前固定的已计量工作对照与保存验证

## 状态和研究对象

这是根据 RUN_MAIN01 已知结果制定的**未来运行方案**，不是对 RUN_MAIN01 倒签预注册。对象是两个已知 PushT 仿真根上的固定顺序与信号顺序，比较已计量的有限 H1/H3 工作；不是完整 P3 动态宽度、深度或 R 策略。`fixed_order` 先 H1 后 H3；`signal_order` 第一次沿原 signal 决策，在支持 continue/progress 时选择 H1 或 H3，第二块强制剩余 horizon。两者都支付完整 signal 和两块工作，fixed 是匹配工作对照，不是最低成本的自然 fixed baseline。第二块使用已执行动作后的观察编码上下文；H3 内每 5 步也重新编码。

沿用四槽固定顺序 root0/fixed、root0/signal、root1/fixed、root1/signal 和既有宿主 runner 的失败处理：单槽失败或原生停止后不补动作，但 lifecycle 会继续尝试后续固定槽；不删槽、不重试、不 padding。当前既有工程 prepare 被固定到 SHA `b6c8dcf650d4e752f6306ed4b6f9434861c224ced51c7c8de25c0e7ca6d323f2`，含上游任务、模型、源码、精度和运行资源合同；新合同不是独立 GPU 运行授权。

## 事前合同

独占冻结文件 `le-wm-main/results/p3_feedback_budget_contract_20261030_v1_FREEZE01.json` 的 SHA256 为 `5781d953d05477e633659b3e9bb58d630ea59447720964af4e8678829f565b4f`。它绑定原 prepare、四槽、指标、验收与新验证器自身 SHA。对未来运行，须在启动前将该 SHA 和完整文件放入有可信时间戳的独立不可变记录，记录研究负责人批准；仅本地文件、mtime 或参数字符串不能证明先于运行。验证器始终将 `pre_run_external_timestamp_verified` 留为 false，需独立审查该外部记录；本任务未创建或核验外部记录。

每个完整臂固定付两个 parent 的 H1/H3 signal、每个选定 horizon 的 endpoints 四候选 whole/segmented replay 及独立 winner replay；总路径 H1+H3、20 计划步。27 轴完整 attempted 和 completed 都须等于合同向量（包括 52 predict/predictor、156 context tokens、4 Jacobian API、768 VJP directions、18 replay、4 signal compile、2 program compile、2 schedule）；新观测编码在单独 hook 核四次，不能把数值 ledger 里的零误说成没有编码。不同轴有嵌套，不加总成“总算力”；物理执行与外部编码单独计数。

主终点预先指定计划第 20 步 native7 distance 的 signal−fixed；辅助为 success、joint XY、object XY、速度误差及 0/5/10/15/20 各步 native7 distance。固定全部分母，同根 cp0 snapshot、pixel、goal 和 root hash 必须一致；原 native reward/termination 与重算结果核对。第 5 步之后，先执行 H1 的臂会在下一步前启动第二次规划，而先执行 H3 的臂尚未到第二块，**不承诺各中间控制步成本匹配**；即便完整末端计数相同也不等 wall、FLOPs、能耗、GPU kernel 或未监测工作。两个根不是 fresh 样本。

## Saved-only 验证实现

独立 `experiments/spectral_reachability/p3_feedback_budget_contract_20261030_v1.py` 只读预固定 SHA 的 prepare/contract、由外部提供 SHA 的严格 receipt、receipt 绑定的 report/completion、report 绑定的 journal。它使用已保存 chunk 重组四槽最终 ledger，核固定向量、完整观察和编码边界、逐根起点身份、原生指标和跨臂 hook `(resource, shape, directions, status)` 多重集；分别报告完整计量向量匹配和形状匹配。它**继承**既有严格 consumer 对来源、tensor IO、observed-context 和日志闭包的认证，不重复执行 full strict，也不是设备远程证明。

任何非 strict 或部分 receipt 的四槽位置和状态保留，计量匹配与任务配对结果标为 null，不将未执行或失败补零；真实部分已付账目应从原 report/journal 和 partial strict receipt 读取，本验证器不会声称已独立重建该部分账。完整 strict 但身份、目标或事前完整向量不符时 fail closed，不发表匹配结论；shape 不同则单独报告 false。所需计量缺失不可视为零。数值局部 ledger wall 保留；完整单槽 wall、CUDA ns 和 FLOPs 缺失为 null。

## 已有数据的回顾性验证

CPU-only 针对原先已消费的 RUN_MAIN01 运行新验证器，保存 `le-wm-main/results/p3_feedback_budget_contract_20261030_v1_RETRO01.json`，SHA256 `24628785a0e20346a7c56c7da10858634e729c03b6622345d0a17ebb37d559e8`。两个已知根的两臂均显示 27 轴 attempted/completed 匹配、hook shape 多重集匹配；20 步 native7 signal−fixed 分别为 +121.212 和 +58.397。它只是**既有运行的回顾性 saved 检查**，不证未来事前固定、等总算力、因果收益或泛化，`scientific_pass=false`。

## 未来执行与消费门

冻结 SHA 和该设计须先经独立时间戳和审读；确认现有 prepare 源码/模型闭包仍通过后，另行取得真实 GPU、仿真/显示和窗口协调，再按原 HOST 协议固定四槽执行、单独运行原 full strict consumer。若 prepare 漂移，当前合同不再适用，须新建版本并在新结果之前重新冻结，不能修改已封存文件。原 full strict 当前要求四槽完整；partial 不具备完整任务比较权。实际预算不匹配、运行停止、无外部先后证据均保留，不能事后重定义门槛。

拿到未来实际 receipt SHA 后，在项目根下仅运行 CPU 读取：

```sh
env CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -m experiments.spectral_reachability.p3_feedback_budget_contract_20261030_v1 inspect \
  --run /ABS/NEW_RUN --receipt /ABS/NEW_STRICT_RECEIPT.json --receipt-sha ACTUAL_RECEIPT_SHA \
  --prepared results/p3_matched_order_feedback_host_v2_codex_20261029_prepare_MAIN01.json \
  --contract results/p3_feedback_budget_contract_20261030_v1_FREEZE01.json \
  --contract-sha 5781d953d05477e633659b3e9bb58d630ea59447720964af4e8678829f565b4f \
  --output results/p3_feedback_budget_contract_20261030_v1_NEW_SAVED.json
```

输出独占创建，不覆盖。未来外部 receipt 的实际哈希不能预填；此命令不执行环境或 GPU，也不替代 full strict。

## 测试

`CUDA_VISIBLE_DEVICES='' ... pytest -q -p no:cacheprovider tests/test_p3_feedback_budget_contract_20261030_v1.py tests/test_p3_run_main01_cpu_comparison_20261030.py`：15 passed。覆盖逐轴缺失、shape 单独不匹配、根/目标/起点替换、失败未知分母、native 距离保留速度、SHA、重复 JSON 键及输出排他。fixture 测试不是实验结果，旧 RUN 的实际 saved 验证也不替代未来运行。没有启动 GPU、模型或仿真，也没有科学通过声明。
