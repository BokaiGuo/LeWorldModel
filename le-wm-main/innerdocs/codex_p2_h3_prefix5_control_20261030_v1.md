# P2 H3 joint 前缀与零后缀探索控制

在已看过原八槽和零动作结果之后、此次执行之前固定：每个已知根复用该根重算 joint 的前五个原始控制，后十个控制改为 `[0, 0]`。比较对象是原已保存 joint 15 步；每臂同根、同目标、同初始化、同 15 个选定物理步。这里零是接口输入值，不额外声称其对所有环境状态是物理静止。

保存后复验确认四根初始上下文一致、前五步观测逐项一致，15 步都完成，无早停；原八槽及新四槽的保存消费者、命令和预算日志核验通过。控制选定 step 60 次、warmup 60 次，root collection step 0 次，reset 36 次，均调用并返回；内部模拟子步及 reset 内部成本未计量。

| 根 | 前缀控制 − 原 joint 位置 | 角度 rad | pose |
|---|---:|---:|---:|
| 0 | -1.374593 | -0.038596 | -2.146508 |
| 1 | -16.491569 | +0.025482 | -15.981937 |
| 2 | +0.513027 | -0.023626 | +0.040516 |
| 3 | 0 | 0 | 0 |

pose 为 `xy + 20*angle`，不是原生成功；该指标控制较优 2 根、较差 1 根、持平 1 根。两组 8/8 原生成功标志均为 false。指标只描述此固定的后缀干预，不能归因于某个模型机制或推断新根上的效果。

控制臂继承相同已计算的 joint 规划，而后处理动作不同；本次无需新神经调用，不代表两次历史实验或其他方法全成本相等。原 joint 是 CUDA chart 重算动作，原规划 raw 位未保存，不可说逐位复现原动作。选择设计时已知道原结果，不是预注册，也不是独立确认；P2 科学门不变。

产物：`results/codex_p2_h3_prefix5_control_20261030_v1_PREP01.json` SHA256 `4a409129986ce4334880205973d43ea217459cfe648927d63af4e7392599e192`；`results/codex_p2_h3_prefix5_control_20261030_v1_RUN01/report.json` SHA256 `3bbff3a319b512bf6b777692baa51fb197202011e181a943a7cce99230be848a`；`results/codex_p2_h3_prefix5_pair_20261030_v1_CPU01.json` SHA256 `012e992db6446deda9cec462116719d55fd7635560413bb119d4816518718d7c`。绑定记录源码哈希，失败和物理日志保存在独占运行目录。

复验测试（`le-wm-main` 下）：

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_codex_p2_h3_prefix5_control_20261030_v1.py tests/test_codex_p2_h3_prefix5_pair_20261030_v1.py
```
