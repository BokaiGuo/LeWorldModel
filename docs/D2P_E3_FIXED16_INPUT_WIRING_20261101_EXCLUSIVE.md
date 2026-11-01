# E3 固定 16 格：保存记录检查与未覆盖输入接线

## 范围

本次仅在 CPU 上检查既有保存结果并建立固定输入接线；没有运行 GPU、
神经推理或物理环境。16 格为 known2 × goal2 × horizon2 × domain2。
旧 15 格扫描按文件名限定且排除 03；10、11 的 cooperative 目录也
不在其匹配范围。因此下表结合此次扫描与既有单格报告，后者是既有
记录的结果，不是本次重新执行物理实验。

| 格 | 既有保存记录或本次接线 |
|---|---|
| 00、01、08、09 | 本次有界扫描记录 completed20 与边界 10、15；先前 CPU 身份联结报告覆盖四格 |
| 02 | 有 control5 数值回执和已有输入 authority；本次有界扫描未发现在线作业 |
| 03 | 先前报告记录 completed20 与分离式物理、编码、规划重放联结；本次 15 格扫描不检查此格 |
| 04、12、13 | 本次有界扫描记录 native_stopped_prefix，未有 10、15 反馈节点 |
| 05、06、07、14、15 | 本次有界扫描未发现在线作业；新增固定输入 authority，均通过保存态 setup 验证 |
| 10、11 | 先前单格保存消费者报告 completed20 与边界 10、15；不在此次有界目录扫描中 |

所有 15 个被扫描格的 control5 回执及其 report/completion 当前字节
联结通过；03 的情况依赖其既有报告。扫描中未发现无法归属的作业，
但命名范围之外仍可能有其他产物。数值 control5 不是物理反馈。
04、12、13 的停止也不能替换成反馈节点；12 的目标与起点重合
和原生停止原因见 `D2P_E3_CELL12_EARLY_STOP_20261031_MAIN.md`。

## 新增接线与验证

`e3_uncovered_fixed_inputs_20261101_exclusive.py` 对 05、06、07、
14、15 分别固定 roster 身份及已有 report、completion、receipt 的
SHA-256，重建原 identity authority 并继承既有来源 pins；生成后调用
原 host authority、native setup 消费者检查格、H、动作域与 cp5。
输出独占创建，不覆盖既有产物。

| 格 | 新 authority SHA-256 |
|---|---|
| 05 | `23bedf07b17e40bf03d10e276de528f126d915317b76b111d0cd41706f78340b` |
| 06 | `0add0b56b7e3f8f09864e36b2447b8df26ff776a296a4fee525f1803c714cae1` |
| 07 | `5d72b7a0f95c407bccf6edc2e523de761f22f07edd8332b6372fd1cf31e70fff` |
| 14 | `7ad242cfcaa49150091e4aeaa66a64111c3d8dbe42d78750fbde80f2dcc68da0` |
| 15 | `304276bbf13684d293c84215abdaf29f5a332e64ab801da5ecfec1c6b4e84aaf` |

文件位于 `le-wm-main/results/e3_uncovered_fixed_inputs_20261101_exclusive_cellXX_RUN01.json`。
五格均返回 `native_saved_setup_resolved=true`；8 项测试通过，
覆盖五格实际保存来源重建、越界格、输出路径限制及来源哈希漂移。
来源哈希是本次固定的本地保存字节，不是外部独立历史认证。

复验测试（`le-wm-main` 目录）：

```sh
env CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -B -m pytest -q -p no:cacheprovider \
  tests/test_e3_uncovered_fixed_inputs_20261101_exclusive.py
```

这五个 authority 仅是后续既有 runner 的固定输入，不是启动授权、
新物理执行、反馈闭环完成或科学准入。现有 03 的分离式重放也不
代表其余格得到同等验证；全 16 格完整 strict、跨任务公平对照及
独立泛化仍未证明。
