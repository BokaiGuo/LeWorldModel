# Fixed164 raw-coordinate terminal RGB HOST CUDA bridge

仅新增本stem源/tests/innerdoc，不改封存模块、计划、结果或MAIN文件。worker仅运行RAM测试；真实HOST CUDA仅MAIN审读后执行。测试通过不证明P1/freshness或物理收益。

## 真实接口和权限

对接已落盘 `agent_observed_raw_coordinate_physical_20261028_v1` 的 `consume(output, report_sha)`，其 own-schema `.prepare/.report/.consume`、`validate_execution`、`audit_guard`、`prefix_match`，不伪造旧suffix4 schema。准备层完整调用上游consume，固定外部receipt文件SHA，并对整份解析结果按physical.raw进行全字段重建相等验证；不只检查status。再核对physical执行绑定/prepare/source，FD已付费原模型、normalizer及所有来源SHA。

物理execution-plan selection/ref/guard使用physical.digest；FD job使用physical.producer_digest。adapter不重新用自己的codec认证跨schema字段。四个baseline索引固定0/41/82/123，继承上游严格认证的原始suffix iterate0 full10，不是observed重优化赢家；所有164项保留，无依据效果筛选。现有两known roots、非fresh。

每行原始task/identity-Q/goal192、observed context/raw history、full10 float32 SHA、producer job、native root/goal/goalpose、condition、cp5 guard和cp10 PNG/native7/时间绑定保留在prepare/report。consume输出全164 actual192/actual_Q/null、PNG来源与物理指标；不推断未提供的perturbed prediction。physical/compile费用单独继承，不能将消费0physics说成历史从未执行物理。

## 原生编码与严格消费

复用冻结observed_suffix_terminal_rgb的bridge.Trace、encode_guard、precision守卫、原Array/Store codec和独立PNG预处理，仅有本schema自己的prepare/run/consume；无moduleglobals monkeypatch。模型eval/frozen、CUDA-only、MATH SDP、禁matmul/cudnn TF32、禁autocast/grad；完整installed class来源和模型tensor hash前后检查。

164行固定顺序逐单图，原PNG→preprocessed→encoder完整257hidden→原CLS→projector192各原精度数组即时保存，模块attempt/return、guard及image事件全序保存；消费重新解码原PNG并逐值/逐bit验证预处理和所有连接数组。完整目录census/SHA/模型未变/native seal/初始roster/未知事件拒绝。冻结Ledger参数表示四图组，故实例参数41覆盖164图；不是沿用旧四图cap。

真native early-stop缺cp10时保留null164分母，减少实际调用，不填旧latent、不补图、不padding。工程失败或不完整physical.consume不准入本编码。已知全可用时164encoder+164projector；始终0predict/AD/新physics/训练/Q拟合。

工程预算：512MiB总输出（含128MiB最终保留），200MiB磁盘floor，1800s包含准备verify；开始创建输出要求完整cap+floor可用。每次实际写入计账，不反复遍历目录。工程cap不是科学门槛。原生数组gzip NPY不含pickle。输出目录/metadata排他新建；失败证据保留，不重试、不覆盖。

## MAIN命令和未决前置

全部路径为实际绝对路径。下列大写参数必须由MAIN填入实际完成physical的值，代码没有未来SHA默认值；不得使用partial或伪造receipt。生产前置仍是Nash/Pasteur封存后的真实prepare/report/完整receipt和对应sourceSHA，以及原冻结模型/安装来源闭包。worker没有执行任何production prepare/run/consume。

从 `/home/gbk/code/LeWorldModel/le-wm-main` 执行，模块为 `experiments.spectral_reachability.agent_observed_raw_coordinate_rgb_20261028_v1`：

```sh
.venv/bin/python -B -m experiments.spectral_reachability.agent_observed_raw_coordinate_rgb_20261028_v1 prepare \
  --physical-output "$RAW164_PHYSICAL_OUTPUT" --physical-report-sha "$RAW164_PHYSICAL_REPORT_SHA" \
  --physical-prepared "$RAW164_PHYSICAL_PREPARED" --physical-prepare-sha "$RAW164_PHYSICAL_PREPARE_SHA" \
  --physical-receipt "$RAW164_PHYSICAL_RECEIPT" --physical-receipt-sha "$RAW164_PHYSICAL_RECEIPT_SHA" \
  --physical-source-sha "$RAW164_PHYSICAL_SOURCE_SHA" --output "$RAW164_RGB_PREPARED"
.venv/bin/python -B -m experiments.spectral_reachability.agent_observed_raw_coordinate_rgb_20261028_v1 verify \
  --prepared "$RAW164_RGB_PREPARED" --prepare-sha "$RAW164_RGB_PREPARE_SHA"
.venv/bin/python -B -m experiments.spectral_reachability.agent_observed_raw_coordinate_rgb_20261028_v1 run \
  --prepared "$RAW164_RGB_PREPARED" --prepare-sha "$RAW164_RGB_PREPARE_SHA" \
  --output "$RAW164_RGB_OUTPUT" --authorize-host-cuda
.venv/bin/python -B -m experiments.spectral_reachability.agent_observed_raw_coordinate_rgb_20261028_v1 consume \
  --output "$RAW164_RGB_OUTPUT" --report-sha "$RAW164_RGB_REPORT_SHA" --receipt "$RAW164_RGB_RECEIPT"
```

prepare/receipt路径须位于inner results且文件名以本NAME加下划线开头；CUDA输出目录也同约束，均须不存在。prepare/verify是完整saved重建，可能有显著I/O费用，不启动模型/环境。MAIN审读批准实际run不等于科学准入。

原生核心RAM测试：

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_observed_raw_coordinate_rgb_20261028_v1.py
```

覆盖：全164单图双模块与全事件、null分母、197拒绝、原独立预处理/CLS/latent变异、guard/未知事件/资源数、真实上游schema的原baseline/control行级join、两种digest拒混、cp5与terminal状态、整份receipt重建差异拒绝、prepare重建、持久消费census/字节/封口/模型/gate/时间变异、排他写入与真实字节cap、冷import禁torch/physics/foreign。RAM夹具不是164真实物理证据；未放宽生产验证。
