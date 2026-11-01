# 固定 moving2 基线结果 assembler v1

实现任务，非代码审计任务。MAIN 负责实际 saved 综合、审查与科学准入；本 worker 仅新 source/test/innerdoc，RAM fixtures，不运行实际综合、prepare、物理、模型或训练。kino 原三文件保持冻结不变。个人 goal 无 token 预算，完成仅指本实现交付，不是 P1 完成。

## 冻结输入与输出合同

固定算法顺序：`line_search_GN`、`full10_conditional_D2P`、`CEM`、`kernel_faithful_mppi_nominal`、`moving_gradient_open_loop`、`fixed_H2_kinodynamic_beam16x4`；每算法 slot0、1 共12行。不是独立确认，不将本地算法称官方复现，不称等搜索域/总算力。sampling 允许两臂一次 invocation，或两个单臂 invocation，但不允许同算法重复出现覆盖。

只接受原 moving encoder report（SHA `e7fb554327b3bb5a96190f6291f8450b2b0723cd981817715cfb7e50aa266bf6`）及原两 task SHA：`a42bfbb1ea692782355e2572a5df2cd048b72cf3e6c41fee40eb12503674959b`、`101add21a7a40591167a6ab8e8b22c3c62cb1fd700c741fd36cd2073e2d5df74`。任务 ID 固定为 `bf89d0d3c5738763c40a2d6481c6782df218de398d48bd7df33543d3444d21fd`、`6643e3cbd4322303fdd8eb158532d75e1c0f2ce720d13a8258d12a9527801b12`。通过冻结 moving public loader 读取非零 history，再核 raw10 和实际 float32 normalized actions2，拒绝旧 zero 任务、换根、target-on。

manifest 自有 schema、外部 SHA。每 source 明确 `complete / partial / missing`，不扫描结果目录猜最新/最好一次。complete 同时绑定 actual report、ownconsume JSON 的绝对路径及外部 SHA；调用被冻结的**saved-data own consume**，与其 JSON 全内容一致后入表（不调用 producer.run/prepare/神经/物理）。这是 assembler 的输入接线，不另开审计支线。partial 仅展示原 report 中未认证完成的 attempt/return counters 和原因，**不**发布其 modelQ、commands或当完整预算。missing 缺数据保留 `value:null` 和原因，绝不转成0。

v1 实现时 Arendt `p1_moving_baseline_physical_20261027_v1`、Jason `p1_moving_baseline_rgb_cuda_20261027_v1` 未有可读取的冻结接口。因此 **不猜字段，不接物理/RGB实际值**。`downstream.physical/rgb` 只允许 `missing / schema_unfrozen` 与 reason，若传 actual 自动拒绝。所有12行 native_success、realQ root/prefix5/terminal10 均显式 missing，物理 expected denominator=12、actual availability=0；这是覆盖数，不是0成功。后续明确 schema 后须 NEW版本接入，不能静默扩展本冻结 v1。

输出 NEW目录两个文件：严格JSON `comparison.json` 与简洁中文 `comparison.md`，合计≤1MiB，保留200MiB floor；不覆盖已有目录。JSON 包含每根、每算法两根summary、run-level来源、完整 algorithm contract、源码SHA/manifestSHA、各缺失原因。只有两根模型都 actual 才计算该算法平均 modelQ；不在缺一根时缩小分母。模型Q明确 native identity192 H2 predicted terminal MSE，不是 native success，也不是实际编码 realQ。非kino从 ownconsume prediction 与原 goal 数组重算；kino ownconsume 本身独立重建80候选及3路径，其 terminal_cost作为已经重算的模型值。没有从物理/terminal观测选择模型、根或候选。

## 资源与可比性

actual module predict/action_encoder attempt/completed 从报告行保留；每调用 batch1，forward_samples=原 predict，不加 AD forward第二次。成功 ownconsume 验证的合同供 ADforward 子集及数学反向方向数；与 meter 区分 basis。保留原 `resources / AD_API_resources / predict_API_resources`，不把嵌套Jacobian API和grad API当独立增加的反向工作。

- GN 每根78 forward（含8 AD forwards）；4手工Jacobian操作、768 grad API/方向，但 functional.jacobian API=0。未单独meter的通用 backward 字段保持 unmeasured。
- D2P 每根66 forward（含11 AD forwards）；11 Jacobian API及11向量化 grad API不应相加为22个额外Jacobian；数学VJP方向2112，SVD12。
- gradient 每根136 forward（含64 AD forwards）；32 backward和32 scalar grad API是同32次reverse sweep，不是64。Jacobian/SVD为显式合同0。
- CEM 每根134 forward，MPPI nominal72，kino84，包含各自真实重放；sampling/kino的无AD/无Jacobian/SVD零值标为冻结算法合同，非另有runtime计数hook。kino pred_projector另meter；其它算法未单独meter该项则 unmeasured，不伪填0。

完整原合同保留采样/优化预算、目标、动作域、选择规则、信息来源、重放、种子等。kino constant packet离散库不同于D2P逐时刻logit-ball；MPPI nominal不冒称官方强MPPI，不能用forward比较宣称总算力、FLOPs、全库自适应优势或公平排名。

原 moving encoder 的 actual8 encoder +8 projector在比较共享总账仅记一次。每算法 standalone 两根执行视图各列8/8，是反事实单独运行的成本视图，不把六个standalone条目相加为共享成本。未测的编码 latency 不填0。

latency仅采纳明确 `report.limits.elapsed_seconds`，保留**整个 invocation guard范围**（包含其全部根/臂及加载等），不用神经调用数估计，不把operation walltime拼为整次walltime；每根/每臂未独立meter保持 unmeasured。一个 source_id 对应一条run-level latency，不复制累计或除2/4均摊。null仍未测，不变0。

## MAIN manifest 与 CLI

Python根 `/home/gbk/code/LeWorldModel/le-wm-main`。模板输出stdout（本 worker不落盘）；MAIN用apply_patch创建NEWmanifest，填真实SHA，不用重定向伪造已封存内容：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_moving_baseline_comparison_20261027_v1 template
```

例如将模板 moving source 条目换为以下**明确占位示例**，其它provider尚未执行可保持missing；sampling的 arms 可为单臂或原顺序双臂。consume 必须是原 ownconsume 的原样严格JSON，没有额外wrapper键，不把CLI日志混入JSON。

```json
{
  "provider": "moving",
  "state": "complete",
  "arms": ["line_search_GN", "full10_conditional_D2P"],
  "report": {
    "path": "/home/gbk/code/LeWorldModel/le-wm-main/results/moving_history_planners_cuda_20261027_v1_cuda_main01/report.json",
    "sha256": "cd8432c06288577d25d9d9b92c7eb3673de592499ee64cae93238156f3f19aa0"
  },
  "consume": {
    "path": "<MAIN实际保存的绝对ownconsume JSON路径>",
    "sha256": "<MAIN实际64hex SHA>"
  }
}
```

partial source 没有consume字段，含 `provider,state,arms,report,reason`，仅展示未放行记录。missing source 含 `provider,state,arms,reason`。报告 schema必须对应本 moving provider，不接受旧zero/其它实验的报表。

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_moving_baseline_comparison_20261027_v1.py
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_moving_baseline_comparison_20261027_v1 assemble --manifest /home/gbk/code/LeWorldModel/le-wm-main/results/p1_moving_baseline_comparison_20261027_v1_manifest_main01.json --manifest-sha <MAIN_MANIFEST_SHA> --output /home/gbk/code/LeWorldModel/le-wm-main/results/p1_moving_baseline_comparison_20261027_v1_main01
```

所有 SHA 与源合同失配均 fail closed；MAIN若选partial仍不得将未完成成本或模型输出作为成功结果。下游物理/RGB冻结协议接线、独立根、强基线公平性和科学准入仍未解除。

实现交付验证：**41 passed**（RAM fixtures，无tmp/result/实际综合写入），覆盖完整12行、缺失/partial、历史/目标防混、ownconsume接线、AD重复计账反例、共享/standalone账、latency缺失/不均摊、输出cap/floor在写入前拒绝。`--help`正常。未执行production saved综合或任何模型/物理；source/test/doc最终SHA见交付消息。
