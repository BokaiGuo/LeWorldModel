# Moving overlay：四条件重新编码并编译八计划（MAIN待运行）

本stem仅source/tests/innerdoc三个新文件。使用experiment-code的自包含实现、RAM回归与持久记录要求；
不改旧代码/共享文档，不运行实际prepare、模型、CUDA或physics。个人实现完成不等于研究goal完成。

## 固定输入与物理语义

唯一source report：`results/p1_moving_target_overlay_source_20261027_v1_main01/report.json`，
SHA256 `85ca1df3de910628bfd6c5bd1a0a482f61880a2ee76bfd404732a77dd8c7666a`。
只消费其两slot×off/on四份真实blind NPZ；严格三个数组：RGB history3、真实raw warmup10、goal RGB。
核报告完整status/caps、四份NPZ文件SHA、各自四图pixel SHA、raw SHA。没有witness/outcome读取回调供编译器选动作。
官方native默认goal_pose outline不是构造taskgoal；只比较原producer切renderer `with_target` 后的图。
不修改goal_pose、state、目标或图像来制造“目标可见”；不把on图解释为修复了真实任务目标。

## 真实重新编译，不是旧动作重评分

每个root/condition重新独立encode observed0/5/10+goal四图，共16encoder+16projector；不跨条件复用latent。
新条件感知task_contract/task_export区分off/on，并绑定prepare/source/blind/pixels/raw/goal。
不调用只接受off的旧moving loader，也不调用zero-history blind.interface；不会改旧模块globals假冒新接口。
仅复用`history.interface`的原生数值校验，以及冻结`blind.execute_arm`、`blind.selected`、GN/D2P内核。
同条件两arm共享这一次新编码的identity192/H2 task；原float32 mean/std、真实warmup、packet cap、求解/选择阈值不变。
每个root/condition：GN78 predict+78 AE、4 Jacobian/768真实grad API；D2P66 predict+66 AE、11Jacobian/12SVD、11grad+11jacobian API。
全四条件共576 predict，不把API/module或ADforward重复相加；完整每条module/AD attempt/return持久化，失败不退款。
冻结内核自带的作用域ADTrace原样复用并恢复；本模块只安装只读forward hooks，不猴补任一旧模块global或loader。
实际run须独占进程，不同agent不并行在同一Python进程内调用内核的作用域AD API instrumentation。

## 数据与资源封存

prepare绑定本三文件、完整静态源码闭包、预处理及torch/transformers依赖、原模型/config/normalizer、source report与四NPZ；
verify完整重建，不加载模型。run HOST CUDA-only，原model一次load、eval/frozen/tensor SHA前后相等、MATH SDPA/TF32off/autocastoff。
16图保存原RGB、实际CUDA `_image_tensor`预处理、encoder完整hidden tokens、CLS→projector输入/输出、native192。
每个plan保存全78/66次AE/predictor/pred_proj实际输入输出NPY、float32数组SHA与文件SHA、全部context/AD/math/candidate/replay日志。
保存每个原result与新plan seal；只按原GN/D2P model-only规则选，不读physics结果、不挑根、不重试。
consume核文件集合/SHA、原任务重建、16图桥接、全module/AD账及全3context/action chronology与最终selected replay输出。
八计划全部成功且全局模型/源码闭合才释放；失败保8分母、已发生attempt、partial张量与原错误，不补根。
MAX wall900s含verify/load/compile/final；全部输出64MiB（留4MiB完整张量账最终报告reserve），磁盘floor200MiB。
cap/超时是工程失败，不准扩cap、减样本或冒称科学负结果。超时只允许用final reserve尽力写真实partial报告。

## Bohr稳定接口

`consume(output, report_sha) -> list`，恰好8plan，顺序固定：slot0 off GN/D2P，slot0 on GN/D2P，slot1同序。
每个plan至少包含：

`{root_slot:int, condition:off|on, arm:line_search_GN|full10_conditional_D2P, commands:rawfloat32[10,2]JSON,
selection_digest:64hex, task_id:64hex, blind_path:absoluteNPZ, blind_sha256:64hex,
source_report_sha256:64hex, planner_report_sha256:64hex}`。

report顶层含`schema/status/rows/errors/model_unchanged/prepare_sha256`，以及完整images/module账/资源/源码与artifact seals。
commands不额外/100、tanh、clip或重归一化。下游只能消费这八个已封存动作，不能读witness来改选。
不编辑Bohr文件；不宣称P1通过、目标构造修复、物理收益或独立确认。

## MAIN审读后命令

cwd `/home/gbk/code/LeWorldModel/le-wm-main`；以下prepare/run本交付均未执行：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_moving_overlay_recompile_cuda_20261027_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.p1_moving_overlay_recompile_cuda_20261027_v1 prepare --output results/p1_moving_overlay_recompile_cuda_20261027_v1_prepare_main01.json
.venv/bin/python -B -m experiments.spectral_reachability.p1_moving_overlay_recompile_cuda_20261027_v1 verify --prepared /ABS/PREPARE --prepare-sha ACTUAL_PREPARE_SHA
.venv/bin/python -B -m experiments.spectral_reachability.p1_moving_overlay_recompile_cuda_20261027_v1 run --prepared /ABS/PREPARE --prepare-sha ACTUAL_PREPARE_SHA --output results/p1_moving_overlay_recompile_cuda_20261027_v1_cuda_main01
.venv/bin/python -B -m experiments.spectral_reachability.p1_moving_overlay_recompile_cuda_20261027_v1 consume --output results/p1_moving_overlay_recompile_cuda_20261027_v1_cuda_main01 --report-sha ACTUAL_PLANNER_REPORT_SHA
```

RAM测试中的encoder/preprocessing/module/AD均明确synthetic；只在人工数组后端运行原GN/D2P数学与metadata wiring。
这些测试不验证CUDA可运行性、真实预算/墙钟是否足够或科学效果；真实pipeline由MAIN审读和运行。

## 首次运行前纠正：patch14实际257 tokens

MAIN实审指出未封存初稿consume错误硬编码197，人工mock也照搬197。这是代码/mock接口漏测，不是理论或科学负结果。
已只读核冻结 `config.json`（SHA `2564086e961e7b5c7c04dffc451091115b389a590645ff19653c64fd0bc16e09`）：
encoder image_size224、patch_size14，projector input_dim192。必须为 `(224/14)^2+CLS=257`。
另核原已保存 `p1_moving_target_overlay_fidelity_cuda_20261027_v1_factual_cuda_main02/report.json` 的
module_000 encoder last_hidden_state receipt为 `[1,257,192]`，文件
`module_000.output.last_hidden_state.npy.gz` SHA `5363644ac0555f03e8916ee8227d2b2d39a5c9bb15f53799f3f9cab609573911`。
本receipt仅用于核接口，没有重编码或作为本次新实验结果。
新prepare从固定checkpoint config推导encoder_output_shape，并严格要求224/14/192及257；run hook和consume都按此核验。
新增257完整CLS接线正测、197/256/258/错误latent维拒绝、patch16/7/28/bool配置拒绝，绝不接受任意shape。

MAIN新预声明 `/home/gbk/code/LeWorldModel/docs/D2P_MOVING_OVERLAY_RECOMPILE_PROTOCOL_20261027.md`
已加入source_files及prepare bindings，固定SHA
`242e3277831ce3b5ac8db7d1da27d384e02a2485a218911bb3dbe80a0db7af95`，build/verify显式核验。
未改此预声明，不复制其他worker算法；下游物理/native与新RGB评价由MAIN/Bohr按该协议另行运行。

## 最终实现交付状态

独占测试文件完整运行：`41 passed in 14.01s`。包含四条件/16编码/8计划调度、原GN/D2P数组后端、
576预测对应的人工module/AD账、严格consumer正测、partial/错序/篡改拒绝、257修复与197反测。
本交付未运行实际prepare/verify/run/consume，也未加载模型或执行任何神经/物理。
三文件最终SHA在交接消息列出；MAIN可立即审读冻结后先独立运行CUDA，无需等待别的worker。
