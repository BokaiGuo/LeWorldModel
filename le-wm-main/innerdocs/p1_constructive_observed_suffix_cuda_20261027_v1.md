# Saved-prefix feedback suffix pilot

这是已知任务 saved-prefix feedback 反事实重放 pilot，不是在线从旧 terminal 选优，也不是 P1/P2 科学准入、非贪心或公平预算证明。仅新增同 stem 源码/tests/本文。MAIN 独占 HOST CUDA；本交付不运行模型/物理/encoder，不 prepare。

## 冻结输入与接线

Meitner source SHA `3508316e56515db38eca74c8516c4f434c50d273f1927830445767ef81400dbe`。
批准 actual report SHA `c77a5d208acb813ad968afe24ef0c547f08a83ff9968a984b29502baf0dda770`。
原 planner report SHA `4dd5823e9a273b820d6b7b3f6a36a4a5a19b193383eaa9cbf94442082e79d98d`。
material() 对完整 report、prepare、journal/PNG/NPY/model/source 逐文件验 SHA，调用原批准 task load_task/task_interfaces 路径；不读取 sourcepack/certificate/witness/blind NPZ。只 np.load rows 0/2/4/6 的 native float32[192] prefix5；terminal10 仅验 artifact bytes SHA，从不加载向量或使用 metrics 选优。原父 planner 封存核验包含其已封存输出，不将 imagined terminal 当 actual 输入。

新 context states=旧 root history[1:]+本臂 actual prefix5，between-state actions=[旧最后 zero token,本臂5 raw normalized]；不串臂，不重新拟合 Q/normalizer。task goal/Q 保持原值。新 observed_problem_id 区分原 task identity。source_slot/blind_sha256 只从批准 task 元数据接入，不解析 native root 身份。

## 预算与选择

固定两根×原 GN/D2P 两臂。每 branch：zero nominal1 + zero-point fresh H1 J 的 ADforward1 + 原 own imagined suffix1 + 新 ridge suffix1 + winner 独立 replay1，共5 predict/5 action_encoder（含 AD，不额外加）；1 J、1 SVD，functional.jacobian API1、nested batched grad API1、数学 VJP192。总20/20/4/4，非等预算方法对照。

原 damping=.002、cap=.75 sqrt(5)；ridge 对 logits 求解再 tanh。候选固定 zero/original/ridge，重复也不省调用，无 early exit；以原 Q nonlinear terminal cost 稳定 index tie-break 选优，winner full context 独立 replay 精确相等。模型前后 tensor SHA、输入源码重验；MATH SDP、TF32 false；逐调用 attempt/completion、失败持久 partial，无重试。输出总 cap2MiB（含 partial），reserve200MiB；gzip flush/fsync 继承冻结 Store，损坏流不伪造 footer。

## MAIN 命令

在 `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_constructive_observed_suffix_cuda_20261027_v1.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_observed_suffix_cuda_20261027_v1 prepare --rgb-report results/p1_constructive_selected_rgb_cuda_20261027_v1_cuda_main01/report.json --rgb-sha c77a5d208acb813ad968afe24ef0c547f08a83ff9968a984b29502baf0dda770 --planner-output results/p1_constructive_blind_planners_cuda_20261027_v1_cuda_main01 --planner-sha 4dd5823e9a273b820d6b7b3f6a36a4a5a19b193383eaa9cbf94442082e79d98d --output results/p1_constructive_observed_suffix_cuda_20261027_v1_prepare01.json
```

将 prepare stdout 的完整 SHA 填入以下 `PREPARE_SHA`（不得猜值）：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_observed_suffix_cuda_20261027_v1 verify --prepared results/p1_constructive_observed_suffix_cuda_20261027_v1_prepare01.json --prepare-sha PREPARE_SHA
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_observed_suffix_cuda_20261027_v1 run --prepared results/p1_constructive_observed_suffix_cuda_20261027_v1_prepare01.json --prepare-sha PREPARE_SHA --output results/p1_constructive_observed_suffix_cuda_20261027_v1_cuda_main01
```

## 物理最短接入与边界

只有完整 report status=`observed_suffix_plans_sealed_not_admitted`、errors=[]、suffix_commands_released=true 可下发。四行 rows 显式 source_slot(int)、arm(原名)、blind_sha256、commands(literal raw10x2=旧 own prefix5+新 suffix5)、selection_sha256。后者覆盖 plan 全内容；另有 plan 文件及 artifact SHA/整份 report 外部 SHA。可直接 physical1.selection 指向 `/rows/i/commands`、`/rows/i/source_slot`、`/rows/i/blind_sha256`、`/rows/i/arm`，不需要新算法 adapter。外部 evaluator 仍负责原 native recipe/roots 映射和 selection manifest；该 producer 禁止读取它们。manifest selection_digest 必须使用 physical.source.digest 的默认 spaces，不能替换为 compact digest。

同 recipe reset/warmup 后完整10步重放；prefix literal 精确相同，因果上新 suffix 不改变先前 prefix，但实际再现须下游核验。新物理尚未执行，不承诺 success 或接触改善。actual selectedRGB root1 D2P Q 显著改善而 native success0，不能简单归因为模型排序反转，更不能以 contact/success0 下理论失败结论。

测试只证明数组预算/接线/封存/失败合同，绝不认证 CUDA 导数或科学效果。科学准入仍 false，MAIN 需审代码并 prepare、CUDA、封存后由独立 evaluator 执行物理。

交付本地实核：批准 actual report 的 material() 已在 RAM 重建通过，4 branches、仅 prefix indices [0,2,4,6]，prepare 序列化 34,888 bytes，未写 prepare。测试覆盖5次调用（含ADforward）、三候选/稳定ties/old suffix可胜、history chronology、raw不静默round/clip、replay失败、SVD失败前J/nominal持久、predispatch cap、完整10步plan SHA、physical.selection直接指针、AD嵌套/恢复与gzip partial/final/broken。无新 neural/physics。
