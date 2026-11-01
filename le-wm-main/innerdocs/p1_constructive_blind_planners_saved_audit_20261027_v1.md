# Constructive blind planner 独立 saved 审计

只新增本 stem 源码/tests/本文及小报告；不导入 planner、producer、GN/D2P compiler、torch 或 simulator；不调用 selected/consume/verify 等验证核心。依赖先前独立 encode saved auditor 的 JSON/散列工具及常量，不调用其 audit/reconstruct。只读明确 task export、prepare、report、result、plan、journal，以及 source/model 文件 hash。不打开 NPZ/PNG/sourcepack/certificate/物理结果；不扩展物理 bridge。

## 外部绑定

- 冻结 planner：`e56bee610f4eb412cc03fbe49b3df91054d0defe23d24b6fd9c77e6a7f1e9f27`
- MAIN actual report：`results/p1_constructive_blind_planners_cuda_20261027_v1_cuda_main01/report.json`
- report SHA：`4dd5823e9a273b820d6b7b3f6a36a4a5a19b193383eaa9cbf94442082e79d98d`
- prepare SHA：`30b58155a59f73fdc8cc132ff069e645ed6c2bbff9434f474b1f7b9c412f7eb4`

## 独立复算内容

taskID/contractID 和 plan selection seal 的 sorted JSON 使用原默认空格序列化，不混用 receipt compact newline 格式。导出原生float32 latent、原模型、identity192 Q、3帧/2token、rawzero→非零 normalized token、with_target=False、无native root身份均复核。

GN：初始 zero 和35次 H2 evaluation 的 request→两packet→complete 顺序、全部70个context、每次更新新AD点、192行raw-control VJP、单次tanh链与192×20 logitJ、梯度、原ridge damping .002、projected endpoint、固定五scale float32 segment、全部 fresh score、incumbent优先稳定选择和每轮独立三重 replay。保存结果、步骤、最终预测及plan逐项对应。

D2P：根J/nominal→task-weighted spectral sign convention/两固定半径/zero/DLS十候选bank；各prefix新conditional J/nominal→有界ridge与ball multiplier、原reach penalty、stage+reach objective、非线性terminal分数；完整10步tanh动作和context、每候选独立whole replay、winner whole/segmented出口。全部十候选保留，不以terminal score冒充stage+reach选择。

NumPy线性代数重算比较采用 `rtol=1e-8,atol=1e-10`，只是浮点复算容差，不是科学有效性阈值；原saved action映射、score和history逐内容精确检查。NumPy SVD/solve仅对保存数组算术，不执行CPU神经。

账本：attempt/return独立索引和嵌套，失败/未返回/重复/未知工作拒绝；真实AD API和数学VJP分开。每根GN78predict/78actionencoder（内含8ADforward）+4J/768VJP/4SVD/4solve/proposal/selection；每根D2P66/66（内含11ADforward）+11J/12SVD/2112数学方向，但实际11grad+11functionalJacobian API。两根总288predict/288actionencoder，不能另加ADforward，也不能将数学方向当API次数。

两算法同task/goal/Q/动作域，**不等预算、不等选择目标**。继承8图编码另账，非全pipeline免费。8MiB原输出cap、完整文件census/hash/最终TOCTOU、错误与partial严格拒绝释放；拒绝时新报告记录失败，不修复原输出。

## 命令

在 `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_constructive_blind_planners_saved_audit_20261027_v1.py
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_blind_planners_saved_audit_20261027_v1 --report results/p1_constructive_blind_planners_cuda_20261027_v1_cuda_main01/report.json --report-sha 4dd5823e9a273b820d6b7b3f6a36a4a5a19b193383eaa9cbf94442082e79d98d --prepare-sha 30b58155a59f73fdc8cc132ff069e645ed6c2bbff9434f474b1f7b9c412f7eb4 --output results/p1_constructive_blind_planners_saved_audit_20261027_v1_saved01.json
```

新输出<128KiB且写前保留200MiB，exclusive不覆盖。测试仅RAM数组/实际saved故障注入，不制造大fixture；无实际receipt时saved专属测试明确skip。不能由saved内部一致性证明神经/导数真值、没有账外调用、科学收益或物理success。没有P1准入或FLOPs等价结论。
