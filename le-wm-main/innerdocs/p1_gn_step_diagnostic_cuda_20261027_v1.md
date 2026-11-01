# B8 GN 固定可行线段步长诊断 v1

仅 MAIN 可在 HOST CUDA 执行。prepare/verify 是保存数组/hash操作；本次交付不执行神经、AD、图像编码、physics、训练或下载，不修改任何旧sealed文件。遵循experiment-code的先冻结实验、完整日志原则；固定专用schema与小输出优先于通用实验目录/图示模板。

## 精确输入和问题

固定 `p1_observed_direct_cuda_20261027_v2_cuda_main01/report.json`，SHA
`954af696d88eecebdcdc7345de702809517491967263f395a8f2819753d889d5`。
原两根、原native192 observed-history task、同goal/Q/model/normalizer；取每个GN `direct_runs[0][7].incumbent` 作为iter8更新前x，取 `[8].proposal` 为原固定大步被拒绝的y。读取原gzip task/arm/journal，核对完整0..8 trajectory、原derivative point/数组及proposal日志；不读MAIN stagnation报告（包括audit02）决定选择或阈值，不把saved绿色结论当原始输入。

四个预声明尺度按原序全部执行：`[.5,.25,.125,.0625]`。
计算float64 `x+s*(y-x)` 后一次float32取整，再用NumPy float32 tanh生成原生10×2 rawcommands。端点和取整后每packet必须满足原logit球，越界拒绝；没有clip、再次project、延长方向或scale1候选，没有自适应上调/缩放、Armijo拟合、新Q或新阈值。这是原feasible logit segment的有限采样，不是重新运行GN或证明收敛。

## 全部32次预算和durability

| 每根工作 | fullH2数 | predict/action_encoder各 |
|---|---:|---:|
| 新复算旧incumbent，完整context逐位对应原保存值 | 1 | 2 |
| 四个尺度，每个真实native wholeH2 | 4 | 8 |
| 选优后独立nominal、whole、segmented | 3 | 6 |
| 每根/两根 | 8/16 | 16/32 |

先完成本根全部四尺度再在新复算incumbent+四新值中作stable `(cost,index)` 最小选择，ties保留incumbent/原次序；即使首尺度已改善也不省略后续尺度。没有改善则选incumbent仍做完整6次独立验证。选优前不把原y的旧分数混入新候选，不新算原大步。真实 `screen.NativeCuda` 保留原full3滚动latent/action history，normalizer精确float32 time-major token；三个验证均从原task root重新开始，segmented在第二packet显式重建完整context。检查全部两个边界内容，不只history shape。

自有Meter对NativeCuda实际predictor/action_encoder module hook预dispatch扣账，严格16个evaluation顺序、每packet actionencoder→predict、每根各16及全局各32，no-grad模式，失败不退款/不重试。每个完整返回context立即fsync journal，所有新eval完整保存（含动作/contexts/原Qcost），选优selection先持久化再独立验证；partial/error和已付费entries保留，不伪补32。NativeCuda的image/projector hook禁止新图像调用，runner没有AD入口。真实命令/selection保留，后续physics必须独立新consumer，不能把本诊断当物理结果。

2MiB为整个execution目录硬cap，预留192KiB终态，写入保持200MiB；run启动须至少202MiB空闲。prepare为单一hash-only JSON，<512KiB，不复制task/Q矩阵或大源码。绑定原B8 prepare、gzip task/arm/journal、原model/config、source/core/normalizer以及新source/tests/doc。保存旧direct/image资源一份作为继承历史，不当免费，也不加入本次32重复收费。模型before必须等原B8真实tensor hash，after同before；原来源文件及模型字节前后重验。异常首错和cleanup/hash/source错误分别保留；硬杀或磁盘无法写时不能保证最终report，但不把缺失receipt当完成。

本工作是固定步长“假stall”反例诊断：若原方向内较小步真实降低同Q nonlinear cost，则不应把B8停滞说成该方向无改善或stationary。若四点均不改善，也不能证明整条连续线段无改善、理论失效或科学准入。不存在事后passing阈值；所有固定点的正/负/零改变量原样报告。仍是原两根calibration、0新独立根，非物理效能、非P1通过。

## MAIN 审阅后命令

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_gn_step_diagnostic_cuda_20261027_v1.py
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_gn_step_diagnostic_cuda_20261027_v1 prepare --output results/p1_gn_step_diagnostic_cuda_20261027_v1_prepare01.json
```

prepare输出准确SHA后，MAIN使用真实值，不能用占位符执行：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_gn_step_diagnostic_cuda_20261027_v1 verify --prepared results/p1_gn_step_diagnostic_cuda_20261027_v1_prepare01.json --prepare-sha ACTUAL_SHA
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_gn_step_diagnostic_cuda_20261027_v1 run --prepared results/p1_gn_step_diagnostic_cuda_20261027_v1_prepare01.json --prepare-sha ACTUAL_SHA --output results/p1_gn_step_diagnostic_cuda_20261027_v1_cuda_main01
```

测试仅内存NumPy mocks与原saved输入，不申请pytest tmp；不创建模型/环境。CUDA正确性、32次实际工作与是否改善均须等待MAIN真实receipt。本模块不声称walltime/VRAM/FLOPs匹配；MAIN应在独占进程使用外部watchdog。禁CPU神经fallback。
