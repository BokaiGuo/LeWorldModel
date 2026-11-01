# P3 RUN_MAIN01：已消费四槽的同根、同控制步对照

本分析是已存运行的 CPU saved-only 后处理，不启动模型、GPU 或环境。先核查了 `D2P_P3_MATCHED_FEEDBACK_HOST_20261030.md`：已有四槽完整保存消费回执，但没有本运行的原生任务结果与配对资源表；旧 commit-H3/MPC 指标分析针对另一面板，不可复用为本运行结果。原输入未修改。

输入是 RUN_MAIN01 的 report、journal 和独立 strict receipt；脚本核 report/receipt 固定 SHA、journal SHA、固定槽序及配对根的 root hash、goal、cp0 snapshot 和像素 hash。完整来源和动作链认证继承现有 receipt，不宣称另行独立重放。两根均为已知根；四槽是两对，不是四个独立根。

## 任务指标

cp0/5/10/15/20 均来自对应计划步实际保存状态；按原 native 7 维 state-goal 的欧氏距离计算，包含速度。cp5/10/15/20 重算 reward 与保存的原生返回逐点核对；成功按原 joint XY 和原 native angle 规则计算，而非 wrapped-angle 替代。这里的“同时间”指相同控制步，不是等 wall time。

| 根 | 步 | signal − fixed：native7 距离 | signal − fixed：joint XY | signal − fixed：速度误差 |
|---|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | 0 |
| 0 | 5 | -60.790 | -2.729 | -70.899 |
| 0 | 10 | 33.235 | 16.445 | 30.853 |
| 0 | 15 | 38.466 | 48.166 | 11.738 |
| 0 | 20 | 121.212 | -2.060 | 131.188 |
| 1 | 0 | 0 | 0 | 0 |
| 1 | 5 | -332.245 | 34.975 | -420.124 |
| 1 | 10 | 58.877 | 2.736 | 117.703 |
| 1 | 15 | 163.291 | -6.171 | 213.820 |
| 1 | 20 | 58.397 | 5.168 | 63.244 |

负的距离差表示 signal 较小。20 步 native7 距离依次为 root0 fixed 185.370、signal 306.581；root1 fixed 387.792、signal 446.188；四槽 native success 均为 false。第 5 步两根 signal 距离较小，第 20 步均较大；root0 第 20 步 joint XY 反而略小，说明不能把 native7 差异简单解释为几何差异。各分项与全部检查点的完整数值见 JSON；不对两个已知根进行推断检验或声称泛化。

## 资源边界

四槽各有 20 计划控制步、10 warmup 步、4 个单独观测编码边界、22 个 native numerical calls；数值 ledger 各记录 52 predictor calls、768 VJP directions、18 replay calls。物理 meter 每槽分别记录 30 public steps、304 physics steps、32 explicit renders；这些是 instrumented intent 计数，完整闭环由已有 strict receipt 支持。ledger 的 encoder/projector 数为 0 不表示没有实际观测编码，它们在独立 encoding hook 记录。

fixed 的 horizon 顺序为 [1,3]，signal 为 [3,1]。数值 ledger wall 分别为 7.371、7.118、7.273、8.990 秒（按槽 0–3）；是局部计时，具体 operation wall 有嵌套包含关系，不能加总为端到端时间。全四槽 report 范围 wall 为 51.017 秒；完整单槽 wall、CUDA 时间与 FLOPs 均未记录，保留 null。相同计数不证明等算力，也不证明延迟优势；本分析没有构造相同 wall 时刻的任务对照。

## 复现与检查

代码 `le-wm-main/experiments/spectral_reachability/p3_run_main01_cpu_comparison_20261030.py`；结果 `le-wm-main/results/p3_run_main01_cpu_comparison_20261030_v2.json`；初版 v1 不含局部 wall 字段，保留但以 v2 为准。单元测试 `le-wm-main/tests/test_p3_run_main01_cpu_comparison_20261030.py`：3 passed。脚本在 `CUDA_VISIBLE_DEVICES=''` 下运行，且只使用 NumPy 和保存日志，无 GPU 调用。单元测试仅检查数值、输入校验和哈希函数，不代替原 strict receipt。
