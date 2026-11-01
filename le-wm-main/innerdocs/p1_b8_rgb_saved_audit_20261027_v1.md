# B8真实12RGB：独立saved-data复算审计

仅NEW同stem审计模块/tests/本文。只读实际PNG、raw NPY、JSON/完整gzip和源码/模型文件hash；不导入producer，不调用其analyze/collect/verify，不导入torch/gymnasium/pymunk，不跑神经/AD/物理。遵循experiment-code的回归与日志要求，任务限定不训练、不作图、不新拟合Q。

## 固定实际来源

所有路径基于 `/home/gbk/code/LeWorldModel/le-wm-main`：

- RGB `results/p1_b8_rgb_cuda_20261027_v1_cuda_main01/report.json` SHA `3f1b346979cd8e53e357aba0576a91d11e4268cc922dc84d63f5785622742d8e`，status=`all12_B8_checkpoints_encoded`，errors=[]。
- RGB prepare SHA `5c14c90a5498df36ef476cc037f9d685ccb909b93b1a25fcc524373e7f54a543`，路径由hash-bound execution_binding读取。
- physical `results/p1_b8_physical_20261027_v1_main01/report.json` SHA `e94a6cbf55dd9b408d350e690d17bfdb9ca693b2ef360a12ab904024f272cd7a`。
- physical prepare SHA `e872fdbfef99ef47a00d48ea9263a590cc3f792dd9b9e3184cebdbb56aaad706`。

独立核验prepare中全部103文件绑定，包括原任务/完整B8 incumbent gzip、source journals/源码/tests/doc、PNG、model config/weights。模型weights仅流式SHA，不反序列化；总绑定144个文件。重验physical journal完整hash chain、四branch receipt、原plan/命令/checkpoint及40 public step、8 reset、44 render的attempt/return分母。

## 复算合同

每个task从原gzip读出，重算完整task_id与goal/Q/context/action/model复合hash，检查原192×192 Q非零/对称/幂等，原模型config/weights声明一致；不重拟合Q、不替换goal。原B0..B8九iterate完整保留、严格incumbent acceptance重验，取B8 incumbent的两段预测边界，而非把GD最后的B7导数当B8导数。

每frame independently重建task/预测/physical diagnostic来源并验证PNG文件SHA、224×224 RGB像素SHA、canonical-agent像素数、实际state chronology；intent/result/report绑定。raw NPY必须float32 finite192、无尾随payload、无dtype转换后冒充native。三项标量独立float64数组复算：

    actual_Q_mse = mean((original_Q @ (raw_native - original_goal)) ** 2)
    predicted_Q_mse = mean((original_Q @ (original_prediction - original_goal)) ** 2)
    native_prediction_mse = mean((raw_native - original_prediction) ** 2)

与原analysis逐行核验身份、physical diagnostic及36标量（rtol=atol=1e-12，固定浮点复算容差，不是科学通过阈值）。所有12行参与：两根×GN/GD×0/5/10，12 distinct PNG paths，10 distinct pixel hashes，10 distinct raw latent hashes；重复step0仍计12编码，不降为10。

72条image事件须逐项匹配frame_started→encoder attempt/complete→projector attempt/complete→native_saved；24条module terminal与report entries完全匹配，资源为encoder 12/12、projector 12/12。predict/AD/physics零是hash-bound成功receipt及禁用列表为空的受限证据，不是本次重执行仪器验证。run文件总字节核验不超过原2MiB cap，声明200MiB floor未改；不声称追溯证明历史每次写入时free disk。

## 实际数值，不作科学准入

root0=`02a139165f6e8465ec882926c2ceb13a33ca5547f736ed1e3817242902050d06`；root1=`05844ca2c3ab8ba09846f9084edd71c8365ad74f7ab5961adaa5b0fded6289a7`。

| 原slot/arm | 起点原Q MSE | 实际端点原Q MSE | 预测端点原Q MSE | 实际端点−起点 |
| --- | ---: | ---: | ---: | ---: |
| root0 GN | 1.361121737181 | 1.245309841991 | 1.155789710870 | -0.115811895189 |
| root0 GD | 1.361121737181 | 1.214026768507 | 1.210409211497 | -0.147094968673 |
| root1 GN | 2.077293442328 | 2.089441582432 | 1.968030271909 | +0.012148140104 |
| root1 GD | 2.077293442328 | 1.743841302070 | 1.716405365543 | -0.333452140258 |

第二根GN预测下降而实际端点上升；如实保留，不删臂、不按成功过滤。这是当前冻结模型/原Q/具体根的诊断，不是驻点证书、独立根确认、物理任务成功率或理论失败。其他三臂实际端点下降也不能代替P1 direct plateau＋两段reach前置。模型tensor前后相同仅由固定执行receipt绑定，本审计重hash模型文件但不加载/重算tensor state。

## 执行与交付

工作目录 `/home/gbk/code/LeWorldModel/le-wm-main`：

    PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_b8_rgb_saved_audit_20261027_v1.py
    PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_b8_rgb_saved_audit_20261027_v1

CLI仅stdout完整小JSON（包括144 SHA、12行、4臂摘要），不生成result/tmp。tests的损坏fixture全部BytesIO/数组内存，无tmp_path；实际case与mock分别命名。输出status=`all12_saved_evidence_recomputed`，new_neural_calls/new_AD_calls/new_physical_steps=0，new_Q/P1_admission/independent_confirmation=false，new_independent_samples=0。无新方法优越性/计算优势结论。

实际验证：60 tests passed，含一次完整真实saved审计；独立进程CLI退出0。源码/测试/本文SHA以及stdout report SHA由交付消息提供，报告不另写磁盘。
