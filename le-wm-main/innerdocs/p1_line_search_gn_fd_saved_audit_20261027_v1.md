# Line-search GN FD：独立saved audit

仅NEW同stem模块/tests/本文。按experiment-code技能执行纯数组回归；无模型/神经/AD/物理/网络调用，不导入FD producer，也不调用其requests/analyze/compare。旧producer、prepare、CUDA输出及共享文档不变。CLI仅stdout，不新建结果或tmp。

## 外部绑定及独立重建

实际目录 `/home/gbk/code/LeWorldModel/le-wm-main/results/p1_line_search_gn_fd_cuda_20261027_v1_cuda_main01`；MAIN明确提供report SHA `1dd13d4f8ce4ac2bcdee7e8e360ef283962bae11314efb19f017cbeb92d44163`。prepare SHA固定`29b70d36fb5a4f49f22474a9fc5932f566c2eaab7357b4de70d2411389e974b0`。先核验report及所有artifact SHA、prepare与完整绑定，再读gzip；拒绝partial EOF、串联流、尾随bytes及torn JSONL。实际文件总量重验2MiB cap、原声明200MiB floor未变，不追溯认证每次历史写入时free disk。

复用原独立line-search saved audit的输入装载与raw-J/GN/预算核验，重新从原5660事件取八个derivative_complete（两根×iter1..4更新前incumbent）、原AD包context、full192×20 logit-J。自行生成三个固定方向与三个eps，按float64加减后一次float32、一次NumPy tanh重建152请求；不从producer helper取得expected。

逐序消费1824 FD事件，核对608次module attempt和608次completion；每次两包顺序固定、资源不可退款，无多余或漏项。**304个完整context不只核对字符串收据**：用原task root history、每包saved terminal和真实signed raw commands按原float32 mean/std重新构造latent/action history，再独立hash，与native_packet及evaluation_complete逐项核验。八个nominal同时匹配原AD两包context hash与终端。共152完整evaluations、72 central comparisons，不去重、不挑eps、不删失败方向。

自行复算全部FD、saved Jv、effective-float32 Jv、effective方向、两套L2/Linf/RMSE/relative/cosine、midpoint drift及相邻eps稳定性；逐字段/向量与analysis.json.gz比较。1e-12 relative/absolute仅为saved浮点算术核验容差，不是导数通过门槛。零分母保留null，summary报告defined/undefined分母，不将其丢弃后伪造全覆盖。

## 实际独立复算数值

每eps覆盖8点×3方向=24组；每方向覆盖8点×3eps=24组，9个direction×eps cell各8组。以下均为effective-float32方向的relative L2，保留全部预固定eps，不挑最有利值。

| eps | minimum | maximum | minimum cosine |
| --- | ---: | ---: | ---: |
| 0.001 | 0.000249669026775 | 0.002693676506742 | 0.999996620146239 |
| 0.003 | 0.000217180402559 | 0.001134559265387 | 0.999999362555361 |
| 0.01 | 0.000238432735400 | 0.000798154215853 | 0.999999685045121 |

| 固定方向 | relative L2 minimum | maximum | minimum cosine |
| --- | ---: | ---: | ---: |
| packet0_e0 | 0.000220087335581 | 0.001919043053073 | 0.999998273729395 |
| packet1_e0 | 0.000217180402559 | 0.000873878339828 | 0.999999622379309 |
| fixed20_sign_unit | 0.000352594691654 | 0.002693676506742 | 0.999996620146239 |

复算与MAIN初读范围相符。supported结论仅为这八个点、三个方向、三个eps下的saved Jv与实际模型central FD及资源收据吻合，不命名“通过阈值”或宣称全部J已认证。

## 资源与边界

新FD：8 nominal H2+144 signed H2=152；304 predict/304 action_encoder。原line-search：156/156已含8个AD fullH2 forward=16 predict，不能漏算或再加；原8J/1536数学VJP另列，新FD AD=0。line-search+FD总460/460，上游B8/image另计一次。本审计无新神经、AD、物理，只读模型文件hash，不加载weights。原tensor前后不变依赖hash-bound CUDA receipt，不是本审计再次算tensor state。

mock保留missing/double tanh、missing normalizer、latent/history detach错误J的反例，且不调用FD producer；mock敏感性断言不是实测阈值。测试还覆盖请求篡改、nominal锚点、错误history/normalizer、重复/漏资源事件、bool冒充计数、gzip partial/final、FD向量/科学scope损坏。全部fixture在RAM，不写tmp。

三方向不是192×20全矩阵认证，也不能排除forward与AD共同的其他语义错误；FD探针允许球外网络延拓，不证明球约束驻点。原Q/历史/正常化合同及具体点外的generalization、P1 direct plateau/两段reach/独立根/物理效用仍未获准入。无理论失败或强基线优越性结论。

## 命令

工作目录 `/home/gbk/code/LeWorldModel/le-wm-main`：

    PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_line_search_gn_fd_saved_audit_20261027_v1.py
    PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_line_search_gn_fd_saved_audit_20261027_v1 --report-sha 1dd13d4f8ce4ac2bcdee7e8e360ef283962bae11314efb19f017cbeb92d44163

stdout小report包含完整绑定、全部72 scalar-comparison摘要、按eps/方向/两者交叉分层统计及固定eps稳定性。status=`saved_fd_arithmetic_verified_not_admitted`；derivative_verdict=`not_assigned`、P1_admission/independent_confirmation=false。

## 本次实际验证

新增审计31 tests通过；与原FD producer的28 tests联合运行59 passed（4.48s），保留tanh/normalizer/detach反例。以上mock测试只验证审计及反例敏感性；另有实际sealed saved-data审计，不是新模型执行。CLI实际退出0、stderr为空，stdout 89756 bytes，SHA256=`6d2a66cfe412575322df0f5077de8ff57427914a2e83655cfd58da667f63d7cf`，仅在RAM捕获、未落盘。实际分母8 anchors/152 evaluations/72 comparisons/304 reconstructed contexts/1824 events全部吻合。
