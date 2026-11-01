# P2 delivered mean: paid actual-token replay v1

事前固定：known roots0/1，slots0/1/3/4 returned、2/5 arbitrary missing，六槽不删。只接受 raw-domain prepare `results/agent_p2_delivered_mean_raw_domain_20261029_v1_prepare_MAIN01.json`，SHA `9df31325d7d2bf149b60db0f6d21ce40e6284060c4bb3af29c1699e1ee056c56`。绑定完整原 controller report/receipt、normalizer source、task 与425 state。prepare/verify只重建 saved 资产，不调用旧 full strict。

HOST CUDA eager float32 独立 mul→add→sub→div，保存每个操作的输入/输出、stride、device、requires_grad。mean按时间顺序 reshape15×2。mul/add所得 raw 必须与已封 NumPy raw 的120bytes逐位相等（包括signed zero）；不相等或超 native[-1,1] 则保槽blocked，不clip/tanh/fallback。实际 sub/div tokens不要求等于旧 mean；逐坐标报告bits、ULP、zero signs。原 Hi scaler 未认证；仅沿 MAIN 指定P2工程约定，无raw部署许可。

每合格槽 n=1、H3 whole一次、segmented[1,2]一次，无CEM/搜索/重选。每步完整 action history先append后AE、last3 predictor、flatten projection，完整保存模块IO与所有causal histories。每槽各6 AE/predict/lowproj；四槽完整时各24调用，24 predict samples，96 AE tokens、72 predictor context tokens、72 lowproj rows。failed dispatch先记账，不退款；partial保存但不可获得完整成功receipt。blocked不虚报24。encoder/high/AD/J/VJP/backward/physics=0。

四槽预测文件全部落盘并生成不可覆盖 prediction_seal.json 后才读取原target计算各whole/segmented terminal identity192 float64 meanMSE。不是planner评分或新选择。独立消费用 math.fsum 复算，浮点归约误差界沿冻结core：8×192×eps64×max(|reference|,tiny64)；这不是tokens/whole-segmented等值容差，后两者必须byte-exact。保存旧mean、实际token、raw身份及两类roundtrip差异；不沿用旧预测成本。

资源：CUDA allocated3GiB/reserved4GiB；总输出512MiB含64MiB终态reserve；floor200MiB；全scope900s、每槽180s（含inverse、同步、数组捕获、持久化，不含构造前置；总scope包含prepare复核、构造、hash）。合作式逐操作检查+900s SIGALRM，不声称抢占CUDA kernel；footer允许越时保存失败。每槽trace payload上限64MiB。先排他保存全部输入再构造；所有源/输入与425参数buffer before/after强等。无CPU neural fallback、无pickle、无新encoding/physics。RAM tests不能证明CUDA身份。

MAIN 从repo根运行（下面M/H/R/S需用上一条命令实际返回值，禁止未来hash）：

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q tests/test_agent_p2_delivered_mean_paid_replay_cuda_20261029_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_delivered_mean_paid_replay_cuda_20261029_v1 prepare --output results/agent_p2_delivered_mean_paid_replay_cuda_20261029_v1_prepare_MAIN01.json
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_delivered_mean_paid_replay_cuda_20261029_v1 verify --manifest "$M" --manifest-sha256 "$H"
CUBLAS_WORKSPACE_CONFIG=:4096:8 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p2_delivered_mean_paid_replay_cuda_20261029_v1 run --manifest "$M" --manifest-sha256 "$H" --output results/agent_p2_delivered_mean_paid_replay_cuda_20261029_v1_cuda_MAIN01 --main-authorized
.venv/bin/python -B -m experiments.spectral_reachability.agent_p2_delivered_mean_paid_replay_cuda_20261029_v1 consume --manifest "$M" --manifest-sha256 "$H" --report "$R" --report-sha256 "$S" --output results/agent_p2_delivered_mean_paid_replay_cuda_20261029_v1_consume_MAIN01.json
```

所有旧资产不改。此交付未运行生产/新fullstrict。工程replay不认证Hi scaler、不主张公平预算/科学P2、不改变prior time unknown、P1/scientific gate false，不授予物理许可。
