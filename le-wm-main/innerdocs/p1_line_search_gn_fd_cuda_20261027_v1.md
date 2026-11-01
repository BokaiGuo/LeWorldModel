# Line-search GN：八个fresh logit-J的实际CUDA central FD协议

仅NEW同stem源码/tests/本文，不改sealed runner、oracle或既有结果。按experiment-code技能先冻结诊断设计再实现日志/纯算术回归；本线程只跑mock与saved material/prepare，不启动CUDA、CPU neural、训练、物理或网络。真实FD只有MAIN显式run后存在。

## 已读实际证据与oracle

输入为 `/home/gbk/code/LeWorldModel/le-wm-main/results/p1_line_search_gn_cuda_20261027_v1_cuda_main01/report.json`，SHA `5f9825f98ed40f637c9d48e49c7c53428d1380e4234e3a1e5503a76f74c4bf1e`，原prepare SHA `e1b6d9783da6a4e32d360326869fe1b64fef8c3f4f168095be4b7f8c215b26d9`。读取实际5660事件、1536 raw rows，独立saved audit重建每根四轮的fresh J和完整预算。已有rawrows/tanh-chain/GN算术自洽不证明模型导数正确。

8个点取 `derivative_complete` 的 `iter_1_AD`…`iter_4_AD`：每轮**更新前**verified incumbent。不是更新后最终点，也不是B8原始导数。初始为zero，后续含非零且靠近packet ball边界的点；八点完整保留，不按FD好坏挑选。

已读 `p1_conditional_direct_screen_20261027_v1.NativeCuda`：AD对NumPy float32 tanh已定义的raw commands求完整H2 Jacobian，最后乘一次`1-commands²`转为logit-J。`_step`按原task mean/std归一化，action history追加当前token再经过action_encoder/predict；latent与history通过cat/切片跨包传播，receipt用detach转CPU不替换计算图。底层LeWM.predict直接调用predictor/pred_proj，不走image rollout的detach分支。FD复用同一native rollout数值oracle，但完全no_grad、每次从原root重新roll两包，不调用jacobian。因而能暴露AD路径中的跨包detach或链式错误；不承诺3方向排除完整192×20矩阵所有误差或同一oracle两侧共同的语义错误。

## 事前固定方向、步长、指标

每点固定三个logit空间单位方向：第一包e0、第二包e0、显式20维符号向量除sqrt(20)。方向不读J谱、梯度、结果或physical标签。eps固定`(1e-3,3e-3,1e-2)`；每eps两个符号，全部72组central comparisons。

    x_plus/minus = float32(float64(saved_incumbent) +/- eps * fixed_unit_direction)
    raw = NumPy tanh(x_plus/minus)  # float32, exactly once
    FD = (actual_CUDA_terminal_plus - actual_CUDA_terminal_minus) / (2 * requested_eps)

不clip、不project、不调eps。部分点在logit球边界，FD探针允许落在球外的网络自然延拓；它们不是可行优化proposal，不能拿来选动作或宣称约束问题的驻点反证。raw tanh仍在[-1,1]。每点先增加一次nominal H2，要求两个完整context hash及terminal与原ADforward逐字节一致，否则立即失败，不继续/重试。

同时比较saved J@v及J@v_effective，其中v_effective由实际float32加减点之差除2eps计算，显示rounding影响，不把它藏进容差。保存FD/Jv/有效Jv、absolute L2/Linf/RMSE、relative L2、symmetric relative L2、cosine、midpoint drift、effective direction误差、固定相邻eps FD稳定性。零分母为null，无拟合floor。**没有导数通过阈值**，derivative_verdict=`not_assigned`；实际数值由MAIN读后诊断，不事后调阈值过关。

## 全部预算，包含原ADforward

| 工作 | H2次数 | predict | action_encoder | J / 数学VJP |
| --- | ---: | ---: | ---: | ---: |
| 新nominal anchors | 8 | 16 | 16 | 0 / 0 |
| 新central正负两侧 | 144 | 288 | 288 | 0 / 0 |
| 新FD总计 | 152 | 304 | 304 | 0 / 0 |
| 已完成line-search（另计一次） | 78（70普通+8AD） | 156 | 156 | 8 / 1536 |
| line-search＋本诊断 | 230 | 460 | 460 | 8 / 1536 |

原156 predict已经包含8个AD fullH2 forward=16 predict，不能漏掉也不能再加16。新608个实际模块入口有pre-hook不可退款attempt及同步后completion，硬cap每类304；nominal不免费，重复点也不缓存。原B8 direct与image acquisition资源从原prepare单独继承一次，不乘8，不伪装为本次新增调用。数学VJP不是CUDA kernel/FLOP。每次collect还执行8个saved数组20×20 ridge solve和来源算术；prepare/verify/run final各自计实际调用，不偷称零CPU工作。新AD/encoder/projector/physics均禁止，不给总墙钟或FLOP等价认证。

## MAIN完整命令

工作目录 `/home/gbk/code/LeWorldModel/le-wm-main`。先纯saved verify/test，再由MAIN独占HOST CUDA：

    PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_line_search_gn_fd_cuda_20261027_v1.py
    PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_line_search_gn_fd_cuda_20261027_v1 prepare --output results/p1_line_search_gn_fd_cuda_20261027_v1_prepare01.json
    PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_line_search_gn_fd_cuda_20261027_v1 verify --prepared results/p1_line_search_gn_fd_cuda_20261027_v1_prepare01.json --prepare-sha PREPARE_SHA256
    PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.p1_line_search_gn_fd_cuda_20261027_v1 run --prepared results/p1_line_search_gn_fd_cuda_20261027_v1_prepare01.json --prepare-sha PREPARE_SHA256 --output results/p1_line_search_gn_fd_cuda_20261027_v1_cuda_main01

将prepare输出的SHA人工带入后两行；如文件已存在不重建、不覆盖，直接verify。不得自动从未核验文件自取hash充当外部授权。模型本地配置pretrained=false，offline环境，explicit cuda:0/no CPU fallback、SDP MATH、与原runner相同的matmul TF32=false。原receipt未记录cuDNN TF32 flag，故本程序不猜原值、不另改，记录启动时实际值并强制nominal锚定；不匹配则停，不自动切换数值模式。模型全eval且参数/buffer CUDA；before/after tensor hash匹配原实际line-search model。原任务/Q/normalizer/source重新hash绑定，run最后再verify。

新prepare为单个<256KiB hash/向量manifest，不复制Q、完整J或gzip源。run新目录全部文件共享2MiB cap；开始要求202MiB free，每次写前保留200MiB，128KiB留给final。events.jsonl.gz逐事件Z_SYNC_FLUSH/fsync，保存每包terminal/full-context hash和每侧完整request/terminal；analysis.json.gz保存全部72比较。每侧完整context在RAM校验原task history/token chronology，不落重复Q/历史大数组。仅完整gzip EOF可以叫完成；失败保留partial、不删除、不重试、不降floor。不能保证磁盘故障时final report仍可写，primary错误与后续错误分开。

## 测试及科学边界

mock为NumPy affine two-step recurrence，不是CPU神经模型：第一包通过latent .7与action-history .3两通道影响终点。测试分别注入missing/double tanh chain、missing normalization、history或latent导数遗漏，证实FD能区别这些错误，另验完整152/72分母、浮点网格、同步失败不退款、root guard恢复、压缩cap（只代表mock）、真实8J material。无tmp_path/大tmp。实际模型压缩大小无法由mock预认证，运行时cap仍硬拒绝。

本交付尚无真实FD结果，不宣称导数已认证。未来FD吻合也不是P1 direct plateau/两段reach/独立根通过，更不能据此下理论失败或强基线优越性结论。

交付验证：28 tests passed，包括真实saved八J material；无torch/gymnasium/pymunk导入。小prepare只在内存序列化核验（约148KB），未在本线程写prepare或启动run。MAIN执行上面prepare后，须核对输出SHA再verify/run。文件SHA与内存预期prepare内容SHA见交付消息。
