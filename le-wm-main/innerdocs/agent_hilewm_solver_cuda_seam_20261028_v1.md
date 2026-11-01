# Hi-LeWM solver CUDA seam v1

本三件套仅供 MAIN 审计后执行安装版 CEM 的完整优化循环接线实验。**非神经 analytic cost，不是 Hi-LeWM 官方模型结果或官方未适配复现**；不训练、不加载权重/pickle、不调用 foreign model loader、无 physics、无 CPU fallback。不改冻结 bridge、安装 CEM、policy/global 或任何封存输入。worker 只运行本文件的 RAM schema/control 测试，不执行 torch/CEM。

## 固定实验与计量

两次独立 solve，顺序 B=5、B=2；chunk=2；S=8，iterations=3，topk=3，H=3，D=2，float32，var_scale=1，seed=1729+B。B5 有2/2/1三批，B2有一批。每批三次 cost，合计12 attempts/12 returns（完整成功时）、168 candidate rows。candidate rows 不等于 predictor samples；predict API=0、predict samples=0、AD=0、physics=0。CEM callback reset2/start_batch4/end_solve2/callback12。无padding、无额外mean replay、无重复solve。

z_init 每行为 `[id,id+0.5]`，z_goal为 `[(id+1)/8,-(id+1)/16]`，id=0..B-1。固定 analytic cost=`sum_H,D((candidate-z_goal)^2)`，只在CUDA执行。dtype sentinel 是一枚非训练零tensor，仅给 CEM.parameters() 读取dtype；不是神经模型。完整H的零init_action在CUDA，不使用actor，也不改变solver原有sample0=mean、topk、mean、sample std(correction=1)更新。安装字段var实际保存std，严禁当variance复算。

通过冻结 `HighBatchBridge.bind_high_model` 显式接线：仅tensor info送入CEM，get_cost边界注入完整high；原info/solver info/expanded info/candidates/cost字节见证都保留。新建CEM实例，无monkeypatch。真实 torch.expand 要求 sample轴stride=0，并按batch offset逐位核对两字段来源，包括尾批。每次原生callback保存完整9项张量，CUDA上重新执行topk/gather/mean/std并要求torch.equal；sample0严格等于prev_mean。保存原生输出actions/mean/var/costs及callback历史。

源码绑定自身三件套、冻结bridge SHA `a651d5f0710d18bea1f9b2caaf3283286816ca775dd32dcdc619f11d219c448b`、安装版CEM及utils/protocol/callback关键文件；冻结bridge继续验证其owner CEM SHA `a6662e7f1f2c6ecaf0a8527255d531aa05ed583339aabdf2798d0206382ef51b`和官方high_info文本。普通Python正常import安装CEM包，可能递归import已安装依赖的定义；不伪造sys.modules包、不抽取/重写solver。不声称绑定整个Python环境的所有传递依赖，运行记录torch/CUDA/device版本；缺安装依赖应保留导入错误，禁止补CPU实现。

## 严格saved消费

`consume` 不import torch/foreign solver，只用NPY(allow_pickle=False)重新检查全部case/步骤/来源、文件和原bytes SHA、双份info见证、每次cost/bridge记录、原生elite gather、状态更新链、最终输出及全部数组使用。无缺失填0，无partial补步。逐case完整9+3迭代分母必须齐全。逐次CUDA精确checks必须全true。

独立NumPy float64复算analytic cost/mean/sample std/elite均价，与保存float32比较的固定工程rounding容差为rtol=atol=8×float32 eps，预先封存在protocol，不是科学效果阈值。gather、来源、sample0、链、输出及SHA严格相等；无outcome调参。saved消费不能独立证明硬件真实性，真实运行身份由MAIN受控执行和保存日志提供；RAM合成CUDA字段只是格式mutation测试，不当CUDA证据。

NPY逐数组exclusive创建，report/receipt也exclusive，旧目录一律拒绝。8MiB总输出cap，保存数组前留1MiB报告余量；200MiB disk floor，120s cooperative wall gate（callback间检查，不能抢占单次CUDA/kernel/import）。MAIN可使用外部timeout作硬界。无吞异常：迭代失败保存partial report及已发生attempt，run CLI非零退出；consume拒绝任何errors/partial。CUDA不可用/来源不符/未授权在创建run输出前拒绝。安装solver本身包含CPU初始化variance和最终CPU序列化；这是原实现的临时参数构造/存储，不是cost/优化循环CPU fallback，且无神经计算。

## MAIN精确命令

在 `/home/gbk/code/LeWorldModel/le-wm-main`，先完整读码并执行RAM tests。以下结果目录只是未来执行位置，不预造实际report/receipt SHA；已存在则用新MAIN编号，禁止覆盖。

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_hilewm_solver_cuda_seam_20261028_v1.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B experiments/spectral_reachability/agent_hilewm_solver_cuda_seam_20261028_v1.py prepare --out results/agent_hilewm_solver_cuda_seam_20261028_v1_prepare_MAIN01
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B experiments/spectral_reachability/agent_hilewm_solver_cuda_seam_20261028_v1.py verify --manifest results/agent_hilewm_solver_cuda_seam_20261028_v1_prepare_MAIN01/manifest.json
OPENBLAS_NUM_THREADS=1 timeout 180s .venv/bin/python -B experiments/spectral_reachability/agent_hilewm_solver_cuda_seam_20261028_v1.py run --manifest results/agent_hilewm_solver_cuda_seam_20261028_v1_prepare_MAIN01/manifest.json --out results/agent_hilewm_solver_cuda_seam_20261028_v1_cuda_MAIN01 --main-authorized
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B experiments/spectral_reachability/agent_hilewm_solver_cuda_seam_20261028_v1.py consume --manifest results/agent_hilewm_solver_cuda_seam_20261028_v1_prepare_MAIN01/manifest.json --report results/agent_hilewm_solver_cuda_seam_20261028_v1_cuda_MAIN01/report.json --out results/agent_hilewm_solver_cuda_seam_20261028_v1_consume_MAIN01.json
```

即使未来实际通过，也只排除安装CEM完整CUDA循环/expand/metadata接线的这一工程缺口，不解除合法完整Hi checkpoint、可信loader、config/normalizer、macro/RGB→latent配对、共同控制器context及科学准入等阻塞。没有模型或任务收益结论。
