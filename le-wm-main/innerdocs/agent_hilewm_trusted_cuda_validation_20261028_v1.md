# MAIN HiJEPA tensor construction + high-latent source-fidelity driver

独占本stem source/tests/doc。冻结loader v1/exporter/共同controller/旧结果不改，不依赖新增macro v2。使用experiment-code来源固定、显式预算与RAM回归；不训练、不拟合、不执行pickle/foreign模块或下载。worker不prepare、不读生产tensor值、不import torch、不运行神经/物理。MAIN审读后执行下列命令。

## 实际prepare不是手写回执

冻结exporter实际签名为`run(output, wall_seconds=300, verify_sha=None)`。prepare先核loader/exporter三文件SHA与固定IR/text闭包，再调用`exporter.run(EXPORT_DIR, wall_seconds=300, verify_sha=3cd7dd6783129893f397ddef63bd253e447fbef93582dd5f4dc8d7144310efd3)`，真实验证425 storage+425 entry NPY、原ZIP和CRC；这只读取原ZIP/data storage，不执行pickle。原export目录不写任何文件。新prepare目录排他保存**精确return dict**为`export_verify.json`，没有添加metadata；其实际SHA写入manifest。随后真实调用loader.authorities绑定它。不拿已有终端输出补写、不把fixture当真回执。失败仅新目录failure.json，保留不覆盖。

verify核manifest自身指定SHA、完整source闭包、政策及receipt字节/原export身份；不重做全部850数组验证，不import torch。run中的loader.construct_cuda逐entry再次读SHA/size/dtype/shape和值hash、传CUDA后核bytes，再核425个state_dict值；完整531模块/491项/66None，不宽松load、不改键。construction receipt完整嵌入新report。

## 固定四项，不选择样本或结果

1. direct：B1,T1,z192,macro32。
2. direct：B2,T4,z192,macro32。
3. rollout：B2,S2,T5，覆盖保存context4后的裁切。
4. high cost：B2,S2,H2,width64，reshape为4个macro32，终点平方L2 **sum**，不是MSE。

所有输入采用固定dyadic算术`float32((arange(numel)%31)-15+seed)/64`，四case seed1..4，z/action/goal偏移0/7/13；不是随机训练/拟合，也不输入任何私有物理结果。全部四分母保留，无outcome选择。owned路由调用冻结root.predict_high/rollout_high/get_cost；reference路由只用state_dict与graph literal字段，手写独立dictionary dispatch，不调用owned节点.forward。source固定本地module.py SHA`0b258a9e8dc24c29fcb1e8c50a09ec78b8ea85aeb79e21dd8adf712396646620`的Attention/ConditionalBlock/Transformer/ARPredictor，官方jepa.py SHA`bb5338bbb4f9e76b0bd15b94767624168883c994ab4c2ad8209337a3d70022a2`的predict_high/rollout_high/get_cost_high。源码只读不import/exec；reference和owned共享同一封存权重、torch functional库。保存实际全部输入/owned输出/reference输出NPY，finite/shape及**float32 bytes完全相等**为预固定工程门，不因误差改阈值。它不是官方历史bytecode的独立parity，更不是任务收益证据。

固定资源分列：owned高predictor11 API/39 batch-sequences/105 context-token positions，reference亦11/39/105；top-level4×2=8dispatch。数据copy425独立于这些量；不能把API、batch、token、copy或时间相加。owned actual pre/post hooks记录每次batch/context；reference在dict计算边界逐attempt/return记录，失败无退款。所有hooks finally移除，完整consumer核计数和顺序。pixel encoder0/macro encoder0/AD0/physics0，完全阻塞像素/macro/low/fullbaseline路径。新v2宏kernel未获认证，不在本driver调用。

HOST CUDA only；无CPU neural fallback。需进程启动前`CUBLAS_WORKSPACE_CONFIG=:4096:8`，run打开deterministic algorithms并关闭matmul/cuDNN TF32，记录torch/CUDA版本与device name。CPU只做NPY读写/hash和保存数组算术，不执行模型。模型全425项before/after必须与原export value SHA逐项强等。任意部分失败保存report+已完成数组/attempt，strict拒绝partial。没有重试/补padding/删负结果。

工程上限bundle32MiB、读取单保存数组1MiB、余量200MiB、run wall600s在各dispatch边界检查（不是能中断单个卡死CUDA调用的系统watchdog；MAIN可外层timeout）。export验证独立wall300s由冻结exporter执行。write均O_EXCL/NOFOLLOW/fsync，仅自己新目录。CUDA分配/模块生命周期进程退出释放，不重置设备。saved consumer不做CPU神经复算，只核全数组/固定输入配方/精确输出配对、权重与来源/设备/真实meter一致，不能单凭JSON证明执行历史。

## MAIN完整命令

cwd `/home/gbk/code/LeWorldModel/le-wm-main`。prepare的实际返回提供manifest SHA；run返回report SHA，后续参数必须填该实际值，不能预造。prepare会真实重新verify，而非复用终端文字。

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_hilewm_trusted_cuda_validation_20261028_v1.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_hilewm_trusted_cuda_validation_20261028_v1 prepare --output results/agent_hilewm_trusted_cuda_validation_20261028_v1_prepare_MAIN01 --main-authorized
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_hilewm_trusted_cuda_validation_20261028_v1 verify --manifest results/agent_hilewm_trusted_cuda_validation_20261028_v1_prepare_MAIN01/manifest.json --manifest-sha256 "${HILEWM_VALIDATION_PREPARE_SHA:?MAIN actual prepare SHA required}"
CUBLAS_WORKSPACE_CONFIG=:4096:8 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_hilewm_trusted_cuda_validation_20261028_v1 run --manifest results/agent_hilewm_trusted_cuda_validation_20261028_v1_prepare_MAIN01/manifest.json --manifest-sha256 "${HILEWM_VALIDATION_PREPARE_SHA:?MAIN actual prepare SHA required}" --output results/agent_hilewm_trusted_cuda_validation_20261028_v1_cuda_MAIN01 --main-authorized
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_hilewm_trusted_cuda_validation_20261028_v1 consume --manifest results/agent_hilewm_trusted_cuda_validation_20261028_v1_prepare_MAIN01/manifest.json --manifest-sha256 "${HILEWM_VALIDATION_PREPARE_SHA:?MAIN actual prepare SHA required}" --report results/agent_hilewm_trusted_cuda_validation_20261028_v1_cuda_MAIN01/report.json --report-sha256 "${HILEWM_VALIDATION_REPORT_SHA:?MAIN actual report SHA required}"
```

不依赖尚未提供的未来receipt SHA。尚未worker实际prepare/CUDA；RAM测试注入exporter/authority边界的synthetic receipt与synthetic输出只检程序，绝非科学证据。即使MAIN四项成功，官方模型/历史ViT/冻结raw normalizer/public RGB-latent/low-high语义配对等仍独立未闭合，`official_original_model_parity=false, baseline_runnable=false, scientific_admission=false`。本轮仅安全装载与owned-source执行工程前置，不把环境或kernel变化冒充独立架构认证。
