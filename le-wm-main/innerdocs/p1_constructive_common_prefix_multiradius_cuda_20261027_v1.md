# Fixed model-only multiradius capability response v1

协议在新增半径输出前冻结。独占本stem源码/tests/doc；不修改v1/v2 common-prefix或physical封存文件，不承担独立代码审计，不运行GPU/物理。使用experiment-code技能的预声明协议、逐调用日志和RAM/mock故障测试；不做训练/选择/按结果改参或额外绘图。

## 固定实验

输入仅获批common-prefix CUDA v2 actual report SHA `73959e915161c3e756068bc30606c7c14a84c9806b52d47ca918b1dbcf598b46`及其prepare；源码SHA `bbcc1d1c1dffab6f956fa9e7371217a31a642293ccbb43bb7328d23eff7a83f2`。从容器白名单提取4个原MATH observed context、原model-only21预测以及批准旧blindtask；**不访问report.analysis、terminal10编码images[8:]或物理响应指标**。report容器包含label，但程序仅取上述model字段；测试显式禁止访问terminal索引。上游封存bindings只hash，不解析sourcepack/certificate/physical结果，不重新编码或选位置。

4positions/2knownroots保持原顺序/roles（G=D）；每位置相同3state/2action context固定。方向固定为同root原批准task `goal-original_root`，不使用ownprefix残差、新半径输出或物理u。原identityQ/goal/动作normalizer/model不变。固定ρ=`.0625,.125,.25,.5`，各10坐标×±，全位置共享各自单独付费的zero一次：**1+4×20=81H1/position；4×81=324predict+324actionencoder**。0encoder/projector/AD/J/VJP/SVD/solve/训练/选择/physics。不额外nominal/replay或padding，不根据曲线删半径/位置/坐标。

顺序position0..3、zero、ρ递增、j=0..9先+后−。logit只一坐标±ρ，一次NumPy float32 tanh、time-major5×2、原`.75sqrt5`球及raw[-1,1]，不clip/重中心化。`.5`仍合法，不改变原球。原MATH/cudnnTF32False/matmulTF32False、batch1 float32/frozen eval/CUDA-only，模型tensorhash前后一致。

每位置zero及ρ=.25的20点要求与v2保存**完整successor context和float32 bits exact**，共84个接线复现点已包含在324新calls；其余240为新增半径响应。reference输出仅用于复现gate，不能替代新forward。输出先durable保存后判gate，失败保留raw、标reference_mismatch、停止剩余dispatch；不放宽容差/重试，不把失败行作为完整response混入算术。

## 输出与资源

每位置/半径192×10中心secant `B=(z+−z−)/(2ρ)`、even项`E=z+ +z−−2z0`和`E/ρ²`、共同方向raw/unit signed列、signed_l2、B Frobenius、相对最小半径B变化和cosine。保存全部324完整H1 successor contexts及每位置输入context。它们是**有限logit响应**，不是AD导数；tanh非线性本身也影响随ρ变化，不能把变化唯一归因模型失配。zero范数/cosine零范数为null，不补0；任何缺±列为null，保留完整324行/4半径/10列分母。

逐module actual attempt/completion、scope(position,probe)和全局324/每位置81 predispatch cap；返回且sync成功才completed，失败不退款不重试。视觉模块/AD受guard禁止。lossless gzip事件Z_SYNC_FLUSH+fsync、源/model/journal/hash终结分离捕获。总output cap16MiB/final reserve512KiB/floor200MiB，启动216MiB；prepare≤2MiB。完全I/O失败只保证已落盘字节，不保证终结report。继承上游96enc/84predict及更早成本单列，不声称总算力配对。

仅检查模型内部局部—有限响应稳定性；没有新增跨半径物理label，因此不能单独判定真实局部/有限失配或唯一根因。与既有ρ=.25 physical现象的解释须保留knownroot/有限干预边界。P1/P2=false，不能由G=D数据推出非贪心优势。

## MAIN命令

cwd `/home/gbk/code/LeWorldModel/le-wm-main`；实现者只测试/saved RAM读取，未创建prepare/results或运行GPU：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_p1_constructive_common_prefix_multiradius_cuda_20261027_v1.py
.venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_common_prefix_multiradius_cuda_20261027_v1 prepare --parent-report results/p1_constructive_common_prefix_probe_cuda_20261027_v2_cuda_main01/report.json --parent-sha 73959e915161c3e756068bc30606c7c14a84c9806b52d47ca918b1dbcf598b46 --output results/p1_constructive_common_prefix_multiradius_cuda_20261027_v1_prepare01.json
.venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_common_prefix_multiradius_cuda_20261027_v1 verify --prepared results/p1_constructive_common_prefix_multiradius_cuda_20261027_v1_prepare01.json --prepare-sha MAIN_PREPARE_SHA
.venv/bin/python -B -m experiments.spectral_reachability.p1_constructive_common_prefix_multiradius_cuda_20261027_v1 run --prepared results/p1_constructive_common_prefix_multiradius_cuda_20261027_v1_prepare01.json --prepare-sha MAIN_PREPARE_SHA --output results/p1_constructive_common_prefix_multiradius_cuda_20261027_v1_cuda_main01
```

所有输出路径须全新；MAIN review后独占HOST CUDA。mock不是新实际预测证据。
