# 固定四词空间 patch saved 诊断

新独占 source/tests/doc；v5及所有旧资产不改。只读实际v5 manifest/report/receipt，经原full consume返回值精确相等才分析。绑定实际report `9505f332fc6c1cbb670d9a77f46290f8d2822486b00d89006ea038359898bdc5`、receipt `33f3f24735131fd8b569c1d0a87068d1b3f4e26dab31618a4afb90005cf8c927`、prepare `79c3c2f1aa1b8f1b9223e8d87f22986f5c3437144e10bbe78a6cc556093beafd`。读取sealed goal、四个真实terminal token与原预测token、原RGB；不加载权重、运行神经/物理、不拟合Q、不遮挡、无winner。

## 固定分母与分解

唯一已知构造OOD根，全四词xy/yx/x_minus_x/zero_zero，各256×384 full token。224×224 RGB与patch14对应16×16位置网格。封存DINO `patch_embed.forward` 的conv后flatten(2).transpose保证行优先；`x_norm_patchtokens`去除CLS/register，保留该序。源文件及原transform进入SHA闭包，但不import外部模型。Transformer全局注意力使位置不等于局部物体因果支持；相同坐标原始RGB并列不是感受野证据。原无增强224正方形预处理保持网格；不重新预处理/编码。

设P=原预测终态、R=真实终态encoder、G=已编码goal。每patch按384通道float64均值生成：(P-G)^2、(R-G)^2、(R-P)^2和有符号交叉项2(P-G)(R-P)，满足real_goal=predicted_goal+prediction_real+cross。交叉项不可截0、不可称独立失败原因；输出每patch代数残差。完整mean与v5封存三项指标复算一致才输出。此为误差恒等式，不能量化“多少任务收益由模型导致”。

每patch对原全图metric贡献=patch通道均值/256。固定全16个4×4-patch tile（各56×56像素）完整覆盖，不按结果筛区域。每tile输出对原全图的贡献和完整索引，不在tile内重新归一化后冒充全图贡献。xy−yx及cancel−zero固定两对输出所有patch差值、真实feature pair距离、RGB pair距离；没有排序或选词。

原RGB独立输出uint8平方像素差均值；不能与latent MSE相加或做大小比较。SVG包括lossless原goal/四终态RGB和所有patch的三项非负feature误差及RGB误差。每列使用四词共同最大值，仅用于显示，JSON完整保留数值；不将图像强度当统一跨单位量纲。signed cross与差值完整保留JSON，不裁掉负数。原agent/goal raster/block混杂不做语义分割、不凭颜色阈值归因；semantic_regions和causal_attribution均null。输出不包含winner。新神经API/样本/physics全0；复用v5 meter单列不重复退费，不宣称总算力相等。科学准入false，不是fresh、跨模型闭环、C2或任务成功。

## 可执行命令

cwd `/home/gbk/code/LeWorldModel/le-wm-main`；指定venv，解释器依赖漂移另查，禁止用系统pytest差异推断科学证据漂移。测试包含真实saved full consume与CLI analyze→verify、RAM恒等式/坐标/失败突变；RAM不是科学证据。

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p5_spatial_patch_attribution_20261028_v1.py

OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p5_spatial_patch_attribution_20261028_v1 analyze \
 --manifest results/agent_p5_terminal_goal_attribution_20261028_v5_prepare_MAIN01.json \
 --producer-report results/agent_p5_terminal_goal_attribution_20261028_v5_cuda_MAIN01/report.json \
 --receipt results/agent_p5_terminal_goal_attribution_20261028_v5_consume_MAIN01.json \
 --report results/agent_p5_spatial_patch_attribution_20261028_v1_MAIN01.json \
 --svg results/agent_p5_spatial_patch_attribution_20261028_v1_MAIN01.svg

OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p5_spatial_patch_attribution_20261028_v1 verify \
 --report results/agent_p5_spatial_patch_attribution_20261028_v1_MAIN01.json \
 --svg results/agent_p5_spatial_patch_attribution_20261028_v1_MAIN01.svg
```

8MiB合计输出cap，独占xb/x不覆盖；失败部分文件保留，不补零或自动重试。saved verify重建完整JSON/SVG，而非只检查status；MAIN可再封SHA。无prepare/CUDA/physics入口，因为已有全部saved前置。worker可运行RAM及saved临时目录回归，不写MAIN结果。

## 下一合法 goal metric 错配检验协议（尚未执行）

不以当前zero选择先验判理论失败，也不根据本图改ROI/权重/阈值。此根只能回顾诊断。下一独立trial在读取结果前固定public goal/context、四词和全图原评分，完整保留分母；冻结模型与metric，不拟合latent→coverage。先分别检验：真实encoder→goal的原metric和原native几何coverage的四词对比是否方向一致；预测→goal排序是否偏离真实encoder→goal。前者不一致提示metric/可视混杂或native目标定义差异，后者提示预测误差；两者可能同时存在，均不自动推出因果机制。

要进一步区分agent/goal raster混杂，须MAIN另审新原生renderer校准协议：同一合法完整物理状态、不推进Space，预先固定agent显示有/无、goal raster有/无的完整2×2渲染控制，全部状态/相机/目标几何/输出身份绑定。只能诊断渲染敏感性，不将遮挡后的token用于当前planner或据结果挑region；新增encoder预算需预声明并MAIN CUDA执行。若原renderer没有合法不变状态toggle，则阻塞于该接口认证，不能本模块用像素涂抹替代。原生task goal语义与coverage/done分离仍需保持，不能把coverage当done或新增成功门。当前不创建这些渲染、资产或未来证书。
