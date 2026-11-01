# frame13 saved geometry/order 诊断 v1

仅处理 MAIN 明确指定的 `xy/frame_13`。原 renderer v1 失败回执 SHA `1b02ed4a2ca136b4dc7f8549073ff030620df8587d654eb2a07ca86036cad0c6` 保持失败；不改 renderer、native source、门槛或任何既存产物。无 env/model/pymunk import，无 Body/Space 构造，无物理、神经、网络或全85帧扫描。本版只新增 source/tests/本文及其独占 saved 诊断结果。

## 固定输入和四种排列

原 replay prepare SHA `c2ecf01381cf339bf8f7e89fa428f4f62ac46075b10458ee7cfc8deac661941d` 通过实际失败 report 的 prepare_sha256/prepare_digest 连接；原物理报告沿其 original_physical_report 与 bindings 连接。只读取其中原/重放 xy13 的 RGB/state/proprio/public snapshot/transition，逐文件 SHA/bytes，三 NPY 逐字节一致，全部公有快照除本次 coverage_arr[-1] 外全等；原 threshold 从 snapshot 读取，不能重拟合。

公共 snapshot 已把 shapes 按语义排序，不能恢复旧 set 的迭代顺序。故事前完整保留 `(goal order,block order)` 四格：`(01,01),(01,10),(10,01),(10,10)`。每格用原两 Polygon→MultiPolygon→intersection.area/goal.area→np.clip(coverage/threshold,0,1)，再用真实前12 coverage+本次候选重算 max_coverage。不是 union，也不调用 normalize/make_valid/buffer(0) 修复原几何。相接 T 两矩形可能使原 MultiPolygon 的 is_valid=false；原实现仍直接交给 intersection，本诊断记录 validity 而不换算法。

输出保存每格 goal/block顺序、交集 WKB、完整面积/coverage/reward/final/max及每个binary64的hex bits；两份真实观测四字段也保存bits。匹配使用 struct.pack 二进制全等，无容差、round、选最近值或删格。输出 conditional_exact_matches 保留全部命中组合。

## 变换精确性边界：不能把命中冒充证书

绑定 native env SHA `6ba34790b6c2ce017a76322c2f1c205ef97d4d3b08a620f844a57075c608a723`；已装 `_chipmunk.abi3.so` SHA `1224ec1d02916a7001d58e5a7c558f79a7941d3b6132cdfbe58b24665daf3bce`。用只读 objdump 保存五个实际符号的反汇编，不加载该 .so：

- cpBodyLocalToWorld 0x1c990：缓存矩阵0x88..0xb0；SSE2分别乘后加，`(c*y+a*x)+tx,(d*y+b*x)+ty`，无FMA。
- cpBodyGetPosition 0x1bfd0：同矩阵作用于(0,0)。本帧 tx/ty 正有限，+0不改变值，因此公有 position 是 local origin 的世界位置，而不是内部 CoG 世界 p，**不能再次减 CoG**。
- cpBodyGetAngle 0x1c6b0 返回0x70字段；cpBodyUpdatePosition 0x1b5f0 与 cpBodySetAngle 0x1c6c0 用 sincos 产生矩阵旋转系数，平移已包含 CoG 处理。

实现使用当前进程已解析的 `sincos`（ctypes.CDLL(None)），记录 dladdr 得到的库路径/字节SHA，按上述 SSE binary64 运算顺序重建顶点。不是 np.matrix/vector化归约，也不调用新 Body 来验证。但历史 snapshot **没有保存缓存矩阵、世界顶点或原 sincos 实际解析库/浮点环境证据**。从现有公有角度重算系数不能独立认证它们与历史缓存逐bit相同；当前库的 SHA 也不能倒推原加载字节。

因此所有结果只作为 **conditional** 计算，`exact_historical_transform_equivalence=null`、`native_transform_certified=false`、`legal_native_permutation_member=null` 固定保留，即使四字段全部命中也不置true。若未命中也不能据此排除真实原生排列可能性。它定位一个可复现的算术解释，不证明原运行真实迭代顺序、唯一原因、所有代码正确、整个85帧复现或P5通过。要升级精确变换认证，最小新增证据是**该旧帧实际 native matrix 或各形状实际 world vertices**及其原运行绑定；不能新运行补造旧缓存。

## 运行与消费

已装 Shapely 允许纯几何离线计算；读取实际 probe 的版本与已记录 imported dependency SHA，绑定当前实际 Shapely/GEOS 文件及当前 libm，但未记录的原系统库加载真实性仍unknown。运行不创建任何物理对象。2MiB独占JSON输出，排他创建，不覆盖失败回执。`verify` 全量重算结果和全部绑定相等，不意味着历史 transform 认证。

```bash
cd /home/gbk/code/LeWorldModel/le-wm-main
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p5_saved_geometry_order_20261028_v1.py
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p5_saved_geometry_order_20261028_v1 analyze --output results/agent_p5_saved_geometry_order_20261028_v1_saved_worker01.json
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_p5_saved_geometry_order_20261028_v1 verify --report results/agent_p5_saved_geometry_order_20261028_v1_saved_worker01.json --sha ACTUAL_REPORT_SHA
```

CLI analyze 打印真实SHA和conditional membership；verify回执固定 `saved_diagnostic_verified_not_native_certification`，renderer_v1_release=false、scientific_admission=false，0physics/0neural。实际结果在交接消息报告，不预填命中或成功。tests包含四格保全、1ULP拒绝、CoG误用、invalid native MultiPolygon不修复及命中仍unknown，测试不是原生反事实运行。
