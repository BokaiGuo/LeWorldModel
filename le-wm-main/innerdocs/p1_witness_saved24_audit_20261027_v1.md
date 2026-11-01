# P1实际24帧saved raw-latent复算

新增 `experiments/spectral_reachability/p1_witness_saved24_audit_20261027_v1.py` 与 `tests/test_p1_witness_saved24_audit_20261027_v1.py`。只读取MAIN已执行的CUDA latent192、protocol、PNG与物理checkpoint，不加载/运行模型或环境，不改封存字节。

## 绑定与独立复算

实际CUDA目录 `results/p1_witness_rgb_cuda_main_20261027_v1_cuda_main01`：report SHA `255acbd9e7bdc66df8fced64120a13787520a821832a28a9c2fe6b5e4ed60da6`，analysis SHA `4f37ac7744d552e570b799ab4256c3a63ab0bea8ebe368a5a9d83d487a84bab0`，image journal SHA `d5350184c19207ef4c3cbdd30af02c3303a335436b6bddffc3fd74ec4c505c9b`。prepare `results/p1_witness_rgb_cuda_main_20261027_v1_prepared01/prepare.json` SHA `ce6b4971fe748718b4e685478c065d5b3ac2ca4a6f08a3d1d9503f555ed54d3f`。

调用冻结producer的纯verify/collect作source/PNG/原选取绑定（不调用analyze/run），另独立从每个frame的5条event核对encoder/projector attempt/completed、raw latent、pixelhash与root/arm/step。按二次型 `e.T@e/192`重算原Q predicted/actual，非投影native prediction MSE，按稳定身份join原analysis及实际物理24checkpoint。全部误差容限2e-12，无新Q。

全分母2root×4arm×3checkpoint，无initial去重：24encoder+24projector，0predict/AD/新physics，model hash前后相等。每root四臂initial RGB及latent均逐位完全一致，最大latent差0。两个initial重复组仍各计算4次，不将重复样本包装成独立root。

## 实际终点结果

以下改善均为 comparator减two_stage，正数表示two_stage原Q更低；仅描述同一批校准根，不能与pose单位混算。

|root|comparator|predicted Q改善|actual Q改善|
|---|---|---:|---:|
|0|zero|0.1859785940|0.1031163839|
|0|full H2 GN|-0.0398024369|-0.0126955113|
|0|full H2 gradient|0.0780459653|0.0024353145|
|1|zero|0.4708133679|0.3515278381|
|1|full H2 GN|0.2477873260|0.3636759782|
|1|full H2 gradient|0.1102151498|0.1187245634|

latent-Q改善不是官方成功或object pose改善。已知物理root0 two_stage物体位置误差较zero恶化约6.22149，root1 two_stage物体未改变；不得据Q优势称物理机制成功。原8分支success0/8，P1false，不是独立确认/等搜索成本比较，不调动作/参数。raw latent真实性依赖冻结实际CUDA收据，本审计没有独立重新编码。

## 测试与命令

14 tests通过：真实保存数组/完整绑定复算，以及纯接口缺失、错root、错pixel、counts、nonfinite、shape、重复、Q缩放等测试；不是CPU神经测试。

cwd `/home/gbk/code/LeWorldModel/le-wm-main`：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_p1_witness_saved24_audit_20261027_v1.py
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.spectral_reachability.p1_witness_saved24_audit_20261027_v1 --output results/p1_witness_saved24_audit_20261027_v1_audit02.json
```

已执行报告 `results/p1_witness_saved24_audit_20261027_v1_audit01.json`，约10KiB，SHA `23536377645defbb81bdac3cf16deb33040c677fe3163ab9d70223bda5a94914`；source SHA `1fb645c2c3d8b91dde823fd9120097dc8c4b2711ed375102195e96f5c8f47df6`。CLI拒绝旧输出，≤512KiB且保留200MiB。后续仍需独立fidelity/heldout及P1科学准入证据，不从这24个相同根图像外推。
