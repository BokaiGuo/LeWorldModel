# branch0 T1 → T2 HOSTCUDA ABI probe

仅工程探针：固定原branch0/root0/off。不是656实验，不是history因果修复，不开放P1/P5门；没有label、MSE、选优或物理反事实。本工作位只实现和fixture测试；MAIN审查后运行唯一生产探针。

## 实际来源与权限

ROOT为`/home/gbk/code/LeWorldModel/le-wm-main`。

固定catalog为`results/agent_p5_short_start_native_abi_20261029_v1_catalog_MAIN01.json`，SHA `1c49b5eba91619a6798a9d387c524b6e7d8658c196a9ac75f1ea9a65226c2678`。prepare要求其164行、设计、authority和branch0记录与原H2 report重新解析完全一致，并实际读取/交叉核验六份NPY；不运行上游fullstrict。

原H2报告`results/agent_p5_raw164_jepa_h2_cuda_20261029_v2_cuda_MAIN01/report.json`，SHA `e5a561215bfe7b6f1c1c224e7661211e3f237fb16458a6a409f5a1da0fc08579`。原prepare SHA `09070780c3ce782a71ae01492e0c4b801de976e9a36cdf9ee1bf1a53be98e924`；原strict SHA `783b65f67899791b5697bfe0819ccfb3231d57212b9e3e85aefc02419459c0ea`。其路径由固定stem的`_prepare_MAIN01.json`、`_consume_MAIN01.json`确定。

具体抽取ABI复用冻结`agent_p5_short_start_native_abi_20261029_v1`：原call2三输入`[:,0:1].copy(order=C)`，使用原encoder/proprio_encoder真实输出交叉核验，原normalizer重算校验不改输入。actual0 visual/action/proprio rawbytes SHA依次严格要求：

```
d6f4eae94431f1926df413adec5ecc34597dba54002a46dfc773f362f5efbf01
cd767833f90a393ee0ecc5422cf72c71caee1d7c1f5667a03048dd7f6ecb1d84
3058e78213d6cbfd85ec89ddae80045a1b53cf13aac71f071551d0ca1b9f3743
```

这些是数组rawbytes，不是NPY文件SHA。NPY文件SHA/shape/size/dtype/EOF分别完整绑定。保留原生agent position/velocity特权输入，不称RGB-only/跨模型公平同输入。原全3槽H2的早期输出不是本次因果短启动输出；不以其替代新pred5。

## 两次真实调用与守卫

T1输入visual`(1,1,1,16,16,384)`、action`(1,1,10)`、proprio`(1,1,256,16)`，预测5；T2对应T=2，历史是actual0与本次已保存pred5的visual及proprio，两者均滚入，不访问actual5。action原packet0/1分别对应raw1..5/6..10。输出完整tuple为visual`(1,T,256,384)`、None、proprio`(1,T,256,16)`，保存所有槽而非只末槽。

仅使用原bridge `load_host_cuda`加载冻结模型，复用原model_receipt，实际加载模型hash及strict schema必须与H2相等；结束再次比较。新局部hooks覆盖三模型但只允许predictor T1再T2，任何encoder/重复/CPU/float64/梯度调用拒绝。不修改旧full3guard，不全局monkeypatch。所有实际导入app/src/dinov2源及loader源绑定前后核SHA。

TF32关闭、inference、请求MATH。原native RoPE内部可能使用ALL_SDPA_BACKENDS覆盖外层MATH，故额外使用局部TorchDispatchMode记录attention op并拒绝观察到的flash/efficient/cudnn attention。不是仅以外层context宣称底层已认证MATH；若原路径触发拒绝，保留partial，不绕开守卫或改模型。T1/T2真实native支持尚待MAIN HOSTCUDA验证。

## 资源与durability

总2predict API、2batch samples、3token-frame samples；0新encoder/proprio_encoder/AD/physics/training。输入输出effective tensor bytes固定2,457,720（10个完整NPY），tensor cap4MiB，全部输出cap16MiB，磁盘floor200MiB，终态预留2MiB，初始CUDA free floor4GiB。工程wall300s从verify前开始，包括加载/调用/终态模型和源hash；不是科学预算或等FLOPs声明。

Store独占创建，fsync文件/目录，JSON恰好一LF，journal规范化hashchain。每次dispatch intent → full inputs → observed return → validated full outputs → stage seal；第一个stage已保存后才构建T2；最后两预测seal。attempted、实际observed returned、validated returned独立记录，坏返回仍计真实return。失败不重试，保留两槽及partial全部成本，不能作为理论负结果。

输出目录含execution_binding.json、journal.jsonl、tensor_000.npy至tensor_009.npy、prediction_seal.json、report.json（partial不保证完整tensor或seal）。report包含完整calls及真实CUDA描述符、stages、加载模型前后receipt、actual imported sources、memory、attention ops、实际elapsed/bytes、errors/source_errors、inventory和journal tail。

saved consume不调用模型或上游fullstrict；重建本prepare并检查整目录hash/唯一顺序NPY、完整IO、因果pred5拼接、真实descriptor守卫、两dispatch实际event字段/封存顺序、总字节/模型/源。complete status为`branch0_T1_T2_saved_native_ABI_verified`；失败为`blocked_partial_probe`并exit2，不认证短启动成功。

## MAIN CLI

在ROOT执行，实际SHA由每步产生，不填写未来SHA。输出必须不存在。

```bash
M=experiments.spectral_reachability.agent_p5_short_start_cuda_probe_20261029_v1
P=results/agent_p5_short_start_cuda_probe_20261029_v1_prepare_MAIN01.json
O=results/agent_p5_short_start_cuda_probe_20261029_v1_cuda_MAIN01
.venv/bin/python -m "$M" prepare --output "$P"
.venv/bin/python -m "$M" verify --prepared "$P" --prepare-sha ACTUAL_PREPARE_SHA
.venv/bin/python -m "$M" run --prepared "$P" --prepare-sha ACTUAL_PREPARE_SHA --output "$O" --authorization MAIN_P5_SHORT_START_T1_T2_HOSTCUDA_ONLY
.venv/bin/python -m "$M" consume --output "$O" --report-sha ACTUAL_REPORT_SHA --receipt results/agent_p5_short_start_cuda_probe_20261029_v1_consume_MAIN01.json
.venv/bin/python -m pytest -q tests/test_agent_p5_short_start_cuda_probe_20261029_v1.py tests/test_agent_p5_short_start_native_abi_20261029_v1.py
```

实现完成条件仅是三新文件及fixture闭环。已有goal属于其他旧目标，未覆盖/假标完成。未运行actual prepare/CUDA/physics/生产consume，不把测试通过等同科学通过。
