# Reacher 候选资产原生入口预检

本地原配置请求 `dmc/reacher_random.h5`，默认缓存未发现。另有工作区内 `data/reacher_official_source_20261024/extracted/reacher.h5`，HDF5 元数据可读，包含原生评测必需的 action、qpos、qvel、pixels 和 episode 时钟；历史 manifest 声明镜像 repository `quentinll/lewm-reacher` 及 SHA256，但本次尚未重算 93 GiB 文件或验证与 checkpoint、原配置数据修订一致。

以该文件的绝对不含扩展名路径覆盖 `eval.dataset_name`，其余保持本地 Reacher 入口，首先单任务、4×2 样本与 2 elite 的 smoke；这是候选资产和接口检查，不是原标准 benchmark。记录原始错误，不自动下载或覆盖其他数据。仅在 smoke 通过后才能考虑运行 50 任务原预算，且仍须明确数据和 checkpoint 来源未认证。
