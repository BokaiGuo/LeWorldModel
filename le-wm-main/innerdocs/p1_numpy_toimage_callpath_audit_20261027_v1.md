# P1 NumPy / ToImage 实际入口审计（独立新增，CPU-only）

目的：区分已复现的 stable_pretraining NumPy HWC→CWH 缺陷与当前项目真实输入路径。不修改旧 seal，不做训练、神经推理、物理运行、联网、根/阈值/动作或 outcome 选择，P1=false。

## 预先固定的范围与检查

静态扫描项目所有一方 Python 文件（含根目录、scripts、experiments），排除 external、results、data、tests、虚拟环境、缓存、符号链接。解析显式调用及 import alias，列出 ToImage/to_image/get_img_preprocessor 与 transform factory import edges；未知调用路径拒绝通过。记录全部扫描文件 hash。不是任意动态 import/getattr、第三方插件或历史版本的全程序证明。

逐一绑定当前训练、directional、latent、两个跨环境入口，以及九个外部视频审计入口。HDF5/Lance reader 在 transform 前转 tensor；外部视频 loader 返回 TCHW；跨环境审计显式 torch.from_numpy/permute。Reacher repair 的 stable factory 仅供参考测试，生产 dataset transform=None、手工 tensor 预处理；检查无显式生产 importer。

五个在线 factory：eval、PushT、Reacher、Cube、TwoRoom。仅提取这些实际源码函数和 installed BasePolicy._prepare_info 的 AST，在 dummy self 上运行 CPU 图像变换，绝不 import planner 模块。源头 NumPy 输入为 [2,3,H,W,3]，pixels 与 goal 各六帧。spy 要求到达 transform 的是 CHW tv_tensors.Image，像素/轴顺序逐值相等；输出必须逐值等于显式 CHW 参考。同时验证这些 torchvision factory 的直接 NumPy 分支也正确。

图像固定为原封存 root1 三帧（协议 hash 绑定），加一个 5×7 非对称 uint8 轴探针。此探针仅用于预处理类型/轴回归，不充作模型或物理证据。总计 20 image/factory pairs、240 policy transform calls；没有模型 frame encode。stable_pretraining NumPy/PIL/tensor 三分支阳性对照独立执行，不把两个同名 ToImage 混淆。

封存 feedback 的实际路径单独检查：p4.encode_cuda → p0._frames_tensor → p0._image_tensor → run_pusht_counterfactual.image_transform。这里不经过 BasePolicy，也没有 ToImage；先 NumPy→CHW tensor。仅提取两个预处理函数，在 CPU 上对同四图各检查 single 与三帧 context，共 16 次图像 transform；不会调用 encode_cuda。全部必须逐值等于当前正确 CHW reference。

本地 checkpoint 只读 README/config/weights 与 HF download metadata：验证 SHA256/LFS 或 gitblob-SHA1、共同 revision。记录本地文件清单、完整 model card/config；不会 unpickle weights，也不读取整个数据集。dataset tag 与当前源码不是实际训练 run/normalizer/camera/codec 来源证明。缓存 metadata 不是联网认证 publisher。未找到权威训练导出不能倒推出 checkpoint 有 bug，也不授权替换 PUBLIC_ACTION stats。

## 冻结与执行

执行前创建独占新结果目录，写 protocol、checkpoint receipt，封存本模块/tests/doc、关键源码与两个 receipt。protocol 通过 input_manifest.json 的 SHA256 绑定完整静态扫描输入及原 RGB，避免完整清单超过 seal 工具单文件 1MiB 限额；完成时重新验证所有 hash 与 source seal。仅新增本版本文件，旧封存内容保持不变。若主线程同时改动绑定源码，审计必须失败，不能静默忽略。

```bash
.venv/bin/python -m pytest -q tests/test_p1_numpy_toimage_callpath_audit_20261027_v1.py
.venv/bin/python -m experiments.spectral_reachability.p1_numpy_toimage_callpath_audit_20261027_v1 --out-dir results/p1_numpy_toimage_callpath_audit_20261027_v1_offline_v2
```

若入口没有实际进入有缺陷分支，不创建生产修复或把潜在代码异味宣传成已证 P1 根因。只有证实在线受影响才值得另建最小修复版本。当前审计不声称代码全排除、理论 negative 或物理收益；实际数值/计数以新 report.json 为准。

本版本非神经测试 15 项；首次测试发现审计自身应显式 import torchvision.transforms.v2，已在新模块内修正，不涉及生产或旧封存文件。

offline_v1 为冻结阶段大清单超限的失败目录，无 report、无神经或物理运行；保留不覆盖。正式离线结果使用 offline_v2。
