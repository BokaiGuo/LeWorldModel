# E3 cell03 HOST：CPU 前置核查与主线程交接

范围为已知根 cell03（`r0.g0.h3.dDx`），不是 fresh 根或 Full D2P；本次没有启动 CUDA、加载模型或运行物理。现有 authority `results/e3_cell03_feedback_prepare_codex_20261030_v1_MAIN01.json` 的 SHA256 为 `1facc17c2bb088c5c2af1ce4f035bc1a0c2742ab3d81ccd9d2830382fedf76e4`。它已满足 chunked HOST 的 authority ABI，不需再套一次 authority wrapper；该 wrapper 用于扩展其他 identity authority 的源码绑定。本次实际对原 authority 做严格 source 复核及物理 setup 保存输入 join，均通过。

## 接线与不可替代的运行时门

`chunked_identity_host.run` → 固定 authority、模型同一对象的分块身份 → before module census → `guarded_episode` 来源/model before guard → 独立原生环境。setup 从原种子重建 reset/placement、10 个 warmup、原 5 步 prefix；逐位检查原历史、状态、快照、目标及实际 cp5 RGB/state/snapshot。原生早停立即保留，不补步。control5 已封存 first5 物理执行成功返回后，才从同环境返回图像编码并连接动作历史，执行边界 10/15 的 H3 GD；每个下一包只取该边界选择的 first5，Dx 新动作 y 必须为 float32 正零。不得用保存中的预测来填补实际 RGB 或缺失终态。初始原生成功时停止，不人为派发动作。guard 后快照与 census、cleanup、独占 publication 及 saved inspect 分开执行。

实际 CPU saved join：root SHA `a3bd09470de75652506544b8bea0ff3c5fbae0c0d3b28775da47ee3e1adbb6f7`、cp5 像素 SHA `5b5f3fb1b83fa832bbbc663d9eb2efcb4171b5d093090a4296e172f3deb2e327`、H3/Dx、封存 first5 的 y 为正零。两个原 native branch 的 c0/c5 一致性和目标/RGB 由原 setup resolver 检查；这里只是 saved-input join，原 prefix 和 cp5 必须在实际环境再次重建。

## 预算及注意事项

单 job 原 Store 上限 512 MiB、最终 reserve 64 MiB、运行磁盘 floor 200 MiB，创建前要求空闲至少 712 MiB，含 authority 检验的 wall 上限 1800 秒；数值 backend 的实际峰值 guard 为 allocated 3 GiB / reserved 4 GiB。原 native 计量 caps 为 constructor 1、reset 2、set_state 4、goal_setter 3、goal_render 2、goal_bind 1、public_step 30、physics_step 304、render 32、close 1；失败尝试不退款。额外的两个 polygon 顶点读取另记事件，不是 render/physics。两个后续节点各最多 204 次 H3 predict 和 32 次 scalar reverse；control5 节点的 204/32 已在先前数值 job 支付，不是本次额外运行。早停实际成本更少，但保留未达边界。

原预算计算器对三节点的保守分区合计 475539776 bytes、余 61331136 bytes，明确 `physical_and_metadata_fit_actual_writer_unproven=true` 和 `full_production_transport_proven=false`；不能将算术余额当作真实整链一定装得下的证明。control5 旧数值 job 的 GPU peak 仅为历史单节点观测，不能证明新 episode 全链峰值。2026-10-30 当前文件系统报告 22G available、内存 26Gi available；这会变化，启动前重查，且由主线程协调 P3 并发和资源。renderer、native 依赖、设备可用性、source guard 在真实启动时仍需验证。已有 candidate closure 是有界 module census，不是全执行轨迹/系统原生库闭包；并发修改 P3 被绑定源码会触发拒绝，不能放松来源守卫。

## 可复验命令

从 `/home/gbk/code/LeWorldModel/le-wm-main` 运行 CPU 检查（约需重哈希原来源；只读、没有创建运行目录）：

```sh
env CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.e3_cell03_host_cpu_precheck_codex_20261030_v1 --authority /home/gbk/code/LeWorldModel/le-wm-main/results/e3_cell03_feedback_prepare_codex_20261030_v1_MAIN01.json --authority-sha256 1facc17c2bb088c5c2af1ce4f035bc1a0c2742ab3d81ccd9d2830382fedf76e4
env CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/pytest -q -p no:cacheprovider tests/test_e3_cell03_host_cpu_precheck_codex_20261030_v1.py tests/test_e3_cell03_feedback_prepare_codex_20261030_v1.py
```

主线程协调 P3 并明确授权后，下面是未来 HOST 入口，**本次未执行；字符串是代码合同，不是授权本身**。运行前重做 CPU 预检、确认输出路径不存在、重查磁盘/GPU、环境与来源未漂移；若失败保留 partial，不覆盖重跑或改门槛：

```sh
env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.spectral_reachability.agent_e3_chunked_identity_host_20261029_v1 run --authority /home/gbk/code/LeWorldModel/le-wm-main/results/e3_cell03_feedback_prepare_codex_20261030_v1_MAIN01.json --authority-sha256 1facc17c2bb088c5c2af1ce4f035bc1a0c2742ab3d81ccd9d2830382fedf76e4 --output /home/gbk/code/LeWorldModel/le-wm-main/results/agent_e3_chunked_identity_host_20261029_v1_CELL03_MAIN01 --authorization MAIN_E3_KNOWN16_CONTINUOUS_HOSTCUDA_ONLY
```

运行产生 report/completion 后，从实际输出取得两份 SHA，执行同一模块 `inspect --authority 上述路径 --authority-sha256 上述SHA --output 上述目录 --report-sha256 实际报告SHA --completion-sha256 实际完成SHA`。独立 saved inspect 和 native 语义审计并不等价 full saved strict，失败/partial 保持原样；真实物理/神经 parity、完整失败语义和科学准入仍未解决。`completed` 是出版完成，不自动意味着任务成功。

本次 CPU 回归：HOST、setup、dispatch、reset、native meter、publication 与 cell03 authority 共 51 passed；新 CPU precheck 与 cell03 authority 另 18 passed。均为 fixtures，不能证明真实环境或 GPU 路径。
