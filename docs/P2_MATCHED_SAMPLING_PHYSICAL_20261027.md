# P2 matched sampling：已封存八个H2选择的真实执行器

只新增 `p2_matched_sampling_physical_20261027_v1.py`、同名测试及本文；不修改
任何既有sealed源码/测试/报告/中央文档。不重新优化、不选根、不训练/readout。
所有后续只作工程校准，P1/P2仍未准入；Q可奖励agent运动，不能声称任务效益修复。

## 输入与验证

消费 `results/p2_matched_sampling_20261027_v1_cuda_v1` 下已完成report及prepared。
调用原已封存 sampling 模块 `load_prepared`，验证所有输入与source seals、旧P2
比较来源；调用 `selected_actions` 校验全部四根、cem/random两arm、各40候选、
80 planning+4 replay、独立完整witness、context/动作域/goal metric/provenance。
物理driver另把planning report/protocol/prepared以及bank/upstream hash绑定并封存。

核对官方root fingerprint、原goal RGB、三帧history整栈hash（不是逐帧列表），
planning witness中的root/goal RGB必须与官方参考相同；每条branch前reset/warmup
并重新检查RGB。root/goal不符时该branch保留失败、零动作，不用替代根/动作。
每根只执行一条CEM和一条random：根0/2 CEM先，根1/3 random先，不按结果变顺序。

## 真实执行与持久化顺序

直接复用已冻结P3物理driver的 `physical_branch`、guarded execution、fsync日志、
PNG/JSON独占写入及encode_saved；新wrapper只接受H2恰10个bounded raw commands。
无H3、无动作补零、无额外replay、无重试或重新优化。P3兼容plan字段
`corrected_search_horizon=2`只是共享持久化helper的元数据，不涉及P3选择或修正。

每一步先fsync attempt，再调用env.step；返回后fsync step_returned，再记录RGB/
state/guard。失败step也保留attempt成本。每个初图/观察图和branch `.physical.json`
在结束时持久化。全部八个census slots存在，包括setup失败blocked项。
guard不可见、环境终止都保留：计划80步，实际可能少于80，不能当固定时域效益。
warmup/reset成本与选中轨迹step计数分开，不宣称总物理预算匹配。

八branch及 `physical_report.json` 写完后，才允许可选CUDA allocation/model加载。
默认不加载神经模型，不调用predict/encode，不是CPU神经fallback。启用编码时只
读取hash匹配的已保存终图，官方冻结eval主机CUDA encode；predict被禁止。编码失败
逐branch写独立encoding receipt，绝不重写已保存物理trace/JSON。CUDA setup失败
为八项保存零attempt encoding失败记录；已有物理执行仍可审计。
始终输出终报告及input/source/model完整性gates；完整性失败exit非0，产物保留。
fatal磁盘写入/进程被kill不能保证最终report，但已fsync事件不因后续编码丢失。

## 主线程命令

cwd `/home/gbk/code/LeWorldModel/le-wm-main`。非执行预检：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  -m experiments.spectral_reachability.p2_matched_sampling_physical_20261027_v1 \
  --planning results/p2_matched_sampling_20261027_v1_cuda_v1 \
  --output results/p2_matched_sampling_physical_20261027_v1_physics_v1 \
  --preflight-only
```

默认模型无关物理执行（不占用GPU、不神经推理）：

```bash
SDL_VIDEODRIVER=dummy PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  -m experiments.spectral_reachability.p2_matched_sampling_physical_20261027_v1 \
  --planning results/p2_matched_sampling_20261027_v1_cuda_v1 \
  --output results/p2_matched_sampling_physical_20261027_v1_physics_v1
```

若主线程需要终图CUDA编码，**选择这一条而非再执行上一条**，使用全新目录：

```bash
SDL_VIDEODRIVER=dummy PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  -m experiments.spectral_reachability.p2_matched_sampling_physical_20261027_v1 \
  --planning results/p2_matched_sampling_20261027_v1_cuda_v1 \
  --output results/p2_matched_sampling_physical_20261027_v1_cuda_v1 \
  --encode-terminals
```

输出目录必须全新且含TAG，已有目录拒绝覆盖。本任务只交付/测试，不自行执行真实
轨迹或GPU。预计默认<20MB（至多88张224PNG和小型源码seal），无权重复制；
可选编码至多8samples/batch1，暂预留GPU4GiB/10分钟为保守估计、非实测。

非神经测试：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_p2_matched_sampling_physical_20261027_v1.py
```

测试读取真实已完成CUDAreport核验八选择，另以metadata/数组fixture测试guard/
失败/日志/落盘先于编码/默认禁止神经入口。它们不是本driver真实轨迹或收益证据。
