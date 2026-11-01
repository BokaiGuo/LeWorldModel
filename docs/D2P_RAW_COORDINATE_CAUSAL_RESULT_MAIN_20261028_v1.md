# 实际 cp5 十坐标对照：执行记录与归因边界

## 已完成：164 项主机 CUDA 模型侧对照

全部四条件、两个已知根，锚点为原suffix（observed recompile iterate0）。
不改目标、history、归一化、前缀或模型，不按物理结果选动作。

|产物|SHA256|
|---|---|
|FD prepare_MAIN01.json|a1a87327d230af7e02e2379909305c9fbbfb8a05768ba85eb4fcbdac85f95ef5|
|FD cuda_MAIN01/report.json|df21e0cd0a235a6e6f43572cc02614f8c62b4704bac2aabb8ec4e2886104aa15|
|FD consume_MAIN01.json|c2c454cf51b11e5f28df12e2c2f9ef12c8c7cd1eba7bf99b9a9cf9c97977e2b0|
|MAIN 独立原数组复算_MAIN01.json|934201dfed3ef067823224d23603b50747c1c122b812f72f4bfb6b32adf7a03b|

前三者前缀 `le-wm-main/results/agent_observed_raw_coordinate_fd_20261028_v1_`；
第四为 `le-wm-main/results/main_observed_raw_fd_recompute_20261028_v1_MAIN01.json`。
完整strict重建成功，164predict、0新AD/encoder/physics，errors=[]、模型和
buffers前后相同，总57.63s（含verify）。四个baseline与原iterate0全context一致。
两h共八个10维比较，FD/AD cosine最低0.9999997906、相对L2误差约
0.0001145–0.0006559。不能用该结果认证物理方向或理论成立。

MAIN独立复算所有80对scalar FD、80组192维secant、实际float32跨度、
原source gradient0与四baseline，全部与strict相同；其4项测试通过。
此复算绑定已执行完整strict的文件字节，没有在独立分析器内再次重跑整个上游。
worker新FD34测试独立通过，旧FD+MAIN scalar四文件联合37通过；计数不可相加
当成科学样本或总独立测试数。

## 实现中发现并修正的问题

新FD读取外层协议的路径错误、篡改测试修改只读数组，已在生产prepare前修复。
新物理driver还捕获future full10与cp5 incoming history混比，以及旧四分支
reset8计数沿用到164分支的问题，均在真实物理前修正。RAM fixture的旧输出
cap和终端bookend接缝已修复，最终MAIN完整29项回归通过；其开发失败不算
物理/科学负结果。

这些是新诊断实现接缝；没有证据表明它们导致此前P1的控制损害。

## 已完成：164 项真实物理与真实终态 CUDA 编码

物理三件套已冻结，MAIN独立29测试通过（276.91s），完整prepare118,598,947
bytes成功，SHA `47ee177e40cb1247a73e7d08bfe5f31ce5d50884a09a5e0bce909027a1b63bd1`。
物理164分支已主机完成，全部horizon10、prefix exact，3280public/33456physics/
10824PNG，errors=[]、157.06s（含verify）。报告SHA
`c4610c7f495137cbda7d70b147c74670db6f85339561da611ce5bee152d7bb10`，
完整strict消费已完成，回执SHA
`8df105747cf6835056d41ca1b1fd80f7f5b4170f7f527ec538213addc4c9d0bb`。
RGB三件套独立52项回归通过（190.58s），随后全部164真实终态已主机CUDA
编码并完成完整strict：164 encoder/164 projector、0 predict/AD/新物理，
errors=[]、模型未变，总162.55s（含verify）。

| RGB 产物（共同前缀 `agent_observed_raw_coordinate_rgb_20261028_v1_`） | SHA256 |
|---|---|
| prepare_MAIN01.json | cc39d01f5f23feeb65b243058141cbdf5489ac2967a584539952d8b36eba9410 |
| cuda_MAIN01/report.json | d13bd315a4cddf96ac2f53cb2d17e07c2baa1378142c144e9e61a0cd5c0d4dd7 |
| consume_MAIN01.json | d6b6ed08732d7595e740f98e0dc2450adf439e726aa3078854badb78f7d8b4b2 |

## MAIN 独立复算：偏移、动作响应、可观测性分开

`le-wm-main/results/main_raw_coordinate_response_20261028_v1_MAIN01.json`，SHA
`703142be3073336d76cd8cfb9cd8ef8fc41d0c3be0dd7141cc222a635e52c061`。
全部80对可用；34对模型/真实scalar目标方向严格反转，72对midpoint三项异号。
改以每个条件的同一baseline定义共同偏移后，45对共同项/响应项相消，
80/80均不满足精确动作无关偏移；不能把逐对midpoint当统一offset。
四baseline的cp10 state/snapshot/PNG均精确重现原完整执行分支。

| root/renderer | h | 模型/真实scalar secant cosine | 反转/10 | 状态变、RGB不变/10 |
|---|---:|---:|---:|---:|
| 0/off | .01 | −.918794 | 9 | 1 |
| 0/off | .001 | null（真实RGB响应全零） | 0 | 10 |
| 0/on | .01 | −.591894 | 6 | 2 |
| 0/on | .001 | −.729807 | 5 | 5 |
| 1/off | .01 | +.037232 | 5 | 0 |
| 1/off | .001 | −.599319 | 6 | 3 |
| 1/on | .01 | +.410182 | 3 | 0 |
| 1/on | .001 | +.442001 | 0 | 3 |

### 可观测性分栏修正（不改原封存分析或gate）

新增 `main_raw_observability_audit_20261028_v1.py`，从固定配对报告和真实物理报告
独立核逐对step10像素hash、完整native7和原始索引；未使用阈值、改h或挑根。
结果 `le-wm-main/results/main_raw_observability_audit_20261028_v1_MAIN01.json`，SHA
`1411a87e95d36ae84971b27e2bbaaed848fbc5c277e9675d591d374125614d3f`。

- 80对严格分为：24不可见响应、22同向、34反向；0 encoder不变但RGB变化、
  0实际目标tie、0模型目标tie。有效方向对56，其中34/56反向（描述性60.7%，
  非独立样本推断）。不能把24零响应算为方向一致或把null当cosine零。
- 24不可见对**全部有pose变化**，0仅速度变化；不是只因RGB不含速度。
  它认证本批渲染映射非单射，不单独认证是哪一处整数取整/遮挡造成。
- 相同RGB却不同latent为0，未见本批确定性编码重复失配。
- 原生方向另见 `main_raw_coordinate_response_20261028_v1_native_MAIN01.json`，SHA
  `5f0066a359781384d03a75ed4831fa165e84d437204eaf74abf903e6eec4cdaa`。
  八个logit-chain局部下降方向均改善jointXY、损害angle；未含L2投影，
  不是实际执行该方向的政策收益证据。

本次MAIN分析/配置溯源五文件联合43测试通过（1.90s）；其中新分栏13项，
与前述生产测试重叠不累加。代码修改修的是诊断中的证据分类与拒绝不一致输入，
不冒充已经修复历史P1控制损害。模型内AD正确、真实动作响应失配、像素不可见性
和latent/物理目标tradeoff并存，尚无唯一历史根因。

## 仍待解决

后续固定双统计量干预现已主机CUDA328预测及完整strict完成：原公开分支全164
完整context/objective逐位复现，archive统计反转34→36（2修正/4新增），不是充分
修复；训练normalizer身份仍未知，不替换部署统计。完整结果和八条件分母见
[后续实际归一化对照](D2P_FIXED_NORMALIZER_RESULT_MAIN_20261028_v1.md)。

仅两个已知根×renderer的四条件；164是分支数，不是独立根数。raw-box诊断
可能越过原logit球，不能当可部署候选。不得用真实secant修正选择器或事后选h。
训练run→dataset→normalizer身份链仍未建立：当前常数一致不等于训练血统。
本次只读访问官方HF API失败（Network unreachable），镜像API返回403，
不能把远端不可达写成上游没有训练材料，更不能自签认证或替换统计。
P1/P2/P3科学准入仍false，未证明理论不可能，也未完成历史算法根治。
