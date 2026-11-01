# Rank-reserve CUDA 执行预声明

这是已封存 queue SHA `2f39483bceb87e638f74b2da170db8cf7ddbf6a1050df4137bbbcf9dde37ed78` 的执行实现。
不改该规则/候选顺序或任何旧source/results。只用已知两根作工程校准，不作fresh确认或P1审计。

固定U,D_rank,S_rank,N；保底5，总20，3候选/block；搜索与回放全H2。
每臂60候选=120predict，两个根各whole/segmented独立回放=8predict。
全部4臂512predict；共享原零动作nominal+fresh全图joint J共8predict，2J、384方向、384grad API、384raw callback、2SVD。
fresh信号必须精确重现seal里的两w，否则失败并保留，不重排或用旧信号硬跑。
Q、goal、root、bank(0zero/1GN/2D2P/3..44固定随机尾)、normalizer/float32/tanh/包域不变。
只在paid30/60候选报已付费best-so-far，稳定(cost,index)选优；零收益、同winner均保留。

真实predict API与action_encoder/predictor/pred_proj分别预dispatch计数，输入/输出全精度保存。
predictor batch samples按实际输入shape[0]求和；context tokens按shape[0]*shape[1]另列；
pred_proj是flatten后的行数，不能冒充predictor samples。输入shape在module prehook观测，不从API数推断。
模型全eval/frozen，主机cuda:0，SDPA MATH，TF32 false；CPU神经、训练、去重、重试均禁止。
每资源attempt/return保存monotonic时间；这些是嵌套inclusive时间，不可相加当FLOPs。
encoding新0；历史8encoder/8projector、anchors288predict及完整旧AD receipts随prepare保留。
shared成本实际只付一次；独立部署U/D/S需136predict+AD/SVD、N128predict，不能称总资源匹配。

600s wall(含验证/加载)、3GiB allocated/4GiB reserved、16MiB output、1MiB final reserve、200MiB diskfloor固定。
超限保留partial，首错/后续错误分列，未派发槽仍在八分母，不自动重跑、不加预算/限额。
同source/models/hash在前后核验；所有全history按NumPy float32逐包重算，无容差放宽。
旧Native与Store仅只读复用；新scope caps、panel、schema、saved consumer独立，不修改旧globals。

执行顺序固定：RAM tests -> prepare/source+tests+doc+input SHA -> verify -> 主机CUDA一次 -> saved consume。
saved consume重算全部候选Q/稳定winner/信号与J、8个selection seals、全部520 native输入/输出链和scope账。
不加载模型、不读physical/outcome、不补尾部。report需外部SHA且exact artifact census。
原signal记录/J不能代替本轮fresh AD；旧缓存缺30条新tail槽，本轮全4臂重新付费，不只补tail。

主比较模型Q的D-U，次D-S/D-N，逐根和均值全保留；这是机制终点而非真实物理收益。
P1/freshness/P3、independent confirmation、compute/wall优势全false。
真正native/RGB收益与其消费者、认证fresh源及P1准入仍另缺，绝不由本执行打开科学gate。
