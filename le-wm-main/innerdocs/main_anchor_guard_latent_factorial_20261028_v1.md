# 真实终态后的独立三层对照

MAIN不导入任何实验/模型/环境模块，直接从三组原报告的gzip NPY读取native192，
核对外部报告SHA、每个NPY文件SHA、完整float32/shape/有限值/无尾随字节，
以原goal192重算真实Q。再从32个原cell预测向量独立重算预测Q，不只相信summary。

逐cell用已独立验算的物理2×2动作来源，核对task/root/condition/full10 rawbits，
保持32分母。control/incumbent的strict标量来自固定已接受回执；zero来自新strict。
本工具重新算原数组，不重新做神经计算或全上游审计。

比较anchor与guard的同分支预测差、真实编码差、真实物理差。严格反转只计
乘积负数；ties单独留在全表而不计反转。不得将32cell或8branch当独立根，
也不把一个方向正确或缺失反转当可靠泛化。所有P1/P2准入false。

实际MAIN01：预测到真实编码的anchor排序反转2/8；无guard的编码到jointXY
反转3/8，有guard4/8。guard-zero改变的branch4预测和实际Q同降，但angle、
velocity和原native distance均损害。它们定位两层仍有缺口，不是唯一根因证明，
不能用事后选择factor或真实outcome重新决定动作。新9项测试是合成/NPY接线
回归，不是独立科学实验。
