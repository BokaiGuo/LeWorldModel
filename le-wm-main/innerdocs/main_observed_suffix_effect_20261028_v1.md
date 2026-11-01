# 真实中点后缀重编译：MAIN 独立数值复算

源 `experiments/spectral_reachability/main_observed_suffix_effect_20261028_v1.py`。
先重新运行完整 upstream compiler.consume，并逐字节比对已持久 receipt，
再从保存的全部九个终态上下文独立 float64 重算原 goal192 的 meanQ。
核 stable(Q,index) 选择、原 prefix5/incumbent float32 bits、新后缀及新上下文
和新分数不可拆分。十项 RAM 回归为纯算术，不加载模型或物理环境。

实际结果 `results/main_observed_suffix_effect_20261028_v1_MAIN01.json`：
SHA `c99c5ff5fa3c11544568d8beebe9073d7cca7d2246ef5b786fa8f04c15217904`。
四条件 modelQ 变化 −.016481974/−.035132153/−.013381888/−.013975304，
全部选 iterate8，后缀确实改变、prefix bitwise 不变。仅两个 known roots，
没有 fresh 确认、没有新物理或真实终态编码；不能据此宣称控制收益或 P1。
