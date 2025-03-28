from monitor import StockMonitor
from Finder import train_initial_weights
from eastmoney_api import EastMoneyAPI

# 股票列表，可以自定义或从文件加载
stock_list = {
    "京粮控股": "000505",
    "中国中铁": "601390",
    "深粮控股": "000019",
    "云南白药":"000538",
    "广弘控股":"000529",
    "佛山照明":"000541",
    "中国建筑":"601668",
    "紫光股份":"000938",
    "比亚迪":"002594",
    "粤高速A":"000429",
    "冀东装备":"000856",
    "紫光国微":"002049",
    "1":"000010",
    "2": "000011",
    "3": "000019",
    "4": "000021",
    "5": "000027",
    "6": "000049",
    "7": "000066",
    "8": "000068",
    "9": "000428",
    "10": "000430",
    "11": "000922",
    "12": "000737",
    "13": "000655",
    "14": "000708",
    "15": "000686",
    "16": "000690",
    "17": "000700",
    "18": "000565",
    "19": "000554",
    "20": "000550",
    "21": "000545",
    "22": "000539",
    "23": "000536",
    "24": "000530",
    "25": "000525",
    "26": "000970",
    "27": "000968",
    "28": "000966",
    "29": "000980",
    "30": "000931",
    "31": "002151",
    "32": "002163",
    "33": "002166",
}

# 策略初始化（这些值可以调整）
strategy = {
    name: {
        "stop_loss_percent": 0.97,
        "fixed_profit_ratio": 1.08,
        "trailing_stop_ratio": 0.04
    } for name in stock_list
}

# 第一步：初步训练每个股票并生成特征权重
initial_weights = train_initial_weights(stock_list)

# 将初始训练得到的权重更新到 strategy 中
for stock_name, weights in initial_weights.items():
    if stock_name in strategy:
        strategy[stock_name]["weights"] = weights.tolist()  # 存为list以便JSON兼容

# 第二步：开始监控并动态迭代模型（使用策略+权重）
monitor = StockMonitor(stock_list=stock_list, strategy=strategy, email_addr="???@xx.com")
monitor.monitor(interval=60)  # 每60秒轮询一次
