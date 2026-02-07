import requests
import pandas as pd

class EastMoneyAPI:
    # 实时数据接口
    REAL_TIME_URL = "https://push2.eastmoney.com/api/qt/stock/get"
    # 历史K线数据接口
    KLINE_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    # A股股票列表接口（通过东财获取所有A股的股票编号）
    STOCK_LIST_URL = "https://push2.eastmoney.com/api/qt/clist/get"

    def __init__(self):
        self.session = requests.Session()

    def get_stock_data(self, stock_code):
        """
        获取实时股票数据
        返回字典：{'current_price': float, 'high_price': float, 'low_price': float, 'open_price': float, 'prev_close': float}
        """
        if stock_code.startswith("6"):
            secid = f"1.{stock_code}"
        else:
            secid = f"0.{stock_code}"

        params = {
            "secid": secid,
            "fields": "f43,f44,f45,f46,f47"
        }
        try:
            response = self.session.get(self.REAL_TIME_URL, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()
            if "data" in data and data["data"]:
                return {
                    "current_price": data["data"]["f43"] / 100,
                    "high_price": data["data"]["f44"] / 100,
                    "low_price": data["data"]["f45"] / 100,
                    "open_price": data["data"]["f46"] / 100,
                    "prev_close": data["data"]["f47"] / 100
                }
            else:
                return None
        except Exception as e:
            print(f"获取实时数据失败: {e}")
            return None

    def get_kline_data(self, stock_code, klt=101, fqt=0, num=60):
        """
        获取历史K线数据，返回pandas DataFrame
        DataFrame包含字段：date, open, close, high, low, volume
        """
        if stock_code.startswith("6"):
            secid = f"1.{stock_code}"
        else:
            secid = f"0.{stock_code}"

        params = {
            "fields1": "f1,f2,f3,f4,f5,f6",
            "fields2": "f51,f52,f53,f54,f55,f56,f57,f58",
            "secid": secid,
            "klt": klt,
            "fqt": fqt,
            "end": "20500000",
            "lmt": num,
            "beg": "0"
        }
        try:
            response = self.session.get(self.KLINE_URL, params=params, timeout=5)
            response.raise_for_status()
            js = response.json()
            if "data" not in js or not js["data"]:
                return None

            klines = js["data"]["klines"]
            records = []
            for k in klines:
                items = k.split(',')
                record = {
                    "date": items[0],
                    "open": float(items[1]),
                    "close": float(items[2]),
                    "high": float(items[3]),
                    "low": float(items[4]),
                    "volume": float(items[5])
                }
                records.append(record)
            df = pd.DataFrame(records)
            df.sort_values("date", inplace=True)
            df.reset_index(drop=True, inplace=True)
            return df
        except Exception as e:
            print(f"获取历史K线数据失败: {e}")
            return None

    def get_all_stock_codes(self):
        """
        获取所有A股股票列表（股票代码及名称）
        返回字典：{股票名称: 股票代码}
        """
        stock_list = {}
        page = 1
        while True:
            params = {
                "pn": page,
                "pz": 1000,  # 每页返回1000条
                "po": 1,      # 排序方式，1表示升序
                "np": 1,      # 数据页数
                "fid": "f3",  # 股票代码字段
                "fltt": 2,    # 股票数据类型
                "invt": 2,    # 是否返回股票的代码
                "fs": "m:1+t:2",  # 股票类型，m:1表示A股
            }
            try:
                response = self.session.get(self.STOCK_LIST_URL, params=params, timeout=5)
                response.raise_for_status()
                data = response.json()
                stocks = data['data']['diff']
                for stock in stocks:
                    stock_list[stock['f14']] = stock['f12']  # 股票名: 股票代码
                if len(stocks) < 1000:
                    break  # 如果返回的数据少于1000条，说明已经是最后一页
                page += 1  # 下一页
            except Exception as e:
                print(f"获取股票列表失败: {e}")
                break
        return stock_list
