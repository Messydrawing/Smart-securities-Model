import time
from datetime import datetime, time as dtime,timedelta
from collections import defaultdict
from eastmoney_api import EastMoneyAPI
from tech_analysis import TechnicalAnalyzer
from notification import send_email
from Finder import train_initial_weights  # 导入训练权重的函数

class StockMonitor:
    def __init__(self, stock_list, strategy, email_addr, initial_positions=None):
        self.api = EastMoneyAPI()
        self.stock_list = stock_list      # dict: {stock_name: stock_code}
        self.strategy = strategy          # dict: {stock_name: {各参数}}
        self.email_addr = email_addr
        self.last_action = {}
        self.last_notify_time = {}
        self.positions = initial_positions if initial_positions else {}
        self.daily_data_cache = {}
        self.peak_prices = defaultdict(float)  # 持仓期间的最高价
        self.model = None

        # 初始权重训练（假设第一次使用前已获得初步的权重参数）
        self.initial_weights = train_initial_weights(stock_list)

    def monitor(self, interval=30):
        """
        每隔 interval 秒轮询一次行情数据，根据指标和公式判别逻辑生成通知信息。
        """
        initial_prices = {}
        while True:
            now = datetime.now()
            current_time = now.time()

            if dtime(9, 0) <= current_time < dtime(12, 0):
                period = "morning"
            elif dtime(12, 0) <= current_time < dtime(15, 0):
                period = "afternoon"
            else:
                period = "off_hours"

            all_notify_msg = ""
            for stock_name, stock_code in self.stock_list.items():
                real_data = self.api.get_stock_data(stock_code)
                if not real_data:
                    print(f"无法获取 {stock_name}({stock_code}) 的实时数据...")
                    continue

                price = real_data["current_price"]
                if stock_name not in initial_prices:
                    initial_prices[stock_name] = price

                df = self.api.get_kline_data(stock_code, klt=101, fqt=0, num=60)
                if df is None or len(df) < 20:
                    print(f"数据不足，跳过 {stock_name}({stock_code}) 指标计算...")
                    continue

                df = TechnicalAnalyzer.calculate_all_indicators(df)
                last_row = df.iloc[-1]
                ma5 = float(last_row.get("ma5", price))
                ma20 = float(last_row.get("ma20", price))
                macd = float(last_row.get("macd", 0.0))
                rsi = float(last_row.get("rsi", 50.0))
                bb_upper = float(last_row.get("bb_upper", price * 1.02))
                bb_lower = float(last_row.get("bb_lower", price * 0.98))

                base_strategy = self.strategy.get(stock_name, {})
                stop_loss_ratio = base_strategy.get("stop_loss_percent", 0.98)
                fixed_profit_ratio = base_strategy.get("fixed_profit_ratio", 1.10)
                trailing_stop_ratio = base_strategy.get("trailing_stop_ratio", 0.05)

                if stock_name not in self.positions:
                    self.positions[stock_name] = 0
                has_position = (self.positions[stock_name] > 0)
                already_bought_today = False

                if stock_name not in self.peak_prices:
                    self.peak_prices[stock_name] = 0.0

                if has_position and "initial_price" in base_strategy:
                    initial_price = base_strategy["initial_price"]
                else:
                    initial_price = initial_prices[stock_name]

                # 获取公式计算建议
                rule_action = self.calc_weighted_action(
                    price=price,
                    ma5=ma5,
                    ma20=ma20,
                    macd=macd,
                    rsi=rsi,
                    bb_lower=bb_lower,
                    bb_upper=bb_upper,
                    initial_price=initial_price,
                    stop_loss_ratio=stop_loss_ratio,
                    fixed_profit_ratio=fixed_profit_ratio,
                    trailing_stop_ratio=trailing_stop_ratio,
                    peak_price=self.peak_prices[stock_name],
                    has_position=has_position,
                    already_bought_today=already_bought_today,
                    total_score_threshold_sell=-0.3,
                    total_score_threshold_buy=0.3
                )

                if period == "morning" and rule_action in ["buy", "sell"]:
                    rule_action = "watch"

                if has_position and price > self.peak_prices[stock_name]:
                    self.peak_prices[stock_name] = price

                final_msg = f"公式建议: {rule_action}"
                detail_msg = f"MA5: {ma5:.2f}, MA20: {ma20:.2f}\nMACD: {macd:.2f}\nRSI: {rsi:.2f}"
                notify_msg = self._prepare_notify_msg(stock_name, price, now, final_msg, detail_msg)
                if notify_msg:
                    all_notify_msg += notify_msg + "\n\n"

                if rule_action == "buy" and not has_position:
                    self.positions[stock_name] = 1000
                    base_strategy["initial_price"] = price
                    self.peak_prices[stock_name] = price
                    already_bought_today = True
                elif rule_action in ["sell", "stop_loss", "take_profit"]:
                    if has_position and not already_bought_today:
                        self.positions[stock_name] = 0
                        base_strategy["initial_price"] = 0.0
                        self.peak_prices[stock_name] = 0.0

            if all_notify_msg:
                subject = "股票监控通知"
                send_email(self.email_addr, subject, all_notify_msg)
            time.sleep(interval)

    def calc_weighted_action(self, price, ma5, ma20, macd, rsi, bb_lower, bb_upper, initial_price,
                             stop_loss_ratio, fixed_profit_ratio, trailing_stop_ratio, peak_price,
                             has_position, already_bought_today, total_score_threshold_sell, total_score_threshold_buy):
        # 使用初步的权重，训练模型后可能会更新权重
        weights = self.initial_weights  # 使用初始化训练权重

        s_ma = self.score_ma(ma5, ma20)
        s_macd = self.score_macd(macd)
        s_rsi = self.score_rsi(rsi)
        s_boll = self.score_boll(price, bb_lower, bb_upper)

        total_score = (s_ma * weights['ma5'] +
                       s_macd * weights['macd'] +
                       s_rsi * weights['rsi'] +
                       s_boll * weights['bb_lower'])

        if has_position:
            if total_score < total_score_threshold_sell and not already_bought_today:
                return "sell"
            else:
                return "hold"

        if total_score > total_score_threshold_buy:
            return "buy"
        else:
            return "watch"

    def score_ma(self, ma5, ma20):
        if ma5 > ma20:
            return 1
        elif ma5 < ma20:
            return -1
        return 0

    def score_macd(self, macd):
        if macd > 0:
            return 1
        elif macd < 0:
            return -1
        return 0

    def score_rsi(self, rsi):
        if rsi < 30:
            return 1
        elif rsi > 70:
            return -1
        return 0

    def score_boll(self, price, lower_val, upper_val):
        if price <= lower_val * 1.01:
            return 1
        elif price >= upper_val * 0.99:
            return -1
        return 0

    def _prepare_notify_msg(self, stock_name, price, now, final_msg, detail_msg=""):
        last_time = self.last_notify_time.get(stock_name)
        if last_time and (now - last_time < timedelta(minutes=30)) and final_msg == self.last_action.get(stock_name):
            return ""
        msg = f"[{stock_name}] 当前价：{price:.2f}，建议：{final_msg}（{now.strftime('%Y-%m-%d %H:%M:%S')}）"
        if detail_msg:
            msg += "\n" + detail_msg
        self.last_notify_time[stock_name] = now
        self.last_action[stock_name] = final_msg
        return msg
